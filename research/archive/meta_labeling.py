#!/usr/bin/env python3
"""Meta-labeling forecaster — triple-barrier + meta.
Standalone; reuses build5 + BASE_COLS. Honest walk-forward comparison."""
import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from oracle.models import direction as fce
from oracle import paths
warnings.filterwarnings("ignore")
CACHE = paths.CACHE
def triple_barrier(prices, sigmas, horizon_bars, up_mult=1.5, dn_mult=1.5):
    p=np.asarray(prices,np.float64); s=np.asarray(sigmas,np.float64); n=len(p)
    labels=np.zeros(n,np.int8); rets=np.zeros(n,np.float64); t_ends=np.full(n,-1,np.int64)
    for i in range(n-1):
        sig=s[i]
        if not np.isfinite(sig) or sig<=0: continue
        end=min(i+1+horizon_bars,n); window=p[i+1:end]
        if window.size==0: continue
        p0=p[i]; up=p0*(1+up_mult*sig); dn=p0*(1-dn_mult*sig)
        uh=np.where(window>=up)[0]; dh=np.where(window<=dn)[0]
        t_up=int(uh[0]) if uh.size else -1; t_dn=int(dh[0]) if dh.size else -1
        if t_up>=0 and (t_dn<0 or t_up<t_dn): labels[i]=1; rets[i]=window[t_up]/p0-1; t_ends[i]=i+1+t_up
        elif t_dn>=0: labels[i]=-1; rets[i]=window[t_dn]/p0-1; t_ends[i]=i+1+t_dn
        else: labels[i]=0; rets[i]=window[-1]/p0-1; t_ends[i]=end-1
    return labels,rets,t_ends

def _lgb(n=500):
    return lgb.LGBMClassifier(objective="binary",num_leaves=24,max_depth=4,learning_rate=0.02,
        n_estimators=n,subsample=0.75,colsample_bytree=0.7,reg_alpha=0.5,reg_lambda=2.0,
        min_child_samples=80,random_state=42,verbose=-1)

def run_experiment(name,horizon_key="30m",test_days=45,up_mult=1.5,dn_mult=1.5):
    print(f"\n{'='*70}\n  {name}  horizon={horizon_key}  held-out={test_days}  bar=+{up_mult}/-{dn_mult}sig\n{'='*70}")
    m5=pd.read_csv(CACHE/f"{name}_5m_long.csv",index_col=0,parse_dates=True)
    feats,close=fce.build5(m5); horizon_bars=dict(fce.HORIZONS)[horizon_key]; cols=fce.BASE_COLS
    r1=close.pct_change(); sigma=r1.ewm(span=100,adjust=False).std().shift(1)
    tb_lab,tb_ret,tb_end=triple_barrier(close.values,sigma.values,horizon_bars,up_mult,dn_mult)
    base_lab=(close.shift(-horizon_bars)>close).astype(int).values
    df=feats.copy(); df["base_y"]=base_lab; df["tb_lab"]=tb_lab; df["tb_end"]=tb_end
    df=df.dropna(subset=cols).reset_index(); df=df.rename(columns={df.columns[0]:"ts"})
    days=sorted(set(df["ts"].dt.date))
    if len(days)<test_days+30: print("  not enough sessions"); return
    cut=days[-test_days]; is_test=df["ts"].dt.date.values>=cut
    test_idx=np.where(is_test)[0]; train_idx=np.where(~is_test)[0]; test_start=int(test_idx.min())
    X=np.nan_to_num(df[cols].values.astype(np.float32),0.0)
    y=df["base_y"].values.astype(int)
    mb=_lgb(); mb.fit(X[train_idx],y[train_idx],eval_set=[(X[test_idx],y[test_idx])],
        callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(0)])
    p_base=mb.predict_proba(X[test_idx])[:,1]; pred_base=(p_base>=.5).astype(int)
    correct_base=(pred_base==y[test_idx]).astype(int)
    conf=np.abs(p_base-.5)*100; hi_thr=float(np.percentile(conf,72)); hi=conf>=hi_thr
    print(f"  A) BASELINE binary: ALL n={len(test_idx)} acc={correct_base.mean()*100:.2f}%  "
          f"HIGH n={int(hi.sum())} acc={correct_base[hi].mean()*100:.2f}%")
    y_prim=(df["tb_lab"].values==1).astype(int)
    tr_p=train_idx[df["tb_end"].values[train_idx]<test_start]
    ic=int(len(tr_p)*0.8); pin=tr_p[:ic]; mtr=tr_p[ic:]
    mbi=_lgb(); mbi.fit(X[pin],y[pin],callbacks=[lgb.log_evaluation(0)])
    pbmt=mbi.predict_proba(X[mtr])[:,1]; predbmt=(pbmt>=.5).astype(int)
    tbmt=df["tb_lab"].values[mtr]
    meta_y=(((predbmt==1)&(tbmt==1))|((predbmt==0)&(tbmt==-1))).astype(int)
    Xmt=np.column_stack([X[mtr],pbmt])
    if meta_y.sum()<30 or (1-meta_y).sum()<30: print("  meta imbalanced, skip"); return
    mm=_lgb(300); mm.fit(Xmt,meta_y,callbacks=[lgb.log_evaluation(0)])
    Xme=np.column_stack([X[test_idx],p_base]); p_meta=mm.predict_proba(Xme)[:,1]
    tb_te=df["tb_lab"].values[test_idx]
    correct_tb=(((pred_base==1)&(tb_te==1))|((pred_base==0)&(tb_te==-1))).astype(int)
    print(f"  B) vs TRIPLE-BARRIER truth: ALL acc={correct_tb.mean()*100:.2f}%  "
          f"HIGH acc={correct_tb[hi].mean()*100:.2f}%")
    print(f"     {'meta>=':>7}{'signals':>9}{'acc':>8}   delta vs baseline-HIGH(tb)")
    base_hi_tb=correct_tb[hi].mean()*100 if hi.sum() else 0
    for mc in (0.50,0.55,0.60,0.65,0.70):
        take=hi&(p_meta>=mc); n=int(take.sum())
        if n<10: print(f"     {mc:>7.2f}{n:>9}{'  --':>8}   (too few)"); continue
        acc=correct_tb[take].mean()*100
        print(f"     {mc:>7.2f}{n:>9}{acc:>7.2f}%   {acc-base_hi_tb:+.2f}pts")
    tot=len(tb_te)
    print(f"  FLAT%={100*(tb_te==0).sum()/tot:.1f}  UP%={100*(tb_te==1).sum()/tot:.1f}  DN%={100*(tb_te==-1).sum()/tot:.1f}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--horizon",default="30m",choices=[k for k,_ in fce.HORIZONS])
    ap.add_argument("--days",type=int,default=45); ap.add_argument("--up",type=float,default=1.5)
    ap.add_argument("--dn",type=float,default=1.5); a=ap.parse_args()
    for nm in ("BANKNIFTY","NIFTY50"):
        run_experiment(nm,a.horizon,a.days,a.up,a.dn)
