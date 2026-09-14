from oracle import paths
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
HZ=[('5m',1),('15m',3),('30m',6),('60m',12)]

def rsi(c,n=14):
    d=c.diff();u=d.clip(lower=0).ewm(alpha=1/n).mean();dn=(-d.clip(upper=0)).ewm(alpha=1/n).mean()
    return 100-100/(1+u/dn.replace(0,np.nan))

def score(c,dates,mask_sig,k):
    n=len(c);s=np.sign(np.asarray(mask_sig,float));fwd=np.roll(c,-k);mv=fwd-c;sd=dates==np.roll(dates,-k)
    v=np.zeros(n,bool);v[:n-k]=True;v&=sd&(s!=0)&np.isfinite(s)&(mv!=0)
    hit=((s>0)==(mv>0));g=int((hit&v).sum());tot=int(v.sum());hh=n//2
    v1=v.copy();v1[hh:]=False;v2=v.copy();v2[:hh]=False
    a1=100*(hit&v1).sum()/max(1,v1.sum());a2=100*(hit&v2).sum()/max(1,v2.sum())
    return g,tot-g,a1,a2

for name,f in [('BANKNIFTY',f'{paths.CACHE}/BANKNIFTY_5m_long.csv'),('NIFTY50',f'{paths.CACHE}/NIFTY50_5m_long.csv')]:
    df=pd.read_csv(f,index_col=0,parse_dates=True);df['d']=df.index.date;c=df['Close']
    ema50=c.ewm(span=50).mean();tp=(df['High']+df['Low']+c)/3
    vwap=tp.groupby(df['d']).transform(lambda x:x.expanding().mean())
    dayopen=c.groupby(df['d']).transform('first');r3=c.pct_change(3);r12=c.pct_change(12);rs=rsi(c)
    sd=c.rolling(20).std();ma=c.rolling(20).mean();pctb=(c-(ma-2*sd))/((ma+2*sd)-(ma-2*sd))
    S=[-np.sign(r3),np.sign(c-ema50),np.sign(c-vwap),np.sign(c-dayopen),-np.sign(rs-50),np.sign(pctb-0.5)]
    vote=sum(np.sign(np.nan_to_num(s.values)) for s in S)
    mins=df.index.hour*60+df.index.minute
    prime=~((mins>=660)&(mins<=810))                 # skip 11:00-13:30 dead zone
    strong_trend=(r12.abs()>=r12.abs().quantile(0.5)).values
    dates=np.array(pd.to_datetime(df.index).normalize().view('int64'));cc=c.values

    print(f'\n================  {name}  ================')
    # ---- TABLE A: standard (all forecasts, ensemble fires) ----
    print('TABLE A — all forecasts')
    print('| TF | ✓ Green | ✗ Red | Accuracy |')
    print('|----|---------|-------|----------|')
    sigA=np.where(np.abs(vote)>=1,np.sign(vote),0)
    for hn,k in HZ:
        g,r,a1,a2=score(cc,dates,sigA,k);tot=g+r
        print(f'| {hn} | {g:,} | {r:,} | {100*g//tot}% (h1 {a1:.0f}|h2 {a2:.0f}) |')

    # ---- TABLE B: STRONG — signals agree (>=4/6) + prime + strong-trend ----
    print('TABLE B — STRONG (4+/6 signals agree, prime hours, strong trend)')
    print('| TF | ✓ Green | ✗ Red | Accuracy |')
    print('|----|---------|-------|----------|')
    sigB=np.where((np.abs(vote)>=4)&prime&strong_trend,np.sign(vote),0)
    for hn,k in HZ:
        g,r,a1,a2=score(cc,dates,sigB,k);tot=g+r
        acc=f'{100*g//tot}% (h1 {a1:.0f}|h2 {a2:.0f})' if tot else '-'
        print(f'| {hn} | {g:,} | {r:,} | {acc} |')
