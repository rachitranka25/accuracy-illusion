from oracle import paths
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
HZ=[('5m',1),('15m',3),('30m',6),('60m',12)]

for name,f in [('BANKNIFTY',f'{paths.CACHE}/BANKNIFTY_5m_long.csv'),('NIFTY50',f'{paths.CACHE}/NIFTY50_5m_long.csv')]:
    df=pd.read_csv(f,index_col=0,parse_dates=True); df['d']=df.index.date; c=df['Close']
    ret=c.pct_change().abs()
    dates=np.array(pd.to_datetime(df.index).normalize().view('int64'))
    print(f'\n================  {name}  — VOLATILITY forecast  ================')
    print('| TF | ✓ Green | ✗ Red | Accuracy |')
    print('|----|---------|-------|----------|')
    for hn,k in HZ:
        # recent vol (past k bars) and future vol (next k bars) = sum of |returns|
        past=ret.rolling(k).sum()
        fut=ret.shift(-k).rolling(k).sum().shift(-(0))    # sum over next k
        fut=ret.rolling(k).sum().shift(-k)                # next-k-bar realized vol
        thr=past.rolling(200).median()                    # causal threshold
        pred=(past>thr).astype(float)                     # predict HIGH if recent vol high
        actual=(fut>thr).astype(float)                    # was next vol actually high?
        n=len(df); g=r=0
        pv=pred.values; av=actual.values
        for i in range(n-k):
            if not (np.isfinite(pv[i]) and np.isfinite(av[i])): continue
            if dates[i]!=dates[i+k]: continue             # same-day
            hit=pv[i]==av[i]; g+=hit; r+=(not hit)
        tot=g+r; acc=f'{100*g//tot}%' if tot else '-'
        print(f'| {hn} | {g:,} | {r:,} | {acc} |')
