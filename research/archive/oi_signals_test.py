from oracle import paths
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np

def maxpain(sub):
    # nearest-expiry options for the day: strike minimizing total writer payout
    ce=sub[sub.OptnTp=='CE'].groupby('StrkPric')['OpnIntrst'].sum()
    pe=sub[sub.OptnTp=='PE'].groupby('StrkPric')['OpnIntrst'].sum()
    strikes=sorted(set(ce.index)|set(pe.index))
    if len(strikes)<3: return np.nan
    best,bestv=None,1e18
    for K in strikes:
        pain=sum(ce.get(s,0)*max(0,K-s) for s in strikes)+sum(pe.get(s,0)*max(0,s-K) for s in strikes)
        if pain<bestv: bestv,best=pain,K
    return best

for name in ['NIFTY','BANKNIFTY']:
    opt=pd.read_csv(f'{paths.CACHE}/{name}_options_eod.csv')
    fut=pd.read_csv(f'{paths.CACHE}/{name}_futures_eod.csv')
    opt['XpryDt']=pd.to_datetime(opt['XpryDt']); opt['TradDt2']=pd.to_datetime(opt['TradDt'])
    # nearest expiry per day for max-pain
    near=opt[opt.XpryDt>=opt.TradDt2]
    nx=near.groupby('TradDt')['XpryDt'].min()
    mp={}
    for d,exp in nx.items():
        sub=near[(near.TradDt==d)&(near.XpryDt==exp)]
        mp[d]=maxpain(sub)
    ce=opt[opt.OptnTp=='CE']; pe=opt[opt.OptnTp=='PE']
    day=pd.DataFrame({
        'ce_oi':ce.groupby('TradDt')['OpnIntrst'].sum(),
        'pe_oi':pe.groupby('TradDt')['OpnIntrst'].sum(),
        'ce_doi':ce.groupby('TradDt')['ChngInOpnIntrst'].sum(),
        'pe_doi':pe.groupby('TradDt')['ChngInOpnIntrst'].sum(),
        'ce_vol':ce.groupby('TradDt')['TtlTradgVol'].sum(),
        'pe_vol':pe.groupby('TradDt')['TtlTradgVol'].sum(),
        'px':opt.groupby('TradDt')['UndrlygPric'].median(),
    }).dropna()
    day['maxpain']=pd.Series(mp)
    fut['XpryDt']=pd.to_datetime(fut['XpryDt']); fut['TradDt2']=pd.to_datetime(fut['TradDt'])
    ff=fut[fut.XpryDt>=fut.TradDt2].sort_values(['TradDt','XpryDt']).groupby('TradDt').first()
    day['fut_oi']=ff['OpnIntrst']; day['fut_doi']=ff['ChngInOpnIntrst']
    day=day.dropna().sort_index()
    day['pcr_oi']=day['pe_oi']/day['ce_oi']; day['pcr_vol']=day['pe_vol']/day['ce_vol']
    day['ret']=day['px'].pct_change()
    day['nxt']=day['px'].shift(-1)-day['px']
    day['pcr_ma']=day['pcr_oi'].rolling(20).mean()
    day=day.dropna(); n=len(day)

    def acc(sig):
        s=np.sign(sig); v=(s!=0)&(day['nxt']!=0)&np.isfinite(s)
        s=s[v]; mv=day['nxt'][v]
        hit=((s>0)&(mv>0))|((s<0)&(mv<0)); hh=len(hit)//2
        return 100*hit.mean(),100*hit.iloc[:hh].mean(),100*hit.iloc[hh:].mean(),int(v.sum())

    q_hi=day['pcr_oi'].quantile(0.8); q_lo=day['pcr_oi'].quantile(0.2)
    S={
     'PCR-OI contrarian': np.sign(day['pcr_oi']-day['pcr_oi'].median()),
     'PCR-OI vs 20d-MA': np.sign(day['pcr_oi']-day['pcr_ma']),
     'PCR extremes (only hi/lo)': np.where(day['pcr_oi']>=q_hi,1,np.where(day['pcr_oi']<=q_lo,-1,0)),
     'PCR-vol contrarian': np.sign(day['pcr_vol']-day['pcr_vol'].median()),
     'PCR change (dPCR)': np.sign(day['pcr_oi'].diff()),
     'Max-pain pull (spot->MP)': np.sign(day['maxpain']-day['px']),
     'Max-pain fade': -np.sign(day['maxpain']-day['px']),
     'Fut dOI sign': np.sign(day['fut_doi']),
     'Long-buildup cont (ret+ & dOI+)': np.where((day['ret']>0)&(day['fut_doi']>0),1,np.where((day['ret']<0)&(day['fut_doi']>0),-1,0)),
     'Buildup fade': np.where((day['ret']>0)&(day['fut_doi']>0),-1,np.where((day['ret']<0)&(day['fut_doi']>0),1,0)),
     'CE-PE dOI imbalance': np.sign(day['pe_doi']-day['ce_doi']),
     'CE-PE vol imbalance': np.sign(day['pe_vol']-day['ce_vol']),
     'Total OI change': np.sign((day['ce_oi']+day['pe_oi']).diff()),
     'Vol-PCR change': np.sign(day['pcr_vol'].diff()),
     'PCR-OI momentum': -np.sign(day['pcr_oi']-day['pcr_oi'].median()),
     'baseline: price momentum': np.sign(day['ret']),
     'baseline: always UP': pd.Series(1,index=day.index),
    }
    res=[]
    for sn,sig in S.items():
        a,a1,a2,nn=acc(pd.Series(np.asarray(sig,float),index=day.index))
        stable=(a1>=53 and a2>=53) or (a1<=47 and a2<=47)
        res.append((a,a1,a2,nn,sn,stable))
    res.sort(key=lambda x:-x[0])
    print('='*66); print(f'  {name}  — ALL OI signals, next-day ({n} days), ranked'); print('='*66)
    print(f'  {"signal":<34}full | h1 | h2   n')
    for a,a1,a2,nn,sn,st in res:
        tag='  ★STABLE' if st else ''
        print(f'  {sn:<34}{a:4.0f} |{a1:3.0f} |{a2:3.0f}  {nn}{tag}')
    print()
