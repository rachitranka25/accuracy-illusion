from oracle import paths
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np

def run(name, D=3):
    o=pd.read_csv(f'{paths.CACHE}/{name}_options_eod.csv')
    o['XpryDt']=pd.to_datetime(o['XpryDt']); o['TradDt']=pd.to_datetime(o['TradDt'])
    res=[]
    for exp,grp in o.groupby('XpryDt'):
        tdays=sorted(grp['TradDt'].unique())
        if len(tdays)<=D: continue
        ref=tdays[-1-D]                       # D trading days before expiry
        day=grp[grp['TradDt']==ref]
        if day.empty: continue
        spot=day['UndrlygPric'].median()
        if not np.isfinite(spot): continue
        strikes=day['StrkPric'].unique()
        atm=strikes[np.argmin(np.abs(strikes-spot))]
        ce=day[(day.StrkPric==atm)&(day.OptnTp=='CE')]['ClsPric']
        pe=day[(day.StrkPric==atm)&(day.OptnTp=='PE')]['ClsPric']
        if ce.empty or pe.empty: continue
        premium=float(ce.iloc[0])+float(pe.iloc[0])
        # underlying at expiry
        exprow=grp[grp['TradDt']==exp]
        if exprow.empty: continue
        exp_spot=exprow['UndrlygPric'].median()
        if not np.isfinite(exp_spot): continue
        move=abs(exp_spot-atm)
        pnl=premium-move                       # SHORT straddle held to expiry
        res.append({'exp':exp,'premium':premium,'move':move,'pnl':pnl})
    R=pd.DataFrame(res).dropna().sort_values('exp')
    n=len(R)
    lot={'NIFTY':75,'BANKNIFTY':15}[name]
    cost_pts=premium_cost=0.10                 # ~10% of premium round-trip (retail options)
    R['net']=R['pnl']-R['premium']*premium_cost
    gross=R['pnl'].sum(); net=R['net'].sum(); hh=n//2
    def wr(x): return 100*(x>0).mean()
    print(f'### {name}  ({D} days-to-expiry entry, {n} weekly straddles) ###')
    print(f'  avg premium sold   : {R.premium.mean():.0f} pts')
    print(f'  avg actual move    : {R.move.mean():.0f} pts   (premium - move = edge)')
    print(f'  GROSS: win {wr(R.pnl):.0f}% | total {gross:+.0f} pts | avg {R.pnl.mean():+.1f}/wk | Rs {gross*lot:+,.0f}')
    print(f'    stability: h1 {wr(R.pnl.iloc[:hh]):.0f}% | h2 {wr(R.pnl.iloc[hh:]):.0f}%')
    print(f'  NET (after ~10% cost): win {wr(R.net):.0f}% | total {net:+.0f} pts | Rs {net*lot:+,.0f}')
    print(f'    worst single week  : {R.pnl.min():.0f} pts (the tail risk)')
    print()

for nm in ['NIFTY','BANKNIFTY']:
    run(nm, D=3)
    run(nm, D=1)
