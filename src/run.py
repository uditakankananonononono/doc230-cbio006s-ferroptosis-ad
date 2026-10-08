import gzip,io,re,json,numpy as np,pandas as pd,requests
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(0)
lines=gzip.open('/tmp/ad/gse5281.gz','rt').read().split('\n')
def meta(k): return [l for l in lines if l.startswith(k)]
title=[x.strip('"') for x in meta('!Sample_title')[0].split('\t')[1:]]
ch=[[x.strip('" ') for x in l.split('\t')[1:]] for l in meta('!Sample_characteristics_ch1')]
def pick(prefix): 
    for r in ch:
        if r[0].startswith(prefix): return [x.split(': ',1)[1] for x in r]
age=np.array([float(re.findall(r'[\d.]+',a)[0]) if re.findall(r'[\d.]+',a) else np.nan for a in pick('Age')]); sex=np.array([1 if s.strip()=='male' else 0 for s in pick('Sex')])
region=np.array([t.split()[0].split('_')[0] for t in title]); ad=np.array([0 if 'control' in t else 1 for t in title])
i0=[i for i,l in enumerate(lines) if l.startswith('!series_matrix_table_begin')][0]; i1=[i for i,l in enumerate(lines) if l.startswith('!series_matrix_table_end')][0]
D=pd.read_csv(io.StringIO('\n'.join(lines[i0+1:i1])),sep='\t',index_col=0); D.columns=[c.strip('"') for c in D.columns]; D.index=[str(x).strip('"') for x in D.index]
if D.values.max()>40: D=np.log2(D.clip(lower=1))
mp=pd.read_csv('/tmp/ad/gpl570_map.csv',dtype=str).dropna(subset=['ENTREZ_GENE_ID']); mp=mp[~mp.ENTREZ_GENE_ID.str.contains('///')]; mp=mp[mp.ID.isin(D.index)]
def gene_matrix(how):
    X=D.loc[mp.ID]; X.index=mp.ENTREZ_GENE_ID.values
    if how=='mean': return X.groupby(level=0).mean()
    m=X.mean(1); X=X.assign(_m=m.values); X=X.sort_values('_m',ascending=False); X=X[~X.index.duplicated(keep='first')].drop(columns='_m'); return X
G=gene_matrix('mean'); G=G[(G.mean(1)>np.log2(50))|(G.mean(1)>5)]
def kegg(p): return set(re.findall(r'hsa:(\d+)',requests.get(f'https://rest.kegg.jp/link/hsa/{p}',timeout=60).text))
FER,APO,OXP=kegg('hsa04216'),kegg('hsa04210'),kegg('hsa00190'); print(len(FER),len(APO),len(OXP),flush=True)
sym=pd.read_csv('/tmp/ad/gpl570_map.csv',dtype=str).dropna(subset=['ENTREZ_GENE_ID']).drop_duplicates('ENTREZ_GENE_ID').set_index('ENTREZ_GENE_ID')['Gene Symbol'].to_dict()
n=len(title); reg=pd.get_dummies(region,drop_first=True).values.astype(float); ok=~np.isnan(age)
def tstats(Gm,adj=True,cols=None):
    idx=np.where(ok)[0]; Y=Gm.values[:,idx].T; z=[np.ones(len(idx)),ad[idx].astype(float)]
    Z=np.column_stack(z+([reg[idx],(age[idx]-age[idx].mean())/age[idx].std(),sex[idx]] if adj else [reg[idx]]))
    B,_,_,_=np.linalg.lstsq(Z,Y,rcond=None); res=Y-Z@B; df=len(idx)-Z.shape[1]; s2=(res**2).sum(0)/df; XtXi=np.linalg.inv(Z.T@Z); se=np.sqrt(s2*XtXi[1,1]); return pd.Series(B[1]/se,index=Gm.index)
T=tstats(G); Tun=tstats(G,adj=False)
def setstat(T,S,B=2000,seed=1):
    S=[g for g in S if g in T.index]; r=np.random.default_rng(seed); obs_a=np.abs(T[S]).mean(); obs_s=T[S].mean(); ar=[];sr=[]
    for _ in range(B): s=r.choice(len(T),len(S),replace=False); v=T.values[s]; ar.append(np.abs(v).mean()); sr.append(v.mean())
    ar=np.array(ar); sr=np.array(sr); return dict(n=len(S),mean_abs_t=round(float(obs_a),3),p_abs=round(float((ar>=obs_a).mean()),4),mean_t=round(float(obs_s),3),p_signed_two_sided=round(float(min(1,2*min((sr<=obs_s).mean(),(sr>=obs_s).mean()))),4))
R={'d1':dict(n=n,n_ad=int(ad.sum()),n_control=int((1-ad).sum()),by_region={r:[int(((region==r)&(ad==0)).sum()),int(((region==r)&(ad==1)).sum())] for r in sorted(set(region))},n_genes_expressed=len(G),ferroptosis_genes_found=len([g for g in FER if g in G.index]),ferroptosis_total=len(FER),age_mean_control=round(float(np.nanmean(age[ad==0])),1),age_mean_ad=round(float(np.nanmean(age[ad==1])),1),male_frac_control=round(float(sex[ad==0].mean()),2),male_frac_ad=round(float(sex[ad==1].mean()),2))}
R['d2_ferroptosis']=setstat(T,FER); R['d2_gate_p_abs_lt_0.05']=bool(R['d2_ferroptosis']['p_abs']<.05); print('d2',R['d2_ferroptosis'],flush=True)
keyg=['GPX4','SLC7A11','FTH1','FTL','TFRC','ACSL4','NCOA4','HMOX1','TP53','SLC40A1']; inv={v:k for k,v in sym.items()}
R['d3_per_gene_t']={g:round(float(T[inv[g]]),2) for g in keyg if g in inv and inv[g] in T.index}
R['d3_top_abs_t_ferroptosis']={sym.get(g,g):round(float(T[g]),2) for g in sorted([g for g in FER if g in T.index],key=lambda g:-abs(T[g]))[:8]}
reps=0;R['d4_per_region']={}
for r in sorted(set(region)):
    idx=np.where(region==r)[0]; Gr=G.iloc[:,idx]; adr=ad[idx]; tt=pd.Series(stats.ttest_ind(Gr.values[:,adr==1],Gr.values[:,adr==0],axis=1).statistic,index=G.index)
    s=setstat(tt,FER,B=1000); R['d4_per_region'][r]=s; reps+=s['p_abs']<.05
R['d4_regions_with_p_lt_0.05']=int(reps); R['d4_gate_ge_3_of_6']=bool(reps>=3); print('d4',R['d4_regions_with_p_lt_0.05'],flush=True)
def clf(S,labels,B=0):
    S=[g for g in S if g in G.index]; X=G.loc[S].values.T.copy()
    for r in set(region): m=region==r; X[m]=(X[m]-X[m].mean(0))/(X[m].std(0)+1e-6)
    pr=np.zeros(n)
    for r in set(region):
        te=region==r; tr=~te; pr[te]=LogisticRegression(C=.1,max_iter=500).fit(X[tr],labels[tr]).decision_function(X[te])
    return roc_auc_score(labels,pr)
a=clf(FER,ad); rnd=np.array([clf(list(rng.choice(G.index,len(FER),replace=False)),ad) for _ in range(200)])
R['d5']=dict(auroc=round(float(a),4),random_mean=round(float(rnd.mean()),4),random_p95=round(float(np.percentile(rnd,95)),4),frac_random_ge=round(float((rnd>=a).mean()),3)); R['d5_gate']=bool(a>=.65 and a>np.percentile(rnd,95)); print('d5',R['d5'],flush=True)
R['d6']=dict(age_vs_ad_spearman=[round(float(x),3) for x in stats.spearmanr(age[ok],ad[ok])],sex_vs_ad_chi2_p=round(float(stats.chi2_contingency(pd.crosstab(sex,ad))[1]),4),unadjusted=setstat(Tun,FER,B=1000),adjusted=R['d2_ferroptosis'])
ish=ad.copy(); 
for r in set(region): m=np.where(region==r)[0]; ish[m]=rng.permutation(ad[m])
sh=[]; 
for k in range(20):
    for r in set(region): m=np.where(region==r)[0]; ish[m]=rng.permutation(ad[m])
    sh.append(clf(FER,ish.copy()))
R['d7_shuffle_auroc_mean_sd']=[round(float(np.mean(sh)),4),round(float(np.std(sh)),4)]
R['d8']=dict(oxphos_positive_control=setstat(T,OXP),apoptosis=setstat(T,APO)); R['d8_positive_control_pass']=bool(R['d8']['oxphos_positive_control']['mean_t']<0 and R['d8']['oxphos_positive_control']['p_signed_two_sided']<.05); print('d8',R['d8'],flush=True)
G2=gene_matrix('maxprobe'); T2=tstats(G2.loc[[g for g in G.index if g in G2.index]]); R['d9_maxprobe']=setstat(T2,FER)
S=[g for g in FER if g in G.index]; loo={sym.get(g,g):round(float(a-clf([x for x in S if x!=g],ad)),4) for g in S}; R['d10_leave_one_gene_out_auroc_drop_top5']=dict(sorted(loo.items(),key=lambda x:-x[1])[:5])
json.dump(R,open('results/results.json','w'),indent=1,default=float); print(json.dumps(R,indent=1,default=float)[:3000])
