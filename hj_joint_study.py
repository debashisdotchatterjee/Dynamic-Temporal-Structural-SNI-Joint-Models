#!/usr/bin/env python3
"""Landmark SNI temporal-structural joint model: reproducible experiment suite.
Usage (Colab): %run hj_joint_study.py  OR  !python hj_joint_study.py
The script downloads the public Mayo PBC sequential dataset, fits proper marginal
joint likelihoods, runs simulation scenarios, creates reports, and ZIPs everything.
"""
from __future__ import annotations
import os, sys, json, hashlib, time, traceback, warnings, shutil, zipfile
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import ndtr, ndtri, gammaln
from scipy.stats import gamma as gamma_dist
from numpy.polynomial.legendre import leggauss
from sklearn.model_selection import train_test_split
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm=lambda x,**kw:x
import torch

# ------------------- USER SETTINGS -------------------
SEED = int(os.environ.get('HJ_SEED', '20261103'))
LANDMARK = float(os.environ.get('HJ_LANDMARK', '2.0')) # years, at-risk subjects only
SIM_REPS = int(os.environ.get('HJ_SIM_REPS', '6'))      # raise to 30+ for publication
SIM_N = int(os.environ.get('HJ_SIM_N', '200'))
REAL_REPEATS = int(os.environ.get('HJ_REAL_REPEATS', '2')) # raise to 5 for publication
MAX_ITER = int(os.environ.get('HJ_MAX_ITER', '115'))  # raise to 250 for publication
ROBUST_QUAD = os.environ.get('HJ_ROBUST_QUAD', '4,4,3')
OUT_DIR = Path(os.environ.get('HJ_OUT_DIR', '/content/HJ_structural_joint_outputs' if 'google.colab' in sys.modules else 'HJ_structural_joint_outputs'))
HORIZONS = [2.0, 4.0] # years after landmark
MODELS = ['Shared-Gaussian', 'Flat-Gaussian', 'Decay-Gaussian', 'Decay-SNI']
# Good defaults are quick diagnostics, not high-powered definitive experiments.
DO_SENSITIVITY = True
DF_B, DF_E = 6.0, 6.0 # fixed, separately distributed U and V for SNI
np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
                     'savefig.dpi': 170, 'figure.dpi': 100, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.fontsize': 8})

BASE_URLS = [
  'https://vincentarelbundock.github.io/Rdatasets/csv/survival/pbcseq.csv',
  'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/pbcseq.csv',
  'https://raw.githubusercontent.com/TheAxeC/cirrhosis-competing-risks/main/data/pbcseq.csv',
]
NAMES = ['beta0','beta_time','beta_sex','beta_trt','alpha0','alpha_age','alpha_sex',
         'alpha_trt','nu','log_se','log_sT','log_sd0','log_sd1','raw_corr',
         'raw_rho','log_lambda','skew0','skew1']
IX = {n:i for i,n in enumerate(NAMES)}


def dirs():
    for sub in ['figures','tables','data','diagnostics','logs']:
        (OUT_DIR/sub).mkdir(parents=True, exist_ok=True)


def save_plot(name):
    plt.tight_layout()
    for ext in ['png','pdf']:
        plt.savefig(OUT_DIR/'figures'/f'{name}.{ext}', bbox_inches='tight')
    plt.close()


def table(df, name):
    df.to_csv(OUT_DIR/'tables'/f'{name}.csv', index=False)
    try:
        (OUT_DIR/'tables'/f'{name}.tex').write_text(df.to_latex(index=False, float_format=lambda x:f'{x:.3f}', na_rep='NA'))
    except Exception: pass


def get_pbcseq():
    """No synthetic data substitution. Fail loudly if public dataset cannot be fetched."""
    local=Path('pbcseq.csv')
    errors=[]; source=None
    if local.exists():
        df=pd.read_csv(local); source=f'local:{local.resolve()}'
    else:
        import requests
        for url in BASE_URLS:
            try:
                response=requests.get(url,timeout=25,headers={'User-Agent':'Academic-Joint-Model-Research/1.0'})
                response.raise_for_status()
                from io import StringIO
                candidate=pd.read_csv(StringIO(response.text))
                if {'id','bili','day','status'}.issubset(set(candidate.columns)) and len(candidate)>1000:
                    df=candidate; source=url; break
                errors.append(f'{url}: required columns missing or too few rows')
            except Exception as ex: errors.append(f'{url}: {str(ex)[:180]}')
        if source is None:
            raise RuntimeError('Could not download genuine PBC sequential data. Upload pbcseq.csv to Colab working directory and rerun. Attempts: '+str(errors))
    if 'Unnamed: 0' in df: df=df.drop(columns=['Unnamed: 0'])
    if 'rownames' in df: df=df.drop(columns=['rownames'])
    required={'id','futime','status','trt','age','sex','day','bili'}
    if not required.issubset(df.columns):raise ValueError('Wrong PBC file; expected columns '+str(required)+', got '+str(df.columns.tolist()))
    df['id']=pd.to_numeric(df.id,errors='raise').astype(int)
    for col in ['futime','status','trt','age','day','bili']:
        df[col]=pd.to_numeric(df[col],errors='coerce')
    if df.id.nunique()<200 or df.shape[0]<1500:
        raise ValueError('Unexpected PBC size; expected approximately 312 patients and 1945 rows')
    df.to_csv(OUT_DIR/'data'/'pbcseq_original.csv', index=False)
    content=(OUT_DIR/'data'/'pbcseq_original.csv').read_bytes()
    (OUT_DIR/'data'/'source.json').write_text(json.dumps({
        'source':source, 'sha256_saved_csv':hashlib.sha256(content).hexdigest(),
        'retrieved_utc':datetime.now(timezone.utc).isoformat(),
        'interpretation':'PBC sequential records; original status=0 censored,1 transplant,2 death',
        'reference':'Therneau and Grambsch (2000); survival::pbcseq',
        'authors_dataset_note':'Research-use public historical data; check source terms.'},indent=2))
    return df, source


@dataclass
class Cohort:
    ids: np.ndarray
    t: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    sex: np.ndarray
    trt: np.ndarray
    age: np.ndarray
    r: np.ndarray
    event: np.ndarray
    landmark: float

    def subset(self, idx):
        return Cohort(self.ids[idx], self.t[idx],self.y[idx],self.mask[idx],
                      self.sex[idx],self.trt[idx],self.age[idx],self.r[idx],
                      self.event[idx],self.landmark)
    def __len__(self):return len(self.ids)


def make_cohort(df, landmark=LANDMARK, min_visits=2):
    # Eligibility requires T>landmark and >=2 distinct valid labs observed by landmark.
    # Composite endpoint: death or transplant; alive is right censored.
    rows=[]; counts={'total':int(df.id.nunique()),'prior_endpoint_or_censor':0,
                     'insufficient_prelandmark_bilirubin':0,'eligible':0}
    for pid,g in df.groupby('id',sort=True):
        g=g.sort_values('day')
        row=g.iloc[0]
        if pd.isna(row.futime) or row.futime <= landmark*365.25:
            counts['prior_endpoint_or_censor']+=1; continue
        sub=g[(g.day>=0) & (g.day<=landmark*365.25+1e-8) &
              g.bili.notna() & (g.bili>0)].copy()
        sub=sub.drop_duplicates(subset='day',keep='first').sort_values('day')
        if len(sub)<min_visits:
            counts['insufficient_prelandmark_bilirubin']+=1; continue
        sex=str(row.sex).strip().lower()
        # survival::pbcseq encodes sex as 'f'/'m' (or possible 1/2 variants)
        female=float(sex in ['f','female','2'])
        trt=float(row.trt==2) if pd.notna(row.trt) else 0.
        age=float(row.age) if pd.notna(row.age) else np.nan
        rows.append({'id':pid,'times':(sub.day/365.25).to_numpy(float),
                     'values':np.log(sub.bili.to_numpy(float)),
                     'age':age,'sex':female,'trt':trt,'r':float(row.futime/365.25-landmark),
                     'event':int(row.status in [1,2]),'n':len(sub)})
    if not rows:raise ValueError('No eligible landmark subjects: check data and landmark')
    counts['missing_baseline_age_imputed']=int(sum(np.isnan(d['age']) for d in rows))
    for d in rows:d['age']=50. if np.isnan(d['age']) else d['age']
    # Pre-fixed clinical scaling avoids estimating transformations from held-out subjects.
    for d in rows:d['age']=(d['age']-50.0)/10.0
    M=max(d['n'] for d in rows); N=len(rows)
    t=np.zeros((N,M)); y=np.zeros((N,M)); mask=np.zeros((N,M))
    for i,d in enumerate(rows):
        n=d['n'];t[i,:n]=d['times'];y[i,:n]=d['values'];mask[i,:n]=1
    counts['eligible']=N
    ids=np.array([d['id'] for d in rows]);
    event=np.array([d['event'] for d in rows]);
    coh=Cohort(ids,t,y,mask,np.array([d['sex'] for d in rows]),
               np.array([d['trt'] for d in rows]),np.array([d['age'] for d in rows]),
               np.array([d['r'] for d in rows]),event,landmark)
    pd.DataFrame([{'id':d['id'],'visits':d['n'],'time_after_landmark':d['r'],
                   'event':d['event'],'age_standardized':d['age'],'female':d['sex'],
                   'placebo':d['trt']} for d in rows]).to_csv(OUT_DIR/'data'/f'landmark_subjects_{landmark:g}y.csv',index=False)
    return coh,counts


def make_quadrature(model):
    if model != 'Decay-SNI':
        return {'h':np.array([np.sqrt(2/np.pi)]), 'u':np.ones(1),
                'v':np.ones(1),'logw':np.zeros(1),'a':np.sqrt(2/np.pi)}
    nh,nu,nv=map(int,ROBUST_QUAD.split(','))
    def GL(n):
        x,w=leggauss(n);return (x+1)/2,w/2
    ph,wh=GL(nh); pu,wu=GL(nu); pv,wv=GL(nv)
    hh=ndtri((ph+1)/2); uu=gamma_dist.ppf(pu,DF_B/2,scale=2/DF_B)
    vv=gamma_dist.ppf(pv,DF_E/2,scale=2/DF_E)
    nodes=list(product(range(nh),range(nu),range(nv)))
    vals={k:[] for k in ['h','u','v','logw']}
    for i,j,k in nodes:
        vals['h'].append(hh[i]);vals['u'].append(uu[j]);vals['v'].append(vv[k]);
        vals['logw'].append(np.log(wh[i]*wu[j]*wv[k]))
    a=np.sqrt(2/np.pi)*np.exp(.5*np.log(DF_B/2)+gammaln((DF_B-1)/2)-gammaln(DF_B/2))
    return {**{k:np.array(v) for k,v in vals.items()},'a':a}


class MarginalModel:
    """Exact 2D random-effects integration, plus deterministic quadrature over
    independent half-normal (skew) H, gamma U (random effects), gamma V (residual).
    Quadrature-approximated SNI marginal likelihood; Gaussian model exact.
    """
    def __init__(self, cohort, name):
        self.c=cohort; self.name=name; N=len(cohort)
        T=lambda a:torch.tensor(a,dtype=torch.float64)
        self.t=T(cohort.t);self.y=T(cohort.y);self.mask=T(cohort.mask)
        self.n=self.mask.sum(dim=1)[:,None]
        self.st=(self.t*self.mask).sum(dim=1)[:,None]
        self.st2=(self.t**2*self.mask).sum(dim=1)[:,None]
        self.sex=T(cohort.sex)[:,None];self.trt=T(cohort.trt)[:,None]
        self.age=T(cohort.age)[:,None]
        self.event=T(cohort.event)[:,None]
        self.logr=T(np.log(cohort.r))[:,None]
        quad=make_quadrature(name)
        self.h=T(quad['h'])[None,:]; self.u=T(quad['u'])[None,:]
        self.v=T(quad['v'])[None,:];self.logw=T(quad['logw'])[None,:]
        self.a=quad['a'];self.K=len(quad['u']);self.nsubj=N
        self.y2=(self.y*self.mask).sum(dim=1)[:,None] # actually sum(y), named for reuse
    def initial(self):
        c=self.c
        i,j=np.where(c.mask>0)
        X=np.column_stack([np.ones(len(i)),c.t[i,j],c.sex[i],c.trt[i]])
        b=np.linalg.lstsq(X,c.y[i,j],rcond=None)[0]
        s=float(np.std(c.y[i,j]-X@b))
        XX=np.column_stack([np.ones(len(c)),c.age,c.sex,c.trt])
        e=(c.event>0)
        # censored log observed R is a lower bound; only a rough starting point.
        a=np.linalg.lstsq(XX[e] if e.sum()>8 else XX,
                np.log(c.r[e]) if e.sum()>8 else np.log(c.r),rcond=None)[0]
        x=np.zeros(len(NAMES));x[:4]=np.clip(b,-2,2);x[4:8]=np.clip(a,-1.5,2)
        x[IX['nu']]=-0.20
        x[IX['log_se']]=np.log(max(.20,min(1.2,s*.7)))
        x[IX['log_sT']]=np.log(.85)
        x[IX['log_sd0']]=np.log(.48);x[IX['log_sd1']]=np.log(.22)
        x[IX['raw_corr']]=0.;x[IX['raw_rho']]=-.35
        x[IX['log_lambda']]=np.log(1.0)
        return x
    def bounds(self):
        bd=[(-4,4),(-2,2),(-3,3),(-3,3),(-3.0,3.5),(-3,3),(-3,3),(-3,3),
            (-3,3),(-3,1.1),(-2.7,1.2),(-3,1.5),(-3.6,1.2),(-2.1,2.1),
            (-1.8,1.8),(np.log(.08),np.log(8)),(-2.5,2.5),(-2,2)]
        if self.name=='Shared-Gaussian':bd[14]=(0,0);bd[15]=(0,0)
        elif self.name=='Flat-Gaussian':bd[15]=(0,0)
        if self.name!='Decay-SNI':bd[16]=(0,0);bd[17]=(0,0)
        return bd
    def likelihoods(self, p, conditional_components=False):
        # p is a trainable torch parameter vector.
        be0,bet,bes,bed,al0,ala,als,ald,nu=p[:9]
        se=torch.exp(p[9]);sT=torch.exp(p[10]);sd0=torch.exp(p[11]);sd1=torch.exp(p[12])
        cor=.95*torch.tanh(p[13]);rho=.95*torch.tanh(p[14]);lam=torch.exp(p[15]);
        if self.name=='Shared-Gaussian':rho=rho*0
        if self.name=='Flat-Gaussian':lam=lam*0
        skew0,skew1=p[16],p[17]
        if self.name!='Decay-SNI':skew0,skew1=skew0*0,skew1*0
        # Longitudinal sufficient statistics; all schedules irregular.
        resid=(self.y-(be0+bet*self.t+bes*self.sex+bed*self.trt))*self.mask
        R0=resid.sum(dim=1)[:,None]
        R1=(resid*self.t).sum(dim=1)[:,None]
        RR=(resid**2).sum(dim=1)[:,None]
        mu0=(self.h/torch.sqrt(self.u)-self.a)*skew0
        mu1=(self.h/torch.sqrt(self.u)-self.a)*skew1
        S0=R0-self.n*mu0-self.st*mu1
        S1=R1-self.st*mu0-self.st2*mu1
        SS=RR-2*mu0*R0-2*mu1*R1+self.n*mu0**2+2*self.st*mu0*mu1+self.st2*mu1**2
        ic=1/(1-cor**2)
        g00=ic/sd0**2; g01=-ic*cor/(sd0*sd1);g11=ic/sd1**2
        mult=self.v/(se**2)
        A=self.u*g00+mult*self.n
        B=self.u*g01+mult*self.st
        C=self.u*g11+mult*self.st2
        det=A*C-B**2
        P00=C/det;P01=-B/det;P11=A/det
        k0=mult*S0;k1=mult*S1
        db0=P00*k0+P01*k1;db1=P01*k0+P11*k1
        post0=mu0+db0;post1=mu1+db1
        logdet_gamma=2*torch.log(sd0)+2*torch.log(sd1)+torch.log(1-cor**2)
        logdet_y=self.n*(2*torch.log(se)-torch.log(self.v))+logdet_gamma-2*torch.log(self.u)+torch.log(det)
        logpy=-.5*(self.n*np.log(2*np.pi)+logdet_y+mult*SS-k0*db0-k1*db1)
        # Normalized kernel removes design-dependent covariance restriction.
        k=torch.exp(-lam*(self.c.landmark-self.t))*self.mask
        w=k/torch.sqrt((k**2).sum(dim=1,keepdim=True))
        W0=w.sum(dim=1,keepdim=True);W1=(w*self.t).sum(dim=1,keepdim=True)
        Wy=(w*resid).sum(dim=1,keepdim=True)
        ares=sT*rho/se
        cb0=-ares*W0;cb1=nu-ares*W1
        Lmu=al0+ala*self.age+als*self.sex+ald*self.trt+ares*Wy+cb0*post0+cb1*post1
        Lvar=sT**2*(1-rho**2)/self.v+cb0**2*P00+2*cb0*cb1*P01+cb1**2*P11
        z=(self.logr-Lmu)/torch.sqrt(Lvar)
        log_event=-.5*(np.log(2*np.pi)+torch.log(Lvar)+z*z)-self.logr
        log_cens=torch.special.log_ndtr(-z)
        log_surv=torch.where(self.event>.5,log_event,log_cens)
        joint=torch.logsumexp(logpy+log_surv+self.logw,dim=1)
        marg=torch.logsumexp(logpy+self.logw,dim=1)
        if conditional_components:
            return joint,marg,(Lmu,Lvar,logpy)
        return joint,marg
    def pen(self, p):
        # weak, predeclared Gaussian penalties used uniformly across model classes;
        # estimates are MAP / penalized likelihood, not unregularized MLE.
        v=(p[0]/5)**2+(p[1]/2)**2+(p[2]/3)**2+(p[3]/3)**2
        v+=(p[4]/4)**2+(p[5]/3)**2+(p[6]/3)**2+(p[7]/3)**2
        v+=(p[8]/2.5)**2+(p[13]/1.5)**2+(p[14]/1.5)**2
        if self.name in ['Decay-Gaussian','Decay-SNI']:
            v+=((p[15]-np.log(1.0))/1.3)**2
        if self.name=='Decay-SNI':v+=(p[16]/1.2)**2+(p[17]/.65)**2
        return .5*v
    def fit(self,maxiter=MAX_ITER):
        x0=self.initial(); bd=self.bounds()
        for idx,(a,b) in enumerate(bd):
            if a==b:x0[idx]=a
            else:x0[idx]=max(a+1e-5,min(x0[idx],b-1e-5))
        def obj(x):
            p=torch.tensor(x,requires_grad=True,dtype=torch.float64)
            joint,marg=self.likelihoods(p)
            loss=-joint.sum()+self.pen(p)
            if not torch.isfinite(loss):
                return 1e30,np.zeros_like(x)
            loss.backward()
            grad=p.grad.detach().numpy()
            return float(loss.detach()),grad
        tic=time.time()
        res=minimize(obj,x0,jac=True,method='L-BFGS-B',bounds=bd,
                     options={'maxiter':maxiter,'ftol':1e-7,'gtol':2e-4,'maxls':18})
        p=torch.tensor(res.x,dtype=torch.float64)
        with torch.no_grad():
            j,m=self.likelihoods(p)
        diagnostics={'model':self.name,'n':len(self.c),'success':bool(res.success),
               'message':str(res.message),'nit':int(res.nit),'nfev':int(res.nfev),
               'elapsed_seconds':round(time.time()-tic,3),
               'penalized_nll':float(res.fun),
               'joint_loglik':float(j.sum()),
               'conditional_survival_loglik':float((j-m).sum()),
               'rho':float(.95*np.tanh(res.x[IX['raw_rho']])) if 'Shared' not in self.name else 0.,
               'lambda_per_year':float(np.exp(res.x[IX['log_lambda']])) if 'Decay' in self.name else 0.}
        if not res.success:
            print('   WARNING: optimizer status',res.message,'for',self.name)
        return res.x,diagnostics
    def predictions(self,x,horizons=HORIZONS):
        p=torch.tensor(x,dtype=torch.float64)
        with torch.no_grad():
            j,m,(lmu,lvar,logpy)=self.likelihoods(p,True)
            logweights=(logpy+self.logw)-m[:,None]
            weights=torch.exp(logweights)
            ans={}
            for h in horizons:
                z=(np.log(h)-lmu)/torch.sqrt(lvar)
                ans[h]=np.asarray((weights*torch.special.ndtr(z)).sum(dim=1))
            return ans,np.asarray(j-m),np.asarray(j),np.asarray(m)


def simulate(n,scenario,rng,landmark=LANDMARK):
    # Underlying data genuinely follow the normalized, landmark-conditional model.
    skew=(scenario=='SkewT-decay'); rho=0.0 if scenario=='Null-residual' else -.58
    lam=1.7; nu=-.55;se=.34;sT=.73
    b0,b1,cor=.62,.24,-.10
    Delta=np.array([.55,.14]) if skew else np.zeros(2)
    u=rng.gamma(DF_B/2,2/DF_B,n) if skew else np.ones(n)
    v=rng.gamma(DF_E/2,2/DF_E,n) if skew else np.ones(n)
    a=np.sqrt(2/np.pi)* (np.exp(.5*np.log(DF_B/2)+gammaln((DF_B-1)/2)-gammaln(DF_B/2)) if skew else 1)
    G=np.linalg.cholesky(np.array([[b0**2,cor*b0*b1],[cor*b0*b1,b1**2]]))
    random_b=(rng.standard_normal((n,2))@G.T + np.abs(rng.normal(size=n))[:,None]*Delta[None,:])/np.sqrt(u[:,None])-a*Delta[None,:]
    sex=rng.binomial(1,.84,n).astype(float);trt=rng.binomial(1,.5,n).astype(float)
    age=rng.normal(size=n)
    times=[];responses=[]; r=[];events=[];censors=[]
    for i in range(n):
        ni=int(rng.integers(2,7))
        ti=np.r_[0.,np.sort(rng.uniform(.05,landmark,ni-1))]
        mean=.2+.17*ti+.16*sex[i]-.10*trt[i]+random_b[i,0]+random_b[i,1]*ti
        eps=(se/np.sqrt(v[i]))*rng.normal(size=ni)
        yi=mean+eps
        w=np.exp(-lam*(landmark-ti));w=w/np.linalg.norm(w)
        # log residual lifetime, conditional on the biomarker trajectory
        l=.75-.25*age[i]+.08*sex[i]+.04*trt[i]+nu*random_b[i,1]
        l+=sT*rho/se*np.dot(w,eps)+(sT*np.sqrt(1-rho**2)/np.sqrt(v[i]))*rng.normal()
        lifetime=np.exp(l)
        censor=np.exp(rng.normal(1.5,.7))
        r.append(min(lifetime,censor));events.append(int(lifetime<=censor))
        times.append(ti);responses.append(yi);censors.append(censor)
    M=max(len(x) for x in times);t=np.zeros((n,M));y=np.zeros((n,M));mask=np.zeros((n,M))
    for i in range(n):
        j=len(times[i]);t[i,:j]=times[i];y[i,:j]=responses[i];mask[i,:j]=1
    cohort=Cohort(np.arange(n),t,y,mask,sex,trt,age,np.asarray(r),np.asarray(events),landmark)
    return cohort,{'rho':rho,'lambda':lam,'nu':nu,'censoring_fraction':float(1-np.mean(events)),
                   'scenario':scenario, 'skew':bool(skew)}


def km(times, events):
    """Kaplan-Meier right-continuous step function; events=1 denotes failure."""
    times=np.asarray(times,float);events=np.asarray(events,int)
    order=np.argsort(times);times=times[order];events=events[order]
    uniq=np.unique(times);surv=[];s=1.;at_risk=len(times)
    for x in uniq:
        mk=(times==x);d=np.sum(events[mk]);n=np.sum(mk)
        if at_risk>0:s*=max(0.,1.-d/at_risk)
        surv.append(s);at_risk-=n
    return uniq,np.asarray(surv)


def km_at(grid, times, surv, before=False):
    grid=np.asarray(grid,float)
    pos=np.searchsorted(times,grid,side='left' if before else 'right')-1
    return np.where(pos>=0,surv[np.maximum(0,pos)],1.)


def ipcw_brier(train,test,pred,horizon):
    ctime,csurv=km(train.r,1-train.event)
    G_t=float(np.clip(km_at([horizon],ctime,csurv)[0],.05,1))
    G_e=np.clip(km_at(test.r,ctime,csurv,before=True),.05,1)
    early=(test.event==1)&(test.r<=horizon)
    late=(test.r>horizon)
    weights=np.where(early,1/G_e,np.where(late,1/G_t,0))
    y=early.astype(float)
    return float(np.mean(weights*(y-pred)**2)),float(np.mean(weights>0))


def harrell_c(c,pred_risk):
    r=np.asarray(c.r);e=np.asarray(c.event);risk=np.asarray(pred_risk)
    comp=(e[:,None]==1)&(r[:,None]<r[None,:])
    cnt=int(comp.sum())
    if not cnt:return np.nan,cnt
    v=risk[:,None]-risk[None,:]
    return float(((v>0)*comp+.5*(v==0)*comp).sum()/cnt),cnt


def evaluate(train,test,model_name,maxiter=MAX_ITER):
    m=MarginalModel(train,model_name)
    x,d=m.fit(maxiter)
    mt=MarginalModel(test,model_name)
    ps,conditional,joint,marg=mt.predictions(x,horizons=HORIZONS)
    res={**d,'test_n':len(test),'test_events':int(test.event.sum()),
         'test_mean_conditional_logscore':float(np.mean(conditional)),
         'test_mean_joint_logscore':float(np.mean(joint))}
    for h in HORIZONS:
        risk=ps[h]
        bs,cover=ipcw_brier(train,test,risk,h)
        res[f'brier_{h:g}y']=bs;res[f'ipcw_coverage_{h:g}y']=cover
        c,pairs=harrell_c(test,risk)
        res[f'cindex_{h:g}y']=c;res[f'cindex_pairs_{h:g}y']=pairs
    return x,res,ps,conditional


def plot_real_data(cohort):
    ids=cohort.ids;order=np.random.default_rng(SEED).permutation(len(cohort))[:min(80,len(cohort))]
    fig,ax=plt.subplots(figsize=(8,5))
    for idx in order:
        wh=cohort.mask[idx]>0
        ax.plot(cohort.t[idx,wh],cohort.y[idx,wh],color=('#aa574b' if cohort.event[idx] else '#4c7196'),alpha=.20,lw=.8)
    ax.axvline(cohort.landmark,c='k',ls='--',lw=1,label='Landmark')
    ax.set(xlabel='Years since enrollment',ylabel='Log serum bilirubin (mg/dL)',
           title='Real cohort: observed pre-landmark biomarker trajectories')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],color='#aa574b',label='Death/transplant'),
                       Line2D([0],[0],color='#4c7196',label='Censored'),
                       Line2D([0],[0],color='k',linestyle='--',label='Landmark')])
    save_plot('real_longitudinal_profiles')
    fig,ax=plt.subplots(figsize=(7,4))
    z,s=km(cohort.r,cohort.event)
    ax.step(np.r_[0,z],np.r_[1,s],where='post',lw=2,label='Observed KM',color='#34546f')
    ax.set(xlabel='Years after landmark',ylabel='Composite event-free survival',
           title='Real cohort: landmark survival curve',ylim=(0,1.02))
    ax.legend();save_plot('real_landmark_KM')


def spectral_verification():
    # Check matrix eigenvalues directly and Schur complements analytically.
    rrho=.7; se=.36; sT=.8;lambda_=1.5; ell=2.
    nvals=np.array([1,2,3,5,10,20,50,100,250,500])
    rows=[]
    for n in nvals:
        t=np.linspace(0,ell,n) if n>1 else np.array([ell])
        k=np.exp(-lambda_*(ell-t));w=k/np.linalg.norm(k)
        c=se*sT*rrho*w
        schur=sT**2-c@c/se**2
        # Analytic eigenvalue of full (n+1)-dimensional covariance.
        emin=.5*(se**2+sT**2-np.sqrt((se**2-sT**2)**2+4*np.dot(c,c)))
        if n<=20:
            sigma=np.block([[se**2*np.eye(n),c[:,None]],[c[None,:],np.array([[sT**2]])]])
            direct=float(np.linalg.eigvalsh(sigma)[0])
            assert abs(direct-emin)<1e-10, 'Analytic/direct eigenvalue mismatch'
        else:direct=np.nan
        cflat=se*sT*rrho*np.ones(n)
        oldschur=sT**2-cflat@cflat/se**2
        cdecay=se*sT*rrho*k
        old_decay_schur=sT**2-cdecay@cdecay/se**2
        rows.append({'visits':n,'normalized_min_eigenvalue':emin,
                     'direct_min_eigenvalue':direct,
                     'normalized_schur':schur,'unnormalized_flat_schur':oldschur,
                     'unnormalized_decay_schur':old_decay_schur,
                     'min_gap':float(np.diff(t).min()) if n>1 else np.nan})
    df=pd.DataFrame(rows);table(df,'positive_definiteness_stress')
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.semilogx(df.visits,df.normalized_schur,'o-',label='Normalized decay: Schur complement')
    ax.semilogx(df.visits,df.unnormalized_flat_schur,'s--',label='Unnormalized flat loading')
    ax.semilogx(df.visits,df.unnormalized_decay_schur,'^:',label='Unnormalized exponential decay')
    ax.axhline(0,color='black',linewidth=.8)
    ax.set(xlabel='Number of observed visits',ylabel='Schur complement',
           title='Positive definiteness with increasingly dense visit schedules')
    ax.legend();save_plot('simulation_positive_definiteness')
    fig,ax=plt.subplots(figsize=(8,4))
    lag=np.linspace(0,2,160)
    for l in [.25,.75,1.5,3.]:
        ax.plot(lag,np.exp(-l*lag),label=fr'$\lambda$={l:g} yr$^{{-1}}$')
    ax.set(xlabel='Years before landmark',ylabel='Unnormalized relative kernel',
           title='Continuous exponential recency weights')
    ax.legend();save_plot('kernel_decay_profiles')
    return df


def real_analysis(cohort):
    print('\n=== REAL PBC ANALYSIS ===',flush=True)
    plot_real_data(cohort)
    records=[];predrows=[];paramrows=[]
    for rep in range(REAL_REPEATS):
        trainid,testid=train_test_split(np.arange(len(cohort)),test_size=.28,
                           random_state=SEED+100+rep,stratify=cohort.event)
        train,test=cohort.subset(trainid),cohort.subset(testid)
        print('Real split',rep+1,'train=',len(train),'test=',len(test),flush=True)
        for name in MODELS:
            print('  fitting',name,flush=True)
            try:
                x,d,ps,ls=evaluate(train,test,name)
                d['repeat']=rep+1;records.append(d)
                par={**{'model':name,'repeat':rep+1},**{NAMES[i]:x[i] for i in range(len(NAMES))}}
                paramrows.append(par)
                for i in range(len(test)):
                    predrows.append({'repeat':rep+1,'model':name,'id':int(test.ids[i]),
                             'observed_residual_time':test.r[i], 'event':test.event[i],
                             'conditional_logscore':ls[i],
                             **{f'pred_event_{h:g}y':ps[h][i] for h in HORIZONS}})
            except Exception as e:
                print('FIT FAILURE',name,repr(e),flush=True)
                (OUT_DIR/'logs'/'failures.log').open('a').write('Real '+name+'\n'+traceback.format_exc()+'\n')
    if not records: raise RuntimeError('All real model fits failed. See logs/failures.log')
    result=pd.DataFrame(records);table(result,'real_model_results_all_splits')
    pred=pd.DataFrame(predrows);pred.to_csv(OUT_DIR/'data'/'heldout_predictions.csv',index=False)
    pd.DataFrame(paramrows).to_csv(OUT_DIR/'tables'/'real_model_parameters.csv',index=False)
    # Paired bootstrap: resample the same held-out individuals for competing
    # models, avoiding false independent-sample comparisons.
    comparisons=[('Decay-Gaussian','Shared-Gaussian'),('Decay-Gaussian','Flat-Gaussian'),
                 ('Decay-SNI','Decay-Gaussian')]
    boot=[]; brng=np.random.default_rng(SEED+355)
    for rep in sorted(pred.repeat.unique()):
        pivot=pred[pred.repeat==rep].pivot(index='id',columns='model',values='conditional_logscore')
        for candidate,reference in comparisons:
            if candidate not in pivot or reference not in pivot:continue
            deltas=(pivot[candidate]-pivot[reference]).dropna().to_numpy()
            if len(deltas)<10:continue
            means=np.mean(brng.choice(deltas,size=(1000,len(deltas)),replace=True),axis=1)
            boot.append({'repeat':rep,'candidate':candidate,'reference':reference,
                         'paired_n':len(deltas),'delta_logscore_mean':float(deltas.mean()),
                         'ci_lower_95':float(np.quantile(means,.025)),
                         'ci_upper_95':float(np.quantile(means,.975)),
                         'two_sided_bootstrap_tail_fraction':float(2*min(np.mean(means<=0),np.mean(means>=0)))})
    if boot:
        bdf=pd.DataFrame(boot);table(bdf,'real_paired_heldout_logscore_bootstrap')
        f,ax=plt.subplots(figsize=(9,4.5))
        labels=[];vals=[];lo=[];hi=[]
        for _,q in bdf.iterrows():
            labels.append(f"{q.candidate} vs {q.reference} (split {q.repeat})")
            vals.append(q.delta_logscore_mean);lo.append(q.delta_logscore_mean-q.ci_lower_95)
            hi.append(q.ci_upper_95-q.delta_logscore_mean)
        ypos=np.arange(len(vals))
        ax.errorbar(vals,ypos,xerr=[lo,hi],fmt='o',capsize=3)
        ax.axvline(0,color='black',ls='--',lw=.8)
        ax.set_yticks(ypos);ax.set_yticklabels(labels,fontsize=8)
        ax.set(xlabel='Paired change in held-out log score (higher favors first model)',
               title='Paired subject bootstrap (descriptive, overlapping splits)')
        save_plot('real_paired_logscore_differences')
    cols=['test_mean_conditional_logscore','test_mean_joint_logscore','brier_2y','brier_4y','cindex_2y','cindex_4y']
    summary=result.groupby('model',sort=False)[cols].agg(['mean','std']).reset_index()
    summary.columns=['model' if c[0]=='model' else (c[0]+'_'+c[1]) for c in summary.columns.to_flat_index()]
    table(summary,'real_predictive_comparison')
    # Performance plots: comparisons are held-out, never in-sample.
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    xlabels=MODELS; xidx=np.arange(len(xlabels));
    for ax,col,title,ylab in zip(axs,['test_mean_conditional_logscore','brier_2y'],
                               ['Held-out conditional survival log score','Held-out 2-year IPCW Brier score'],
                               ['Mean log score (higher better)','IPCW Brier (lower better)']):
        vals=[]
        for lab in xlabels:
            rr=result[result.model==lab];vals.append(rr[col].mean() if len(rr) else np.nan)
        ax.bar(xidx,vals,alpha=.85)
        for k,lab in enumerate(xlabels):
            r=result[result.model==lab]
            if len(r):ax.scatter(np.full(len(r),k),r[col],color='black',s=15,zorder=3)
        ax.set_xticks(xidx);ax.set_xticklabels(xlabels,rotation=24,ha='right',fontsize=8)
        ax.set(title=title,ylabel=ylab)
    save_plot('real_heldout_model_comparison')
    # Calibration, risk stratification on one fixed split; visualization descriptive.
    sub=pred[(pred.repeat==1) & (pred.model=='Decay-Gaussian')].copy()
    if len(sub)>=20:
        testidx=sub.id.to_numpy()
        # Each person appears once in this split; match associated test rows.
        risk=sub['pred_event_2y'].to_numpy();r=sub.observed_residual_time.to_numpy();e=sub.event.to_numpy()
        quant=np.quantile(risk,[0,.33,.67,1]);bins=np.unique(quant)
        fig,ax=plt.subplots(figsize=(7,4.5))
        for j in range(len(bins)-1):
            mk=(risk>=bins[j])&(risk<bins[j+1] if j<len(bins)-2 else risk<=bins[j+1])
            if mk.sum()<5:continue
            kt,ks=km(r[mk],e[mk]);ax.step(np.r_[0,kt],np.r_[1,ks],where='post',
                  label=f'Predicted risk group {j+1} (n={mk.sum()})')
        ax.set(xlabel='Years after landmark',ylabel='Observed event-free survival',
               title='Out-of-sample risk separation (descriptive)')
        ax.legend();save_plot('real_risk_strata_KM')
        bins=np.quantile(risk,[0,.25,.5,.75,1]); bins=np.unique(bins)
        cals=[]
        for j in range(len(bins)-1):
            mk=(risk>=bins[j])&(risk<bins[j+1] if j<len(bins)-2 else risk<=bins[j+1])
            if mk.sum()<5:continue
            kt,ks=km(r[mk],e[mk]);observed=1-float(km_at([2.],kt,ks)[0])
            cals.append({'group':j+1,'n':int(mk.sum()),'mean_predicted_risk':float(risk[mk].mean()),
                         'KM_observed_2year_event':observed})
        if cals:
            cal=pd.DataFrame(cals);table(cal,'real_2year_calibration_bins')
            fig,ax=plt.subplots(figsize=(5.5,5))
            ax.plot([0,1],[0,1],'k--',alpha=.65,label='Ideal')
            ax.scatter(cal.mean_predicted_risk,cal.KM_observed_2year_event,s=60)
            for _,z in cal.iterrows(): ax.annotate(f"n={int(z['n'])}",
                    (z.mean_predicted_risk,z.KM_observed_2year_event),xytext=(6,5),textcoords='offset points',fontsize=8)
            ax.set(xlabel='Mean predicted 2-year event probability',
                   ylabel='KM 2-year observed event probability',
                   title='Out-of-sample calibration, binning uncertainty',xlim=(0,1),ylim=(0,1))
            save_plot('real_calibration_2year')
    return result,summary


def simulation_study():
    print('\n=== SIMULATION VERIFICATION ===',flush=True)
    out=[];pars=[]
    rng=np.random.default_rng(SEED+20)
    scenarios=['Gaussian-decay','SkewT-decay','Null-residual']
    for scenario in scenarios:
        for rep in tqdm(range(SIM_REPS),desc=f'Simulation {scenario}',leave=True):
            # Repeated independent replicates; prospective data splits.
            one,truth=simulate(SIM_N,scenario,rng)
            ii,jj=train_test_split(np.arange(len(one)),test_size=.3,
                     random_state=SEED+rep,stratify=one.event)
            train,test=one.subset(ii),one.subset(jj)
            print(f'  {scenario}: rep {rep+1}/{SIM_REPS}',flush=True)
            for name in MODELS:
                try:
                    x,d,ps,ls=evaluate(train,test,name,maxiter=max(65,int(MAX_ITER*.8)))
                    out.append({'scenario':scenario,'rep':rep+1,'model':name,
                         'true_rho':truth['rho'],'true_lambda':truth['lambda'],
                         'true_censoring_fraction':truth['censoring_fraction'],**d})
                    pars.append({'scenario':scenario,'rep':rep+1,'model':name,
                         'rho_hat':d['rho'],'lambda_hat':d['lambda_per_year'],
                         'nu_hat':float(x[IX['nu']]),'true_rho':truth['rho'],
                         'true_lambda':truth['lambda'],'true_nu':truth['nu']})
                except Exception:
                    (OUT_DIR/'logs'/'failures.log').open('a').write('Sim '+scenario+' '+name+'\n'+traceback.format_exc()+'\n')
    if not out:raise RuntimeError('Simulation models all failed; inspect logs/failures.log')
    df=pd.DataFrame(out);table(df,'simulation_all_replicates')
    pr=pd.DataFrame(pars);table(pr,'simulation_parameter_estimates')
    recover=[]
    for (scenario,model),g in pr.groupby(['scenario','model'],sort=False):
        for estim,truth in [('rho_hat','true_rho'),('lambda_hat','true_lambda'),('nu_hat','true_nu')]:
            if estim=='lambda_hat' and (scenario=='Null-residual' or model in ['Shared-Gaussian','Flat-Gaussian']):
                continue # lambda is not identified when rho=0, and not fitted in two models.
            if estim=='rho_hat' and model=='Shared-Gaussian':continue
            a=g[estim].to_numpy(); t=g[truth].to_numpy()
            recover.append({'scenario':scenario,'model':model,'parameter':estim,
                            'replications':len(g),'mean_estimate':float(a.mean()),
                            'true_value':float(t.mean()),'bias':float(np.mean(a-t)),
                            'rmse':float(np.sqrt(np.mean((a-t)**2)))})
    if recover:table(pd.DataFrame(recover),'simulation_bias_rmse')
    metrics=['test_mean_conditional_logscore','brier_2y','cindex_2y','rho','lambda_per_year']
    summ=df.groupby(['scenario','model'],sort=False)[metrics].agg(['mean','std','count']).reset_index()
    summ.columns=['_'.join(c).strip('_') for c in summ.columns.to_flat_index()]
    table(summ,'simulation_summary')
    fig,axs=plt.subplots(1,3,figsize=(13.6,4.2))
    for ax,sc in zip(axs,['Gaussian-decay','SkewT-decay','Null-residual']):
        dd=df[df.scenario==sc];parts=[];labels=[]
        for name in MODELS:
            z=dd.loc[dd.model==name,'test_mean_conditional_logscore'].dropna()
            if len(z):parts.append(z.to_numpy());labels.append(name.replace('-Gaussian','-G').replace('Decay-SNI','SNI'))
        if parts:
            ax.boxplot(parts,tick_labels=labels,showfliers=False)
            ax.tick_params(axis='x',rotation=35,labelsize=8)
        ax.set_title(sc);ax.set_ylabel('Held-out survival log score')
    save_plot('simulation_predictive_logscore')
    fig,axs=plt.subplots(1,3,figsize=(13.6,4.2))
    for ax,sc in zip(axs,['Gaussian-decay','SkewT-decay','Null-residual']):
        dd=pr[(pr.scenario==sc)&(pr.model=='Decay-Gaussian')]
        ax.scatter(dd.rho_hat,dd.lambda_hat,alpha=.7,s=30)
        ax.axvline(float(dd.true_rho.iloc[0]) if len(dd) else 0,c='k',ls='--',lw=1,label='True rho')
        if len(dd):ax.axhline(float(dd.true_lambda.iloc[0]),c='grey',ls=':',lw=1,label='True lambda')
        ax.set(xlabel='Estimated rho',ylabel='Estimated lambda (/year)',title=sc)
        ax.legend(fontsize=7)
    save_plot('simulation_rho_lambda_recovery')
    fig,ax=plt.subplots(figsize=(8,4.5))
    for sc in scenarios:
        dd=df[(df.scenario==sc)&(df.model.isin(['Shared-Gaussian','Decay-Gaussian','Decay-SNI']))]
        vals=[dd.loc[dd.model==name,'brier_2y'].mean() for name in ['Shared-Gaussian','Decay-Gaussian','Decay-SNI']]
        ax.plot(['Shared-Gaussian','Decay-Gaussian','Decay-SNI'],vals,marker='o',label=sc)
    ax.set(xlabel='Fitted model',ylabel='Mean held-out IPCW Brier, 2 years',
           title='Simulation prediction comparison (lower is better)')
    ax.tick_params(axis='x',rotation=15);ax.legend();save_plot('simulation_brier_comparison')
    return df,summ


def sensitivity(cohort):
    if not DO_SENSITIVITY:return
    rows=[]
    for landmark in [1.0,3.0]:
        # Each analysis is redefined at a new landmark, using only eligible pre-landmark labs.
        try:
            pbc=pd.read_csv(OUT_DIR/'data'/'pbcseq_original.csv')
            tmp,counts=make_cohort(pbc,landmark=landmark)
            if len(tmp)<60:continue
            tr,te=train_test_split(np.arange(len(tmp)),test_size=.30,random_state=SEED+19,
                                   stratify=tmp.event)
            for name in ['Shared-Gaussian','Decay-Gaussian']:
                x,d,p,ls=evaluate(tmp.subset(tr),tmp.subset(te),name,maxiter=max(85,MAX_ITER//2))
                rows.append({'landmark_years':landmark,'eligible':len(tmp),**d})
        except Exception:
            (OUT_DIR/'logs'/'failures.log').open('a').write('Sensitivity '+str(landmark)+'\n'+traceback.format_exc()+'\n')
    if rows:table(pd.DataFrame(rows),'real_landmark_sensitivity')


def run_self_tests():
    print('Running strict generative / mathematical sanity checks...',flush=True)
    rng=np.random.default_rng(44)
    for scenario in ['Gaussian-decay','SkewT-decay','Null-residual']:
        x,t=simulate(35,scenario,rng)
        assert len(x)==35 and np.all(x.r>0) and np.all(x.mask.sum(axis=1)>=2)
        for name in ['Shared-Gaussian','Decay-Gaussian','Decay-SNI']:
            m=MarginalModel(x,name);p=torch.tensor(m.initial(),requires_grad=True)
            j,py=m.likelihoods(p)
            assert bool(torch.isfinite(j).all() and torch.isfinite(py).all()),(scenario,name)
            loss=-j.sum();loss.backward()
            assert bool(torch.isfinite(p.grad).all()),(scenario,name,'derivative')
            preds,ls,_,_=m.predictions(p.detach().numpy())
            assert all(np.all((v>=0)&(v<=1)) for v in preds.values())
    # Verify Gaussian marginalized likelihood against direct multivariate-normal
    # log density including the log-time Jacobian.
    from scipy.stats import multivariate_normal
    small,_=simulate(12,'Gaussian-decay',np.random.default_rng(800))
    small=small.subset(np.array([0]));small.event[:]=1
    mdl=MarginalModel(small,'Decay-Gaussian')
    x=mdl.initial();x[IX['raw_rho']]=-.5;x[IX['log_lambda']]=np.log(1.3)
    pp=torch.tensor(x,dtype=torch.float64)
    with torch.no_grad():lj,_=mdl.likelihoods(pp)
    ni=int(small.mask[0].sum());tt=small.t[0,:ni];yy=small.y[0,:ni]
    ww=np.exp(-np.exp(x[15])*(small.landmark-tt));ww/=np.linalg.norm(ww)
    se,sT,sd0,sd1=np.exp(x[9:13]);corr=.95*np.tanh(x[13]);rho=.95*np.tanh(x[14]);nu=x[8]
    GG=np.array([[sd0**2,corr*sd0*sd1],[corr*sd0*sd1,sd1**2]])
    Z=np.column_stack([np.ones(ni),tt]);e1=np.array([0.,1.])
    cov_y=Z@GG@Z.T+se**2*np.eye(ni)
    cross=nu*(Z@GG@e1)+se*sT*rho*ww
    cov_l=nu**2*GG[1,1]+sT**2
    COV=np.block([[cov_y,cross[:,None]],[cross[None,:],np.array([[cov_l]])]])
    mean_y=x[0]+x[1]*tt+x[2]*small.sex[0]+x[3]*small.trt[0]
    mean_l=x[4]+x[5]*small.age[0]+x[6]*small.sex[0]+x[7]*small.trt[0]
    direct=multivariate_normal.logpdf(np.r_[yy,np.log(small.r[0])],
                       mean=np.r_[mean_y,mean_l],cov=COV)-np.log(small.r[0])
    assert abs(float(lj[0])-direct)<1e-8, ('Gaussian likelihood identity failed',float(lj[0]),direct)
    # Exact n=1 and arbitrarily dense schedules satisfy the normalized Schur identity.
    for n in [1,2,20,1000]:
        q=np.exp(-1.5*(2-np.linspace(0,2,n)));w=q/np.linalg.norm(q)
        assert abs(np.dot(w,w)-1)<1e-10
    print('Self-tests passed.',flush=True)


def main():
    tic=time.time();dirs()
    config={'seed':SEED,'landmark_years':LANDMARK,'simulation_repetitions':SIM_REPS,
       'simulation_subjects_per_replicate':SIM_N,'real_train_test_repeats':REAL_REPEATS,
       'optimizer_maxiter':MAX_ITER,'sni_quadrature_h_u_v':ROBUST_QUAD,
       'sni_df_random_effects':DF_B,'sni_df_residuals':DF_E,
       'horizons_after_landmark_years':HORIZONS,'models':MODELS,
       'description':'Genuine public PBC data and simulation experiments; no manufactured empirical outcomes'}
    (OUT_DIR/'run_config.json').write_text(json.dumps(config,indent=2))
    run_self_tests();spectral_verification()
    real=None;sim=None;err=[]
    try:
        df,source=get_pbcseq()
        coh,counts=make_cohort(df)
        table(pd.DataFrame([{'quantity':k,'value':v} for k,v in {
            **counts,'observed_event_after_landmark':int(coh.event.sum()),
            'censored_after_landmark':int((1-coh.event).sum()),
            'median_biomarker_visits':float(np.median(coh.mask.sum(axis=1))),
            'max_biomarker_visits':int(coh.mask.sum(axis=1).max()),
            'median_followup_after_landmark':float(np.median(coh.r))}.items()]),'real_cohort_summary')
        print('PBC dataset:',source,'landmark cohort:',counts,flush=True)
        real,_=real_analysis(coh)
        sensitivity(coh)
    except Exception as ex:
        err.append('Real-data analysis failed: '+str(ex))
        (OUT_DIR/'logs'/'failures.log').open('a').write('REAL DATA\n'+traceback.format_exc()+'\n')
        print('REAL DATA FAILURE:',str(ex),flush=True)
    try:
        sim,_=simulation_study()
    except Exception as ex:
        err.append('Simulation study failed: '+str(ex))
        (OUT_DIR/'logs'/'failures.log').open('a').write('SIMULATIONS\n'+traceback.format_exc()+'\n')
        print('SIMULATION FAILURE:',str(ex),flush=True)
    status={'status':'completed' if not err else 'partial_or_failed',
            'errors':err,'elapsed_seconds':round(time.time()-tic,1),
            'real_fits':len(real) if real is not None else 0,
            'simulation_fits':len(sim) if sim is not None else 0,
            'finished_utc':datetime.now(timezone.utc).isoformat()}
    (OUT_DIR/'run_status.json').write_text(json.dumps(status,indent=2))
    # Include exact executed source, package versions, and reproducibility metadata.
    try:shutil.copy2(Path(__file__),OUT_DIR/'hj_joint_study.py')
    except Exception:pass
    versions={'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,
            'torch':torch.__version__}
    (OUT_DIR/'environment_versions.json').write_text(json.dumps(versions,indent=2))
    bundle=OUT_DIR.parent/'HJ_structural_joint_outputs.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT_DIR.rglob('*')):
            if p.is_file():z.write(p,arcname=str(p.relative_to(OUT_DIR.parent)))
    print('\nSUCCESS/STATUS:',status,flush=True)
    print('ZIP:',bundle.resolve(),f'({bundle.stat().st_size/1e6:.2f} MB)',flush=True)
    if 'google.colab' in sys.modules:
        from google.colab import files
        files.download(str(bundle))
    return bundle

if __name__=='__main__':main()
