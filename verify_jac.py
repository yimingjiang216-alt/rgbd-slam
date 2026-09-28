# -*- coding: utf-8 -*-
"""验证修正后 BA 的解析雅可比"""
import numpy as np, sys, os, cv2
sys.path.insert(0, r"C:\Users\r26304\Documents\codex-shop\slam_vo")
from pipeline import FX,FY,CX,CY
from scipy.sparse import csr_matrix
from collections import defaultdict
seq=r"C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_freiburg1_xyz"
d=np.load(os.path.join(seq,"slam_front.npz"))
P,pts,obs=d["poses"],d["points"],d["obs"]
frames=list(range(0,40))
byf=defaultdict(list)
for o in obs: byf[int(o[0])].append(o)
obs_w=np.array([o for f in frames for o in byf[f]],float)
gc=defaultdict(int)
for o in obs: gc[int(o[1])]+=1
cnt=defaultdict(int); first={}
for o in obs_w:
    p=int(o[1]); cnt[p]+=1
    f0=int(o[0])
    if p not in first or f0<first[p]: first[p]=f0
cand=sorted([p for p,c in cnt.items() if c>=3 and cnt[p]==gc[p]],key=lambda p:-cnt[p])
q=defaultdict(int); keep=[]
for p in cand:
    f0=first[p]
    if q[f0]<250: keep.append(p); q[f0]+=1
ks=set(keep); sel=np.array([o for o in obs_w if int(o[1]) in ks],float)
fidx={f:i for i,f in enumerate(frames)}; pidx={p:i for i,p in enumerate(keep)}
N,npt=len(frames),len(keep); npar=N*6+npt*3
of=np.array([fidx[int(o[0])] for o in sel],int); op=np.array([pidx[int(o[1])] for o in sel],int)
ouv=sel[:,2:4]; nobs=len(sel)
print("N=%d npt=%d nobs=%d npar=%d"%(N,npt,nobs,npar))
x0=np.zeros(npar)
for f,i in fidx.items():
    r,_=cv2.Rodrigues(P[f][:3,:3]); x0[i*6:i*6+3]=r.ravel(); x0[i*6+3:i*6+6]=P[f][:3,3]
for p,j in pidx.items(): x0[N*6+j*3:N*6+j*3+3]=pts[p]
def rod(rv):
    R=np.zeros((len(rv),3,3))
    for i in range(len(rv)): R[i],_=cv2.Rodrigues(rv[i].reshape(3,1))
    return R
def residual(x):
    pr=x[:N*6].reshape(N,6); p3=x[N*6:].reshape(npt,3)
    R=rod(pr[:,:3])
    Xc=np.einsum("nij,nj->ni",R[of].transpose(0,2,1),p3[op]-pr[:,3:6][of])
    z=Xc[:,2].copy(); z[np.abs(z)<1e-9]=1e-9
    du=FX*Xc[:,0]/z+CX-ouv[:,0]; dv=FY*Xc[:,1]/z+CY-ouv[:,1]
    res=np.empty(2*nobs); res[0::2]=du; res[1::2]=dv; return res
def jacobian(x):
    pr=x[:N*6].reshape(N,6); p3=x[N*6:].reshape(npt,3)
    R=rod(pr[:,:3])
    Xc=np.einsum("nij,nj->ni",R[of].transpose(0,2,1),p3[op]-pr[:,3:6][of])
    X,Y,Z=Xc[:,0],Xc[:,1],Xc[:,2]
    zz=np.where(np.abs(Z)<1e-9,1e-9,Z); iz=1/zz; iz2=iz*iz
    du=np.stack([FX*iz,np.zeros(nobs),-FX*X*iz2],1); dv=np.stack([np.zeros(nobs),FY*iz,-FY*Y*iz2],1)
    J2=np.stack([du,dv],1).reshape(nobs,2,3)
    ru=np.arange(0,2*nobs,2); rv=np.arange(1,2*nobs,2); idx=np.arange(0,nobs*6,6)
    JR=np.einsum("nij,njk->nik",J2,R[of].transpose(0,2,1))
    pc=N*6+op*3
    a=np.empty(nobs*6,int); b=np.empty(nobs*6,int); c=np.empty(nobs*6)
    for t in range(3):
        a[idx+2*t]=ru; a[idx+2*t+1]=rv; b[idx+2*t]=pc+t; b[idx+2*t+1]=pc+t
        c[idx+2*t]=JR[:,0,t]; c[idx+2*t+1]=JR[:,1,t]
    r_pt,c_pt,d_pt=a.copy(),b.copy(),c.copy()
    JT=np.einsum("nij,njk->nik",J2,-R[of].transpose(0,2,1)); fc=of*6
    for t in range(3):
        a[idx+2*t]=ru; a[idx+2*t+1]=rv; b[idx+2*t]=fc+3+t; b[idx+2*t+1]=fc+3+t
        c[idx+2*t]=JT[:,0,t]; c[idx+2*t+1]=JT[:,1,t]
    r_t,c_t,d_t=a.copy(),b.copy(),c.copy()
    Yk=p3[op]-pr[:,3:6][of]
    S=np.zeros((nobs,3,3))
    S[:,0,1]=-Yk[:,2]; S[:,0,2]=Yk[:,1]
    S[:,1,0]=Yk[:,2]; S[:,1,2]=-Yk[:,0]
    S[:,2,0]=-Yk[:,1]; S[:,2,1]=Yk[:,0]
    SR=np.einsum("nij,njk->nik",S,R[of].transpose(0,2,1))
    Jw=np.einsum("nij,njk->nik",J2,SR)
    for t in range(3):
        a[idx+2*t]=ru; a[idx+2*t+1]=rv; b[idx+2*t]=fc+t; b[idx+2*t+1]=fc+t
        c[idx+2*t]=Jw[:,0,t]; c[idx+2*t+1]=Jw[:,1,t]
    r_w,c_w,d_w=a.copy(),b.copy(),c.copy()
    return csr_matrix((np.concatenate([d_pt,d_t,d_w]),(np.concatenate([r_pt,r_t,r_w]),np.concatenate([c_pt,c_t,c_w]))),shape=(2*nobs,npar))
r0=residual(x0)
print("初始残差 rms=%.3f px (期望 ~1)"%np.sqrt((r0**2).mean()))
J=jacobian(x0)
eps=1e-6
for k in [0,1,2,3,4,5, N*6, N*6+1, N*6+2, N*6+3, 500, 1500]:
    xp=x0.copy(); xp[k]+=eps
    num=(residual(xp)-r0)/eps
    ana=J[:,k].toarray().ravel()
    # 稀疏行取子集比较
    idxs=np.where(np.abs(ana)>1e-12)[0][:200]
    if len(idxs)==0: print("param %5d: 全零列"%k); continue
    rel=np.linalg.norm(num[idxs]-ana[idxs])/(np.linalg.norm(num[idxs])+1e-12)
    print("param %5d: rel_err=%.5f  %s"%(k,rel,"OK" if rel<0.01 else "MISMATCH"))
