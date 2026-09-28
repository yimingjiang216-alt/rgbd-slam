# -*- coding: utf-8 -*-
"""清洗地图点: 保留三角化质量好的点 + 剔除外点残差
判据:
 1) 点在相机前方 (z>0) 且深度合理 (0.3m .. 30m) —— TUM fr1 场景室内几米量级
 2) 在相机视野内 (投影落在图像范围)
 3) 用 VO 位姿重投影误差 < 阈值 (RANSAC 式剔除)
产出新的 ba_data_clean.npz
"""
import numpy as np, os, sys, cv2
from collections import defaultdict
seq = r"C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_freiburg1_xyz"
d = np.load(os.path.join(seq, "ba_data.npz"))
poses, points, obs, K = d["poses"], d["points"], d["obs"], d["K"]
fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
F=len(poses)
print("raw: frames=%d points=%d obs=%d"%(F,len(points),len(obs)))

nf=np.isfinite(points).all(1)
print("finite points: %d"%nf.sum())

# 逐观测计算残差
pf=obs[:,1].astype(int); ff=obs[:,0].astype(int)
R=np.array([p[:3,:3] for p in poses]); T=np.array([p[:3,3] for p in poses])
Xw=points[pf]
Xc=np.einsum("nij,nj->ni",R[ff],Xw)+T[ff]
z=Xc[:,2].copy()
zsafe=z.copy(); zsafe[np.abs(zsafe)<1e-6]=1e-6
uv=np.stack([fx*Xc[:,0]/zsafe+cx, fy*Xc[:,1]/zsafe+cy],1)
err=np.linalg.norm(uv-obs[:,2:4],axis=1)
valid = (z>0.3)&(z<30)&(uv[:,0]>=0)&(uv[:,0]<640)&(uv[:,1]>=0)&(uv[:,1]<480)&(err<3.0)
print("valid obs: %d / %d (%.1f%%)"%(valid.sum(),len(obs),100*valid.mean()))

# 每个点的有效观测数
pc=defaultdict(int)
for p in pf[valid]: pc[int(p)]+=1
good=[p for p,c in pc.items() if c>=2]
print("points with >=2 valid obs: %d"%len(good))
goodset=set(good)
keep = valid & np.array([int(p) in goodset for p in pf])
print("final obs: %d"%keep.sum())
obs2=obs[keep]
# 只保留被用到且在原 obs 里的点
used=sorted(set(int(p) for p in obs2[:,1]))
remap={p:i for i,p in enumerate(used)}
pts2=points[used]
obs2=obs2.copy(); obs2[:,1]=np.array([remap[int(p)] for p in obs2[:,1]])
print("clean: points=%d obs=%d"%(len(pts2),len(obs2)))
n=np.linalg.norm(pts2,axis=1)
print("clean point norm: median=%.2f p90=%.2f p99=%.2f"%(np.median(n),np.percentile(n,90),np.percentile(n,99)))
ccs=defaultdict(int)
for p in obs2[:,1].astype(int): ccs[p]+=1
v=np.array(list(ccs.values()))
print("obs per point: mean=%.2f median=%.0f max=%d"%(v.mean(),np.median(v),v.max()))
print("pts >=3 obs: %d ; >=5: %d ; >=10: %d"%((v>=3).sum(),(v>=5).sum(),(v>=10).sum()))
cov=len(set(obs2[:,0].astype(int)))
print("frame coverage: %d frames"%cov)
np.savez(os.path.join(seq,"ba_data_clean.npz"), poses=poses, points=pts2, obs=obs2, K=K)
print("saved ba_data_clean.npz")
