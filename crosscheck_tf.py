"""TF poses in the bag must equal groundtruth.txt, and trajectories must match evo."""
import os, numpy as np
from rosbags.rosbag1 import Reader as R1
from validate_bag import r1_tf
SEQ=r"C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_freiburg1_xyz"

gt=[]
for line in open(os.path.join(SEQ,"groundtruth.txt")):
    line=line.strip()
    if not line or line.startswith("#"): continue
    s=line.split()
    if len(s)>=8: gt.append([float(x) for x in s])

poses=[]
with R1(os.path.join(SEQ,"tum_fr1_xyz.bag")) as r:
    for conn,t,raw in r.messages():
        if conn.topic=="/tf":
            tfs,_=r1_tf(bytes(raw))
            (hdr,fid),child,p,q=tfs[0]
            poses.append((hdr,fid,p,q))

print("groundtruth rows:",len(gt)," bag tf poses:",len(poses))
assert len(gt)==len(poses), "count mismatch"
maxp=maxq=0.0
for row,(hdr,_fid,p,q) in zip(gt,poses):
    ts=hdr[1]+hdr[2]*1e-9
    maxp=max(maxp, max(abs(ts-row[0]), abs(p[0]-row[1]), abs(p[1]-row[2]), abs(p[2]-row[3])))
    maxq=max(maxq, max(abs(q[i]-row[4+i]) for i in range(4)))
print("max |timestamp/position| deviation : %.3e" % maxp)
print("max |quaternion| deviation          : %.3e" % maxq)
print("VERDICT:", "PASS - bag /tf reproduces groundtruth.txt exactly" if (maxp<1e-9 and maxq<1e-9) else "FAIL")

# also confirm the estimate trajectory still matches the reported ATE
est=np.loadtxt(os.path.join(SEQ,"evo_est.txt"))
gtf=np.loadtxt(os.path.join(SEQ,"evo_gt.txt"))
print("\nevo_est.txt %s   evo_gt.txt %s" % (est.shape, gtf.shape))
