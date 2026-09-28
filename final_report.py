import numpy as np, os, sys
sys.path.insert(0, r"C:\Users\r26304\Documents\codex-shop\slam_vo")
from slam2 import se3_align, rot_angle, q2R
seq=r"C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_freiburg1_xyz"
gt=np.loadtxt(os.path.join(seq,"gt_xyz.txt"))
Tq=np.loadtxt(os.path.join(seq,"gt_quat.txt"))
Rg=np.array([q2R(Tq[k][3:7]) for k in range(len(Tq))])
# 只有一个结果(在线BA已写进map_v2), 与历史记录对比
P=np.load(os.path.join(seq,"map_v2.npz"))["poses"]
est=P[:,:3,3]
R_,t_,_=se3_align(est,gt); al=(R_@est.T).T+t_
e=np.linalg.norm(al-gt,axis=1)
rot=np.array([rot_angle(P[k][:3,:3].T@(Rg[0].T@Rg[k])) for k in range(0,len(P),10)])
print("===== 最终结果 (在线局部BA) =====")
print("  ATE RMSE   = %.4f m"%np.sqrt((e**2).mean()))
print("  ATE mean   = %.4f m"%e.mean())
print("  ATE max    = %.4f m"%e.max())
print("  旋转误差中位数 = %.2f deg"%np.median(rot))
print("  轨迹长度   = %.3f m  (真值 8.011)"%np.linalg.norm(np.diff(al,axis=0),axis=1).sum())
print("  相对误差   = %.2f %%"%(100*np.sqrt((e**2).mean())/8.011))
# 保存对齐后轨迹与图
np.savetxt(os.path.join(seq,"final_traj.txt"),al)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(13,6))
ax[0].plot(gt[:,0],gt[:,2],"g-",lw=2.5,label="ground truth")
ax[0].plot(al[:,0],al[:,2],"r-",lw=1.8,label="RGB-D SLAM w/ local BA (ATE=%.3f m)"%np.sqrt((e**2).mean()))
ax[0].set_xlabel("X (m)"); ax[0].set_ylabel("Z (m)")
ax[0].set_title("TUM fr1/xyz: RGB-D SLAM vs Ground Truth")
ax[0].legend(); ax[0].grid(True); ax[0].axis("equal")
ax[1].plot(e,"r-"); ax[1].set_xlabel("frame"); ax[1].set_ylabel("position error (m)")
ax[1].set_title("Position error over time (mean %.3f m)"%e.mean()); ax[1].grid(True)
plt.tight_layout(); plt.savefig(os.path.join(seq,"final_result.png"),dpi=150)
print("  图:",os.path.join(seq,"final_result.png"))
