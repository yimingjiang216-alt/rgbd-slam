import numpy as np, os
seq=r"C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_freiburg1_xyz"
est=np.loadtxt(os.path.join(seq,"rgbd_traj_est.txt"))
gt=np.loadtxt(os.path.join(seq,"rgbd_traj_gt.txt"))
print("frame | est                    | gt                     | err(m)")
for f in [0,50,100,200,300,400,500,600,700,797]:
    print("%5d | %s | %s | %.4f"%(f,np.round(est[f],3),np.round(gt[f],3),np.linalg.norm(est[f]-gt[f])))
e=np.linalg.norm(est-gt,axis=1)
print("\nATE RMSE=%.4f m  max=%.4f m  median=%.4f m"%(np.sqrt((e**2).mean()),e.max(),np.median(e)))
print("traj extent GT: X[%.2f,%.2f] Y[%.2f,%.2f] Z[%.2f,%.2f]"%(gt[:,0].min(),gt[:,0].max(),gt[:,1].min(),gt[:,1].max(),gt[:,2].min(),gt[:,2].max()))
print("traj extent EST: X[%.2f,%.2f] Y[%.2f,%.2f] Z[%.2f,%.2f]"%(est[:,0].min(),est[:,0].max(),est[:,1].min(),est[:,1].max(),est[:,2].min(),est[:,2].max()))
print("drift: net displacement GT=%.3f m EST=%.3f m"%(np.linalg.norm(gt[-1]-gt[0]),np.linalg.norm(est[-1]-est[0])))
print("relative drift = %.2f%%"%(100*np.linalg.norm(est[-1]-gt[-1])/np.linalg.norm(np.diff(gt,axis=0),axis=1).sum()))
