import numpy as np, os, sys
src = r'data\rgbd_dataset_freiburg1_desk'
def read_list(p):
    out=[]
    for ln in open(p):
        ln=ln.strip()
        if not ln or ln.startswith('#'): continue
        s=ln.split()
        if len(s)>=2: out.append((float(s[0]), s[1]))
    return out
rgb = read_list(os.path.join(src,'rgb.txt'))
gt=[]
for ln in open(os.path.join(src,'groundtruth.txt')):
    ln=ln.strip()
    if not ln or ln.startswith('#'): continue
    s=ln.split()
    if len(s)>=8: gt.append((float(s[0]), np.array([float(x) for x in s[1:8]])))
print('rgb frames %d  gt samples %d' % (len(rgb), len(gt)))
tgt=np.array([g[0] for g in gt]); vgt=np.array([g[1] for g in gt])
trgb=np.array([r[0] for r in rgb])
# nearest-neighbour interpolation of GT onto rgb timestamps, then drop
# timestamps outside the GT range / too far from any GT sample
idx=np.searchsorted(tgt, trgb).clip(1,len(tgt)-1)
lo=idx-1; hi=idx
choose=np.where(np.abs(tgt[lo]-trgb)<=np.abs(tgt[hi]-trgb), lo, hi)
gap=np.abs(tgt[choose]-trgb)
keep=gap < 0.02
print('kept %d / %d rgb frames (max gt gap %.4f s)' % (keep.sum(), len(trgb), gap[keep].max() if keep.sum() else -1))
sel=np.where(keep)[0]
np.savetxt(os.path.join(src,'gt_xyz.txt'), vgt[choose[sel]][:,:3], fmt='%.18e')
np.savetxt(os.path.join(src,'gt_quat.txt'), vgt[choose[sel]], fmt='%.18e')
print('wrote gt_xyz.txt, gt_quat.txt  (%d rows)' % len(sel))
print('GT path length %.3f m' % np.linalg.norm(np.diff(vgt[choose[sel]][:,:3],axis=0),axis=1).sum())
