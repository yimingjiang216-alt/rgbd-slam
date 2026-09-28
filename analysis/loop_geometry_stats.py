# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, r"C:\Users\r26304\Documents\codex-shop\_push_slam")
import numpy as np
import slam2
import loop_detection as ld

SEQ = slam2.SEQ
out = []
def w(s): out.append(s)

d = np.load(os.path.join(SEQ, "map_v2.npz"))
poses = d["poses"]
L = np.load(os.path.join(SEQ, "loops.npz"))
li, lj, LT = L["i"], L["j"], L["T"]

Tq = np.loadtxt(os.path.join(SEQ, "gt_quat.txt"))
gx = np.loadtxt(os.path.join(SEQ, "gt_xyz.txt"))

def gtT(k):
    T = np.eye(4)
    T[:3, :3] = slam2.q2R(Tq[k][3:7])
    T[:3, 3] = gx[k]
    return T

base = []; rotgt = []; te = []; re = []; tv = []; rv = []
for t in range(len(li)):
    i = int(li[t]); j = int(lj[t])
    Ti = gtT(i); Tj = gtT(j)
    dt, dr = ld.pose_difference(Ti, Tj)
    base.append(dt); rotgt.append(dr)
    Zg = np.linalg.inv(Ti) @ Tj
    a, b = ld.pose_difference(Zg, LT[t]); te.append(a); re.append(b)
    a, b = ld.pose_difference(Zg, np.linalg.inv(poses[i]) @ poses[j]); tv.append(a); rv.append(b)

base = np.array(base); rotgt = np.array(rotgt); te = np.array(te); re = np.array(re)
tv = np.array(tv); rv = np.array(rv)
w("baseline          med/mean/max = %.4f / %.4f / %.4f m" % (np.median(base), base.mean(), base.max()))
w("GT rel rot        med/mean/max = %.2f / %.2f / %.2f deg" % (np.median(rotgt), rotgt.mean(), rotgt.max()))
w("PnP loop pose vs GT  trans %.4f / %.4f m" % (np.median(te), te.mean()))
w("PnP loop pose vs GT  rot   %.2f / %.2f deg" % (np.median(re), re.mean()))
w("VO rel pose   vs GT  trans %.4f / %.4f m" % (np.median(tv), tv.mean()))
w("VO rel pose   vs GT  rot   %.2f / %.2f deg" % (np.median(rv), rv.mean()))
w("PnP rotation beats VO on %d / %d loops" % (int((re < rv).sum()), len(re)))
for thr in (0.0, 0.15, 0.20, 0.25):
    m = base >= thr
    if m.sum() == 0:
        w("baseline >= %.2f : none" % thr); continue
    w("baseline >= %.2f : %3d pairs | PnP rot %.2f deg | VO rot %.2f deg" % (thr, int(m.sum()), np.median(re[m]), np.median(rv[m])))
w("est path %.3f m | gt path %.3f m" % (float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum()), float(np.linalg.norm(np.diff(gx, axis=0), axis=1).sum())))

open("stats_extra_out.txt", "w", encoding="utf-8").write("\n".join(out))
print("done")
