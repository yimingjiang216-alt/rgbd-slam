# -*- coding: utf-8 -*-
# Rebuild a ba_data.npz-equivalent from map_v2.npz and measure the dense
# Jacobian the TRF (finite-difference) BA in ba.py would have to allocate.
import os
import numpy as np
from collections import defaultdict

SEQ = r"C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_freiburg1_xyz"
d = np.load(os.path.join(SEQ, "map_v2.npz"))
xyz = d["xyz"]; obs = d["obs"]
# obs columns: [point_id, image_index, u, v]  (map_v2 convention)
O = obs[:, [1, 0, 2, 3]]     # -> [frame, point, u, v]  (ba_data convention)
lines = []
lines.append("map_v2: points=%d obs=%d frames=%d" % (len(xyz), len(obs), int(obs[:, 1].max()) + 1))
lines.append("")
lines.append("what ba.py printed: frames=200 points=233856 obs=467712")
for win in (40, 200):
    sel = O[O[:, 0] < win]
    pc = defaultdict(int)
    for r in sel:
        pc[int(r[1])] += 1
    keep = [p for p, c in pc.items() if c >= 3]
    ks = set(keep)
    sel2 = np.array([r for r in sel if int(r[1]) in ks])
    npose, npt = win, len(keep)
    nres = len(sel2) * 2
    npar = npose * 6 + npt * 3
    lines.append("window=%d -> poses=%d points(>=3 obs)=%d obs=%d" % (win, npose, npt, len(sel2)))
    lines.append("   parameters npar = %d  (poses %d + points %d)" % (npar, npose * 6, npt * 3))
    lines.append("   residuals nres = %d" % nres)
    lines.append("   DENSE finite-difference Jacobian nres x npar float64 = %.1f GiB"
                 % (nres * npar * 8 / 1024 ** 3))
    # sparse structure actually needed: each residual touches 1 pose (6) + 1 point (3) = 9 columns
    nnz = nres * 9
    lines.append("   what the dense array actually spends its memory on:")
    lines.append("     non-zero entries per residual row = 9 (1 pose block + 1 point block)")
    lines.append("     real nnz = %d  -> a CSR Jacobian would be %.1f MB (float64 + int32 idx)"
                 % (nnz, nnz * 12 / 1024 ** 2))
    dens = npar / 9.0
    lines.append("     dense/sparse memory ratio = %.0fx  (npar / 9 = %.0f)" % (dens, dens))
    lines.append("")
open("ba_mem_out.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
