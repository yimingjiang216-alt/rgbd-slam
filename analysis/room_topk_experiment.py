# -*- coding: utf-8 -*-
"""Room 保守接受实验: 只让内点数最高的 N 条回环边入图, 看 ATE 走向."""
import json
import subprocess
import sys

SEQ = r"C:\slam_data\rgbd_dataset_freiburg1_room"

for n in (3, 5, 10, 20):
    subprocess.run([sys.executable, "slam3.py", "graph", SEQ,
                    "--max_loop_edges", str(n)],
                   stdout=open("results/room_top%d.txt" % n, "w"),
                   stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, "slam3.py", "eval", SEQ],
                   stdout=open("results/room_top%d_eval.txt" % n, "w"),
                   stderr=subprocess.STDOUT, check=True)
    ev = json.load(open(SEQ + "/eval_v3.json"))
    lc = ev["+loopclosure"]
    print("top %-3d edges -> ATE %.4f m, rot med %.1f deg"
          % (n, lc["ate"], lc["rot_med"]))
print("baseline (VO+localBA, no loops): ATE %.4f m" % ev["VO+localBA"]["ate"])
