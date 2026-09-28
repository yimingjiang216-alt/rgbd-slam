# -*- coding: utf-8 -*-
"""Validate the TUM -> rosbag conversion.

Reads back the ROS1 .bag and the ROS2 .db3 that tum2bag.py produced and checks
them against the source text files: message counts per topic, timestamp
fidelity, image payload bytes, camera intrinsics and /tf poses.

Run:  python validate_bag.py
"""
import os
import struct
import numpy as np

SEQ = r"C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_freiburg1_xyz"
ROS2_DIR = os.path.join(SEQ, "tum_fr1_xyz_ros2")
BAG1 = os.path.join(SEQ, "tum_fr1_xyz.bag")
K_EXPECT = (517.3, 0.0, 318.6, 0.0, 516.5, 255.3, 0.0, 0.0, 1.0)


# ------------------------------------------------------------- deserialisers

class Blob:
    """Little-endian reader for the ROS1 binary message layout."""

    def __init__(self, b):
        self.b = b
        self.i = 0

    def u32(self):
        v = struct.unpack_from("<I", self.b, self.i)[0]
        self.i += 4
        return v

    def f64(self):
        v = struct.unpack_from("<d", self.b, self.i)[0]
        self.i += 8
        return v

    def f64s(self, n):
        v = struct.unpack_from("<%dd" % n, self.b, self.i)
        self.i += 8 * n
        return list(v)

    def u8(self):
        v = self.b[self.i]
        self.i += 1
        return v

    def string(self):
        """ROS1 strings are length-prefixed; the length excludes the NUL."""
        n = self.u32()
        s = self.b[self.i:self.i + n].decode("utf-8", "replace")
        self.i += n
        return s

    def header(self):
        seq = self.u32()
        sec = self.u32()
        nsec = self.u32()
        frame_id = self.string()
        return (seq, sec, nsec), frame_id


def r1_image(raw):
    """sensor_msgs/Image (ROS1)."""
    bl = Blob(raw)
    header = bl.header()
    height = bl.u32()
    width = bl.u32()
    encoding = bl.string()
    is_bigendian = bl.u8()
    step = bl.u32()
    n = bl.u32()
    data = bl.b[bl.i:bl.i + n]
    bl.i += n
    return dict(header=header, height=height, width=width,
                encoding=encoding, is_bigendian=is_bigendian,
                step=step, data=data), bl.i


def r1_caminfo(raw):
    """sensor_msgs/CameraInfo (ROS1).

    Layout here is: Header, height, width, distortion_model (string),
    D (count-prefixed float64[]), K (fixed float64[9]), R (float64[9]),
    P (float64[12]), then binning/ROI.  The count prefix appears only on D
    because in ROS1's generated code the fixed-size arrays carry no length,
    while the variable-size D does.
    """
    bl = Blob(raw)
    header = bl.header()
    height = bl.u32()
    width = bl.u32()
    dist_model = bl.string()
    nD = bl.u32()
    D = bl.f64s(nD)
    K = bl.f64s(9)
    R = bl.f64s(9)
    P = bl.f64s(12)
    return dict(header=header, height=height, width=width,
                distortion_model=dist_model, D=D, K=K, R=R, P=P), bl.i


def r1_tf(raw):
    """tf2_msgs/TFMessage (ROS1): TransformStamped[]."""
    bl = Blob(raw)
    n = bl.u32()
    out = []
    for _ in range(n):
        header = bl.header()
        child = bl.string()
        p = bl.f64s(3)
        q = bl.f64s(4)
        out.append((header, child, p, q))
    return out, bl.i


# ------------------------------------------------------------------- helpers

def read_list(p):
    out = []
    for line in open(p):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        s = line.split()
        if len(s) >= 2:
            out.append((float(s[0]), s[1]))
    return out


def ros2_message_counts(db3):
    """Messages per topic in a ROS2 sqlite3 bag (no rosbags needed)."""
    import sqlite3
    con = sqlite3.connect(db3)
    rows = con.execute(
        "SELECT t.name, COUNT(*) FROM messages m "
        "JOIN topics t ON t.id = m.topic_id GROUP BY t.name ORDER BY t.name"
    ).fetchall()
    total = con.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    con.close()
    return rows, total


# ---------------------------------------------------------------------- main

def main():
    print("=" * 70)
    print("TUM fr1/xyz  ->  rosbag conversion check")
    print("=" * 70)
    bad = 0
    rgb = read_list(os.path.join(SEQ, "rgb.txt"))
    depth = read_list(os.path.join(SEQ, "depth.txt"))
    gt = []
    for line in open(os.path.join(SEQ, "groundtruth.txt")):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        s = line.split()
        if len(s) >= 8:
            gt.append([float(x) for x in s])
    print("source: %d rgb, %d depth, %d groundtruth"
          % (len(rgb), len(depth), len(gt)))

    if not os.path.exists(BAG1):
        print("missing %s" % BAG1)
        return 1
    from rosbags.rosbag1 import Reader as R1

    # ---- pass 1: counts, encodings, camera info
    print("\n--- ROS1  %s" % os.path.basename(BAG1))
    n_img = n_dep = n_ci = n_tf = 0
    info_ok = None
    topics = set()
    with R1(BAG1) as r:
        for conn, t, raw in r.messages():
            topic = conn.topic
            topics.add(topic)
            if topic.endswith("image_color"):
                n_img += 1
                if n_img == 1:
                    m, used = r1_image(bytes(raw))
                    print("  rgb    encoding=%s %dx%d step=%d data=%d B"
                          % (m["encoding"], m["width"], m["height"],
                             m["step"], len(m["data"])))
                    if used != len(raw):
                        print("  note: parsed %d of %d bytes" % (used, len(raw)))
            elif topic.endswith("depth/image"):
                n_dep += 1
                if n_dep == 1:
                    m, _ = r1_image(bytes(raw))
                    print("  depth  encoding=%s %dx%d data=%d B"
                          % (m["encoding"], m["width"], m["height"],
                             len(m["data"])))
            elif topic.endswith("camera_info"):
                n_ci += 1
                if info_ok is None:
                    m, _ = r1_caminfo(bytes(raw))
                    info_ok = m
            elif topic == "/tf":
                n_tf += 1

    if info_ok is not None:
        K = tuple(round(v, 6) for v in info_ok["K"])
        print("  caminfo model=%s K=%s" % (info_ok["distortion_model"], K))
        if all(abs(a - b) < 1e-9 for a, b in zip(K, K_EXPECT)):
            print("  K matches the TUM fr1 intrinsics exactly")
        else:
            print("  FAIL K does not match expected intrinsics"); bad += 1
    else:
        print("  FAIL no camera_info found"); bad += 1

    print("  counts: rgb=%d depth=%d caminfo=%d tf=%d  (total %d)"
          % (n_img, n_dep, n_ci, n_tf, n_img + n_dep + n_ci + n_tf))
    print("  per-frame rgb matches source: %s" % (n_img == len(rgb)))
    print("  per-frame depth matches source: %s" % (n_dep == len(depth)))
    print("  every groundtruth row has a /tf pose: %s" % (n_tf == len(gt)))
    if n_img != len(rgb):
        bad += 1
    if n_dep != len(depth):
        bad += 1
    if n_tf != len(gt):
        bad += 1

    # ---- ROS2
    print("\n--- ROS2  %s" % os.path.basename(ROS2_DIR))
    db3s = []
    for root, _dirs, files in os.walk(ROS2_DIR):
        for f in files:
            if f.endswith(".db3"):
                db3s.append(os.path.join(root, f))
    if not db3s:
        print("  FAIL no .db3 found"); bad += 1
    grand = 0
    for d in sorted(db3s):
        rows, total = ros2_message_counts(d)
        print("  %s: %d messages in %d topics"
              % (os.path.basename(d), total, len(rows)))
        for name, c in rows:
            print("      %-26s %d" % (name, c))
        grand += total
    print("  total %d messages (ROS1 had %d)" % (grand, n_img + n_dep + n_ci + n_tf))
    if grand != n_img + n_dep + n_ci + n_tf:
        print("  FAIL ROS2/ROS1 message count differs"); bad += 1

    # ---- pass 2: pixel payload spot check
    print("\n--- pixel payload spot check (ROS1 vs source PNG)")
    import cv2
    import random
    random.seed(0)
    pick_r = sorted(random.sample(range(len(rgb)), 6))
    pick_d = sorted(random.sample(range(len(depth)), 6))
    got_r, got_d = {}, {}
    nr = nd = 0
    with R1(BAG1) as r:
        for conn, t, raw in r.messages():
            if conn.topic.endswith("image_color"):
                if nr in pick_r:
                    m, _ = r1_image(bytes(raw))
                    got_r[nr] = (np.frombuffer(m["data"], np.uint8)
                                 .reshape(480, 640, 3).copy(), m["header"])
                nr += 1
            elif conn.topic.endswith("depth/image"):
                if nd in pick_d:
                    m, _ = r1_image(bytes(raw))
                    got_d[nd] = (np.frombuffer(m["data"], np.uint16)
                                 .reshape(480, 640).copy(), m["header"])
                nd += 1
    for i, (arr, hdr) in sorted(got_r.items()):
        sec, nsec = hdr[0][1], hdr[0][2]
        ref = cv2.imread(os.path.join(SEQ, rgb[i][1]), cv2.IMREAD_COLOR)
        same = np.array_equal(arr, ref)
        ts_ok = abs((sec + nsec * 1e-9) - rgb[i][0]) < 1e-9
        print("  rgb[%3d] %-22s pixels_identical=%-5s timestamp_ok=%s"
              % (i, rgb[i][1], same, ts_ok))
        bad += (not same) or (not ts_ok)
    for i, (arr, hdr) in sorted(got_d.items()):
        sec, nsec = hdr[0][1], hdr[0][2]
        ref = cv2.imread(os.path.join(SEQ, depth[i][1]),
                         cv2.IMREAD_UNCHANGED).astype(np.uint16)
        same = np.array_equal(arr, ref)
        ts_ok = abs((sec + nsec * 1e-9) - depth[i][0]) < 1e-9
        print("  dep[%3d] %-22s pixels_identical=%-5s timestamp_ok=%s"
              % (i, depth[i][1], same, ts_ok))
        bad += (not same) or (not ts_ok)

    # ---- pass 3: /tf vs groundtruth.txt
    print("\n--- /tf vs groundtruth.txt")
    poses = []
    with R1(BAG1) as r:
        for conn, t, raw in r.messages():
            if conn.topic == "/tf":
                tfs, _ = r1_tf(bytes(raw))
                (hdr, fid), child, p, q = tfs[0]
                sec, nsec = hdr[1], hdr[2]
                poses.append((sec + nsec * 1e-9, p, q))
    if len(poses) != len(gt):
        print("  FAIL pose count %d != %d" % (len(poses), len(gt))); bad += 1
    else:
        mp = mq = 0.0
        for row, (ts, p, q) in zip(gt, poses):
            mp = max(mp, abs(ts - row[0]), abs(p[0] - row[1]),
                     abs(p[1] - row[2]), abs(p[2] - row[3]))
            mq = max(mq, max(abs(q[k] - row[4 + k]) for k in range(4)))
        print("  max |timestamp/position| deviation %.3e" % mp)
        print("  max |quaternion| deviation           %.3e" % mq)
        if mp > 1e-9 or mq > 1e-9:
            print("  FAIL /tf does not reproduce groundtruth.txt"); bad += 1
        else:
            print("  /tf reproduces groundtruth.txt exactly")

    print("\n" + "=" * 70)
    print("VERDICT:", "PASS" if bad == 0 else "FAIL (%d problems)" % bad)
    print("=" * 70)
    return bad


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
