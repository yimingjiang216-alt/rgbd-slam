# -*- coding: utf-8 -*-
"""
TUM RGB-D -> rosbag converter (pure Python, no ROS install needed).

Outputs:
  tum_fr1_xyz_ros2/   ROS2 bag (sqlite3, ROS2 Humble type definitions)
  tum_fr1_xyz.bag     ROS1 bag (ROS1 Noetic type definitions)

Topics:
  /camera/rgb/image_color     sensor_msgs/Image        (bgr8)
  /camera/depth/image         sensor_msgs/Image        (16UC1, raw TUM units)
  /camera/rgb/camera_info     sensor_msgs/CameraInfo
  /tf                         tf2_msgs/TFMessage       (ground-truth pose)

Usage: python tum2bag.py <TUM sequence dir> [ros2|ros1|both]
"""
import os
import sys
import shutil
import numpy as np
import cv2
from rosbags.rosbag2 import Writer as Writer2
from rosbags.rosbag1 import Writer as Writer1
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

SEQ = r"C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_freiburg1_xyz"
K = (517.3, 516.5, 318.6, 255.3)          # fx, fy, cx, cy  (TUM fr1)
DEPTH_SCALE = 5000.0                      # depth_png / 5000 = metres

TFMSG_DEF = """geometry_msgs/TransformStamped[] transforms
================================================================================
MSG: geometry_msgs/TransformStamped
std_msgs/Header header
string child_frame_id
geometry_msgs/Transform transform
================================================================================
MSG: geometry_msgs/Transform
geometry_msgs/Vector3 translation
geometry_msgs/Quaternion rotation
"""


def read_list(path):
    out = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        s = line.split()
        if len(s) >= 2:
            out.append((float(s[0]), s[1]))
    return out


def read_gt(path):
    out = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        s = line.split()
        if len(s) >= 8:
            out.append([float(x) for x in s])
    return out


def _get(ts, name, kind):
    """Return the message class for a ROS1 or ROS2 typestore."""
    key = name if kind == "ros2" else name
    try:
        return ts.types[key]
    except KeyError:
        return None


def build_messages(seq, ts, kind):
    """Build every message once as (topic, timestamp_ns, serialized, msgtype)."""
    # ROS1 needs tf2_msgs registered by hand; ROS2 ships it.
    if kind == "ros1" and "tf2_msgs/msg/TFMessage" not in ts.types:
        ts.register(get_types_from_msg(TFMSG_DEF, "tf2_msgs/msg/TFMessage"))

    Image = ts.types["sensor_msgs/msg/Image"]
    CameraInfo = ts.types["sensor_msgs/msg/CameraInfo"]
    TFMessage = ts.types["tf2_msgs/msg/TFMessage"]
    TransformStamped = ts.types["geometry_msgs/msg/TransformStamped"]
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    Transform = ts.types["geometry_msgs/msg/Transform"]
    Vector3 = ts.types["geometry_msgs/msg/Vector3"]
    Quaternion = ts.types["geometry_msgs/msg/Quaternion"]
    RegionOfInterest = ts.types["sensor_msgs/msg/RegionOfInterest"]

    # ROS1 Header carries `seq`; ROS2 does not. Key names also differ in
    # camera_info (D/K/R/P vs d/k/r/p).
    hdr_fields = set(Header.__dataclass_fields__)
    ci_fields = set(CameraInfo.__dataclass_fields__)
    serialize = ts.serialize_ros1 if kind == "ros1" else ts.serialize_cdr

    def make_header(t, frame_id, seq_id=0):
        kw = dict(stamp=Time(sec=int(t), nanosec=int(round((t % 1) * 1e9))),
                  frame_id=frame_id)
        if "seq" in hdr_fields:
            kw["seq"] = seq_id
        return Header(**kw)

    rgb = read_list(os.path.join(seq, "rgb.txt"))
    depth = read_list(os.path.join(seq, "depth.txt"))
    gt = read_gt(os.path.join(seq, "groundtruth.txt"))
    fx, fy, cx, cy = K

    msgs = []

    for seq_id, (t, p) in enumerate(rgb):
        img = cv2.imread(os.path.join(seq, p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        header = make_header(t, "camera_rgb_optical_frame", seq_id)
        m = Image(header=header, height=h, width=w, encoding="bgr8",
                  is_bigendian=0, step=w * 3,
                  data=np.frombuffer(img.tobytes(), np.uint8))
        msgs.append(("/camera/rgb/image_color", int(t * 1e9),
                     serialize(m, "sensor_msgs/msg/Image"),
                     "sensor_msgs/msg/Image"))

        ci_kw = dict(header=header, height=h, width=w,
                     distortion_model="plumb_bob", binning_x=0, binning_y=0,
                     roi=RegionOfInterest(x_offset=0, y_offset=0, height=0,
                                          width=0, do_rectify=False))
        ci_kw["D" if "D" in ci_fields else "d"] = np.array([0.0] * 5, np.float64)
        ci_kw["K" if "K" in ci_fields else "k"] = np.array(
            [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0], np.float64)
        ci_kw["R" if "R" in ci_fields else "r"] = np.eye(3, dtype=np.float64).ravel()
        ci_kw["P" if "P" in ci_fields else "p"] = np.array(
            [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0], np.float64)
        msgs.append(("/camera/rgb/camera_info", int(t * 1e9),
                     serialize(CameraInfo(**ci_kw), "sensor_msgs/msg/CameraInfo"),
                     "sensor_msgs/msg/CameraInfo"))

    for seq_id, (t, p) in enumerate(depth):
        img = cv2.imread(os.path.join(seq, p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        h, w = img.shape
        header = make_header(t, "camera_rgb_optical_frame", seq_id)
        m = Image(header=header, height=h, width=w, encoding="16UC1",
                  is_bigendian=0, step=w * 2,
                  data=np.frombuffer(img.astype(np.uint16).tobytes(), np.uint8))
        msgs.append(("/camera/depth/image", int(t * 1e9),
                     serialize(m, "sensor_msgs/msg/Image"),
                     "sensor_msgs/msg/Image"))

    for seq_id, row in enumerate(gt):
        t = row[0]
        header = make_header(t, "world", seq_id)
        tr = TransformStamped(
            header=header,
            child_frame_id="camera_rgb_optical_frame",
            transform=Transform(
                translation=Vector3(x=row[1], y=row[2], z=row[3]),
                rotation=Quaternion(x=row[4], y=row[5], z=row[6], w=row[7])))
        msgs.append(("/tf", int(t * 1e9),
                     serialize(TFMessage(transforms=[tr]), "tf2_msgs/msg/TFMessage"),
                     "tf2_msgs/msg/TFMessage"))

    msgs.sort(key=lambda x: x[1])
    return msgs


def write_ros2(msgs, out, ts):
    if os.path.exists(out):
        shutil.rmtree(out)
    conns = {}
    with Writer2(out, version=8) as w:
        for topic, _, _, mt in msgs:
            if topic not in conns:
                conns[topic] = w.add_connection(topic, mt, typestore=ts)
        for topic, tns, raw, _ in msgs:
            w.write(conns[topic], tns, raw)
    print("wrote ROS2 bag: %s" % out, flush=True)


def write_ros1(msgs, out, ts):
    if os.path.exists(out):
        os.remove(out)
    conns = {}
    with Writer1(out) as w:
        for topic, _, _, mt in msgs:
            if topic not in conns:
                conns[topic] = w.add_connection(topic, mt, typestore=ts)
        for topic, tns, raw, _ in msgs:
            w.write(conns[topic], tns, raw)
    print("wrote ROS1 bag: %s (%.1f MB)"
          % (out, os.path.getsize(out) / 1e6), flush=True)


def main(seq=SEQ, fmt="both"):
    print("reading", seq, flush=True)
    if fmt in ("ros2", "both"):
        ts2 = get_typestore(Stores.ROS2_HUMBLE)
        write_ros2(build_messages(seq, ts2, "ros2"),
                   os.path.join(seq, "tum_fr1_xyz_ros2"), ts2)
    if fmt in ("ros1", "both"):
        ts1 = get_typestore(Stores.ROS1_NOETIC)
        write_ros1(build_messages(seq, ts1, "ros1"),
                   os.path.join(seq, "tum_fr1_xyz.bag"), ts1)


if __name__ == "__main__":
    s = sys.argv[1] if len(sys.argv) > 1 else SEQ
    f = sys.argv[2] if len(sys.argv) > 2 else "both"
    main(s, f)
