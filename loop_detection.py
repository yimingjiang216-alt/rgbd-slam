# -*- coding: utf-8 -*-
"""Loop-closure detection for the RGB-D pipeline.

Given a set of keyframes (image index, pose, ORB descriptors of the map points
spawned at that keyframe) this searches for places the camera has revisited and
estimates the relative pose between the two visits.

The three stages are the usual ones:

  1. candidate search - a keyframe only looks at keyframes that are far away
     *along the trajectory* (so neighbours are excluded) but close *in space*
     according to the current estimate.  Nothing else can be a loop.
  2. descriptor matching - ORB descriptors of the two keyframes are matched
     with a ratio test, then the 3D points behind the matched keypoints are
     tested with PnP/RANSAC.  PnP is what actually decides: it needs enough
     inliers before a candidate is accepted.
  3. geometric verification - the relative pose from PnP has to agree with the
     pose the odometry already predicts, otherwise the match is a repeated
     texture (a poster, a tiled floor) rather than a real revisit.

This module deliberately does NOT decide what to do with the constraints; it
only produces them.  pose_graph.py optimises them.
"""
import numpy as np
import cv2

# ---------------------------------------------------------------- ORB helpers


def orb_features(gray, nfeatures=2000):
    """Detect ORB keypoints/descriptors on an 8-bit grayscale image."""
    orb = cv2.ORB_create(nfeatures=nfeatures)
    return orb.detectAndCompute(gray, None)


def match_descriptors(des_a, des_b, ratio=0.75):
    """Ratio-test match.  Returns (idx_a, idx_b) arrays of matched rows."""
    if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
        return np.empty(0, int), np.empty(0, int)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = bf.knnMatch(des_a, des_b, k=2)
    ia, ib = [], []
    for pr in pairs:
        if len(pr) < 2:
            continue
        m, n = pr
        if m.distance < ratio * n.distance:
            ia.append(m.queryIdx)
            ib.append(m.trainIdx)
    return np.array(ia, int), np.array(ib, int)


def backproject(u, v, d, K):
    """Pixel + depth -> camera-frame 3D point, or None if depth is invalid."""
    if d <= 0:
        return None
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([(u - cx) * d / fx, (v - cy) * d / fy, d])


def pose_from_pnp(pts3d, pts2d, K, ransac_px=2.0, iterations=2000,
                  confidence=0.999):
    """PnP + RANSAC.  Returns dict(ok, T, inliers, inlier_ratio, reproj)."""
    if len(pts3d) < 6:
        return dict(ok=False, T=None, inliers=np.empty(0, int),
                    inlier_ratio=0.0, reproj=np.inf)
    dist = np.zeros(5)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        np.asarray(pts3d, np.float64).reshape(-1, 1, 3),
        np.asarray(pts2d, np.float64).reshape(-1, 1, 2),
        K, dist,
        iterationsCount=iterations,
        reprojectionError=ransac_px,
        confidence=confidence,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or inl is None or len(inl) < 6:
        return dict(ok=False, T=None, inliers=np.empty(0, int),
                    inlier_ratio=0.0, reproj=np.inf)

    inl = inl.ravel()
    # refine on the inliers only, with a proper LM solve
    rvec, tvec = cv2.solvePnPRefineLM(
        np.asarray(pts3d, np.float64).reshape(-1, 1, 3)[inl],
        np.asarray(pts2d, np.float64).reshape(-1, 1, 2)[inl],
        K, dist, rvec, tvec)

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()

    proj, _ = cv2.projectPoints(
        np.asarray(pts3d, np.float64).reshape(-1, 1, 3)[inl], rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - np.asarray(pts2d, np.float64)[inl],
                         axis=1)
    return dict(ok=True, T=T, inliers=inl,
                inlier_ratio=float(len(inl)) / len(pts3d),
                reproj=float(np.median(err)))

# ------------------------------------------------------------ relative pose


def relative_pose(T_wi, T_wj):
    """T_i^-1 T_j, the motion from frame i to frame j (i.e. Z_ij)."""
    return np.linalg.inv(T_wi) @ T_wj


def pose_difference(T_a, T_b):
    """Translation (m) and rotation (deg) between two camera-to-world poses."""
    dt = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
    dR = T_a[:3, :3].T @ T_b[:3, :3]
    c = (np.trace(dR) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    return dt, float(np.degrees(np.arccos(c)))


# ---------------------------------------------------------- candidate search


def find_candidates(kf_ids, poses, cur, min_gap=20, max_dist=0.6):
    """Keyframes that could close a loop with `cur`.

    min_gap  : at least this many keyframes of trajectory separation, so that
               temporal neighbours (which always match) are never returned.
    max_dist : the two visits must already be close in the *estimated* pose,
               because the estimate is all we have before verification.
    """
    out = []
    T_cur = poses[cur]
    for k in kf_ids:
        if k >= cur:
            continue
        # trajectory separation, measured in keyframe counts
        if sum(1 for x in kf_ids if k < x < cur) < min_gap:
            continue
        dt, dr = pose_difference(poses[k], T_cur)
        if dt <= max_dist and dr <= 45.0:
            out.append((k, dt, dr))
    out.sort(key=lambda z: z[1])
    return out


# ------------------------------------------------------- the whole detector


def detect_loops(kf_ids, poses, gray, map_xyz, obs, min_gap=20, max_dist=0.6,
                 ratio=0.75, ransac_px=2.0, min_inliers=25,
                 min_inlier_ratio=0.15, agree_trans=0.35, agree_rot_deg=35.0,
                 verbose=False):
    """Search every keyframe for loops back to earlier keyframes.

    kf_ids  : sorted list of keyframe *image* indices
    poses   : dict image_index -> 4x4 camera-to-world pose (current estimate)
    gray    : dict image_index -> uint8 grayscale image
    map_xyz : (M,3) 3D positions of map points, indexed by point id
    obs     : (N,4) columns [point_id, image_index, u, v]

    Returns a list of dicts: i, j, T_ij (i->j relative pose), inliers,
    inlier_ratio, reproj, cand_dist.
    """
    # point ids spawned at each keyframe: we match in image space against the
    # keyframe that produced the point, because that is where we have its uv.
    uv_at_kf = {}
    for pid, fidx, u, v in obs:
        fidx = int(fidx)
        if fidx in gray:
            uv_at_kf[(fidx, int(pid))] = (u, v)

    # which points are visible in each keyframe (for the PnP side)
    pts_in_kf = {}
    for pid, fidx, u, v in obs:
        fidx = int(fidx)
        if fidx in gray:
            pts_in_kf.setdefault(fidx, {})[int(pid)] = (u, v)

    kfset = set(kf_ids)
    K_ = None
    results = []

    for cur in kf_ids:
        cands = find_candidates(kf_ids, poses, cur, min_gap=min_gap,
                                max_dist=max_dist)
        if not cands:
            continue
        kp_cur, des_cur = orb_features(gray[cur])
        if des_cur is None or len(kp_cur) < 20:
            continue
        for cand, cdist, cang in cands:
            # --- 2. descriptor matching against the candidate keyframe
            kp_c, des_c = orb_features(gray[cand])
            if des_c is None:
                continue
            ia, ib = match_descriptors(des_c, des_cur, ratio=ratio)
            if len(ia) < min_inliers:
                continue

            # 3D points come from the candidate keyframe, 2D from the current
            # one; match by nearest keypoint so we can look the point id up.
            pts3d, pts2d = [], []
            # build a fast uv->point lookup for the candidate
            # (kp_c[i] -> nearest observed point in that keyframe)
            cand_pts = pts_in_kf.get(cand, {})
            if not cand_pts:
                continue
            cpu = np.array([np.array(uv_at_kf[(cand, p)]) for p in cand_pts], np.float64)
            cpx = np.array([map_xyz[p] for p in cand_pts], np.float64)
            for a, b in zip(ia, ib):
                p_uv = np.array(kp_c[a].pt)
                d2 = np.sum((cpu - p_uv) ** 2, axis=1)
                t = int(np.argmin(d2))
                if d2[t] > 9.0:            # must be within 3 px of an obs
                    continue
                pts3d.append(cpx[t])
                pts2d.append(np.array(kp_cur[b].pt))
            if len(pts3d) < min_inliers:
                continue

            if K_ is None:
                K_ = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]],
                              np.float64)
            r = pose_from_pnp(pts3d, pts2d, K_, ransac_px=ransac_px)
            if not r["ok"]:
                continue
            if len(r["inliers"]) < min_inliers:
                continue
            if r["inlier_ratio"] < min_inlier_ratio:
                continue

            # --- 3. geometric verification against the current estimate
            T_pred = relative_pose(poses[cand], poses[cur])
            dt, dr = pose_difference(T_pred, r["T"])
            if dt > agree_trans or dr > agree_rot_deg:
                continue

            # Z_ij for the pose graph: from node `cur` to node `cand`
            # PnP was given WORLD-frame 3D points, so it returned T_cw(cur),
            # the world-to-camera pose of the current frame.  Invert it to get
            # the camera-to-world pose the loop is asserting, then write it as a
            # relative measurement against the candidate keyframe, which is what
            # the pose graph expects (Z_ij: T_wc_j = T_wc_i @ Z_ij).
            T_wc_meas = np.linalg.inv(r["T"])
            T_ij = relative_pose(poses[cand], T_wc_meas)
            results.append(dict(i=cur, j=cand, T_ij=T_ij,
                                T_wc_meas=T_wc_meas,
                                inliers=int(len(r["inliers"])),
                                inlier_ratio=r["inlier_ratio"],
                                reproj=r["reproj"],
                                cand_dist=cdist,
                                agree_trans=dt, agree_rot=dr))
            if verbose:
                print("  loop %4d <-> %4d  inliers=%3d ratio=%.2f reproj=%.2f px"
                      "  cand_dist=%.2f m" % (cur, cand, len(r["inliers"]),
                                              r["inlier_ratio"], r["reproj"], cdist),
                      flush=True)
    return results
