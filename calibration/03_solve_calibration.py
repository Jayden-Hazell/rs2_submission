#!/usr/bin/env python3
# 41069 Robotics Studio 2
# Authors: Jayden Hazell (24953364), Lachlan Partridge (24964557), Tomas Klimes (24773804)
# Runs all five OpenCV hand-eye solvers and writes the best-consensus result to calib_result.yaml.
"""
03_solve_calibration.py

Loads calib_samples.json (produced by 02_collect_samples.py), runs all five
cv2.calibrateHandEye() methods, post-processes the result to obtain
T_tool0_to_d435i_link, and writes calib_result.yaml.

Usage:
    cd /home/jayden/ros2_ws/src/rs2
    python3 scripts/calibration/03_solve_calibration.py

Reads:  scripts/calibration/calib_samples.json
Writes: scripts/calibration/calib_result.yaml

Post-processing math (matches master.launch.py lines 136-151):
    T_tool0_optical     = calibrateHandEye output  (cam2gripper)
    T_d435i_link_optical = captured at startup in 02_collect_samples.py
    T_tool0_d435i_link  = T_tool0_optical @ inv(T_d435i_link_optical)
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

SAMPLES_FILE = Path(__file__).parent / 'calib_samples.json'
RESULT_FILE  = Path(__file__).parent / 'calib_result.yaml'

MIN_SAMPLES = 5

METHODS = [
    ('TSAI',       cv2.CALIB_HAND_EYE_TSAI),
    ('PARK',       cv2.CALIB_HAND_EYE_PARK),
    ('HORAUD',     cv2.CALIB_HAND_EYE_HORAUD),
    ('ANDREFF',    cv2.CALIB_HAND_EYE_ANDREFF),
    ('DANIILIDIS', cv2.CALIB_HAND_EYE_DANIILIDIS),
]

# Tolerances for the outlier filter
TRANS_TOL_M   = 0.02   # 20 mm
ROT_TOL_DEG   = 5.0    # degrees


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_samples(path: Path):
    with open(path) as f:
        data = json.load(f)

    samples = data['samples']
    n = len(samples)
    if n < MIN_SAMPLES:
        print(f'ERROR: only {n} samples — need at least {MIN_SAMPLES}.')
        sys.exit(1)

    tf_link_optical = data['tf_d435i_link_to_optical']

    R_g2b_list, t_g2b_list = [], []
    R_t2c_list, t_t2c_list = [], []

    for s in samples:
        R_g2b_list.append(np.array(s['R_gripper2base'], dtype=np.float64))           # (3,3)
        t_g2b_list.append(np.array(s['t_gripper2base'], dtype=np.float64).reshape(3, 1))  # (3,1)
        R_t2c_list.append(np.array(s['R_target2cam'],   dtype=np.float64))           # (3,3)
        t_t2c_list.append(np.array(s['t_target2cam'],   dtype=np.float64).reshape(3, 1))  # (3,1)

    return R_g2b_list, t_g2b_list, R_t2c_list, t_t2c_list, tf_link_optical, n


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------

def build_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = t.flatten()
    return T


def T_from_dict(d: dict) -> np.ndarray:
    """Build 4x4 from {'translation': {x,y,z}, 'rotation_xyzw': {x,y,z,w}}."""
    tr = d['translation']
    ro = d['rotation_xyzw']
    R  = Rotation.from_quat([ro['x'], ro['y'], ro['z'], ro['w']]).as_matrix()
    t  = np.array([tr['x'], tr['y'], tr['z']], dtype=np.float64)
    return build_T(R, t.reshape(3, 1))


def T_to_xyzquat(T: np.ndarray):
    """Return (x, y, z, qx, qy, qz, qw) from 4x4 homogeneous matrix."""
    x, y, z = T[:3, 3]
    q = Rotation.from_matrix(T[:3, :3]).as_quat()   # scipy returns (x, y, z, w)
    return float(x), float(y), float(z), float(q[0]), float(q[1]), float(q[2]), float(q[3])


# ---------------------------------------------------------------------------
# Calibration and post-processing
# ---------------------------------------------------------------------------

def run_all_methods(R_g2b, t_g2b, R_t2c, t_t2c) -> dict:
    results = {}
    for name, flag in METHODS:
        try:
            R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=flag)
            results[name] = build_T(R, t)
            print(f'  {name:12s}  OK')
        except Exception as e:
            print(f'  {name:12s}  FAILED: {e}')
    return results


def postprocess(raw_results: dict, T_link_optical: np.ndarray) -> dict:
    """
    T_tool0_optical    = calibrateHandEye output (R_cam2gripper, t_cam2gripper)
    T_tool0_d435i_link = T_tool0_optical @ inv(T_d435i_link_optical)
    """
    T_inv_link_optical = np.linalg.inv(T_link_optical)
    processed = {}
    for name, T_cam2gripper in raw_results.items():
        processed[name] = T_cam2gripper @ T_inv_link_optical
    return processed


# ---------------------------------------------------------------------------
# Outlier filtering and consensus
# ---------------------------------------------------------------------------

def rotation_geodesic_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    dR    = R1.T @ R2
    angle = Rotation.from_matrix(dR).magnitude()
    return float(np.degrees(angle))


def outlier_filter(T_dict: dict):
    """
    A method is 'good' if at least half of the other methods agree with it
    within TRANS_TOL_M and ROT_TOL_DEG.
    Returns (good_dict, outlier_names).
    """
    names = list(T_dict.keys())
    Ts    = list(T_dict.values())
    n     = len(Ts)
    agree = [0] * n

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dt = float(np.linalg.norm(Ts[i][:3, 3] - Ts[j][:3, 3]))
            dr = rotation_geodesic_deg(Ts[i][:3, :3], Ts[j][:3, :3])
            if dt < TRANS_TOL_M and dr < ROT_TOL_DEG:
                agree[i] += 1

    threshold = (n - 1) // 2   # at least half of the others must agree
    good      = {names[i]: Ts[i] for i in range(n) if agree[i] >= threshold}
    outliers  = [names[i]         for i in range(n) if agree[i] <  threshold]
    return good, outliers


def consensus_T(T_dict: dict) -> np.ndarray:
    """Average translation; geodesic mean rotation via scipy Rotation.mean()."""
    ts    = np.stack([T[:3, 3] for T in T_dict.values()])
    Rs    = Rotation.from_matrix(np.stack([T[:3, :3] for T in T_dict.values()]))
    t_avg = ts.mean(axis=0)
    R_avg = Rotation.mean(Rs).as_matrix()
    return build_T(R_avg, t_avg.reshape(3, 1))


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def sanity_check(T_tool0_optical: np.ndarray) -> None:
    """
    The user confirmed camera Z-axis is perpendicular to tool Z-axis
    (camera points radially outward from the flange).
    Express camera Z in tool0 frame and check the angle with tool Z [0,0,1].
    """
    cam_z_in_tool0 = T_tool0_optical[:3, :3] @ np.array([0.0, 0.0, 1.0])
    cos_angle      = np.clip(abs(cam_z_in_tool0[2]), 0.0, 1.0)
    angle_deg      = float(np.degrees(np.arccos(cos_angle)))

    if angle_deg < 45.0:
        print(f'  SANITY WARNING: cam Z is {angle_deg:.1f}° from tool Z '
              f'(expected ~90° for radial mounting).')
        print('  If the board geometry is wrong, this warning can be ignored.')
    else:
        print(f'  Camera Z vs tool Z: {angle_deg:.1f}° apart — consistent with '
              f'radial mounting (expected ~90°).')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not SAMPLES_FILE.exists():
        print(f'ERROR: {SAMPLES_FILE} not found.')
        print('Run 02_collect_samples.py first.')
        sys.exit(1)

    print(f'Loading samples from {SAMPLES_FILE}')
    R_g2b, t_g2b, R_t2c, t_t2c, tf_link_optical_dict, n = load_samples(SAMPLES_FILE)
    print(f'  {n} valid samples loaded\n')

    T_link_optical = T_from_dict(tf_link_optical_dict)
    print(f'T_d435i_link_to_optical (static TF captured during collection):')
    print(f'  {T_link_optical}\n')

    # --- Run all five methods ---
    print('Running cv2.calibrateHandEye() — all 5 methods:')
    raw_results = run_all_methods(R_g2b, t_g2b, R_t2c, t_t2c)
    if not raw_results:
        print('\nERROR: all methods failed. Check that samples have sufficient pose diversity.')
        sys.exit(1)

    # --- Post-process to tool0 -> d435i_link ---
    print('\nPost-processing to T_tool0_d435i_link:')
    print('  (T_tool0_d435i_link = T_tool0_optical @ inv(T_d435i_link_optical))\n')
    processed = postprocess(raw_results, T_link_optical)

    print(f'  {"Method":<12}  {"x (m)":>10}  {"y (m)":>10}  {"z (m)":>10}'
          f'  {"qx":>9}  {"qy":>9}  {"qz":>9}  {"qw":>9}')
    print('  ' + '-' * 100)
    per_method = {}
    for name, T in processed.items():
        x, y, z, qx, qy, qz, qw = T_to_xyzquat(T)
        per_method[name] = (x, y, z, qx, qy, qz, qw)
        print(f'  {name:<12}  {x:+10.6f}  {y:+10.6f}  {z:+10.6f}'
              f'  {qx:+9.6f}  {qy:+9.6f}  {qz:+9.6f}  {qw:+9.6f}')

    # --- Outlier filter ---
    print('\nOutlier detection:')
    good, outliers = outlier_filter(processed)
    if outliers:
        print(f'  Outliers (excluded from consensus): {outliers}')
    else:
        print('  No outliers — all methods agree within tolerance.')
    print(f'  Consensus built from: {list(good.keys())}')

    if not good:
        print('\nERROR: all methods disagree — collect more samples with greater pose diversity.')
        sys.exit(1)

    # --- Consensus ---
    T_consensus = consensus_T(good)
    x, y, z, qx, qy, qz, qw = T_to_xyzquat(T_consensus)

    print('\n' + '=' * 65)
    print('  CONSENSUS RESULT  (tool0 -> d435i_link)')
    print('=' * 65)
    print(f'  x  = {x:+.9f} m')
    print(f'  y  = {y:+.9f} m')
    print(f'  z  = {z:+.9f} m')
    print(f'  qx = {qx:+.9f}')
    print(f'  qy = {qy:+.9f}')
    print(f'  qz = {qz:+.9f}')
    print(f'  qw = {qw:+.9f}')

    # --- Sanity check (use raw T_tool0_optical from any good method) ---
    first_good_name = list(good.keys())[0]
    T_tool0_optical_sample = raw_results[first_good_name]
    print('\nSanity check (camera orientation):')
    sanity_check(T_tool0_optical_sample)

    # --- Per-method spread (quality indicator) ---
    good_vals = np.array([per_method[n][:3] for n in good])
    trans_spread = float(np.max(np.linalg.norm(good_vals - good_vals.mean(axis=0), axis=1)))
    print(f'\n  Translation spread across good methods: {trans_spread*1000:.2f} mm '
          f'(< 5 mm is excellent, < 15 mm is acceptable)')

    # --- Save calib_result.yaml ---
    calib_result = {
        'translation':    {'x': x,   'y': y,   'z': z},
        'rotation_xyzw':  {'x': qx,  'y': qy,  'z': qz,  'w': qw},
        'frame_id':       'tool0',
        'child_frame_id': 'd435i_link',
        'n_samples':      n,
        'methods_used':   list(good.keys()),
    }
    with open(RESULT_FILE, 'w') as f:
        yaml.safe_dump(calib_result, f, default_flow_style=False, sort_keys=False)
    print(f'\nCalibration result written: {RESULT_FILE}')

    # --- Print launch file arguments ---
    args_str = (
        f'--x {x:.15f} --y {y:.15f} --z {z:.15f} '
        f'--qx {qx:.9f} --qy {qy:.9f} --qz {qz:.9f} --qw {qw:.9f} '
        f'--frame-id tool0 --child-frame-id d435i_link'
    )
    print()
    print('=' * 65)
    print('  MASTER.LAUNCH.PY — replace handeye_tf Node arguments:')
    print('=' * 65)
    print(f'\n  {args_str}\n')

    # --- Validation command ---
    print('=' * 65)
    print('  VALIDATION — test against existing scan session:')
    print('=' * 65)
    print()
    print('  python3 helpers/reconstruct.py \\')
    print('      --session scans/20260512_135743 \\')
    print('      --recompose \\')
    print('      --override-tf scripts/calibration/calib_result.yaml')
    print()
    print('  Compare the reconstructed point cloud visually:')
    print('  - Flat surfaces (workbench, table) should appear flat and level.')
    print('  - Object heights should be consistent across all 22 captures.')
    print('  - Run WITHOUT --override-tf first to see the baseline (old values).')
    print()


if __name__ == '__main__':
    main()
