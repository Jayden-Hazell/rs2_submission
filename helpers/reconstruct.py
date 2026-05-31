#!/usr/bin/env python3
# 41069 Robotics Studio 2
# Authors: Jayden Hazell (24953364), Lachlan Partridge (24964557), Tomas Klimes (24773804)
# Point cloud reconstruction pipeline: merges per-capture RGB-D frames into a filtered, coloured 3D cloud.
"""
reconstruct.py

Merges all captures in a session directory into a single combined point cloud.

Scan format detection (automatic per capture):
  New format  — capture.json present → Mode A or Mode B
  Legacy format — pose.yaml present, no capture.json → --legacy-pose required

Reconstruction modes:
  Mode A (default)    Use pose_composed from capture.json directly.
  Mode B (--recompose) Recompose pose from pose_chain + session.json calibration.
                       Enables --override-tf to substitute the handeye transform.
  Legacy (--legacy-pose) Read pose.yaml directly; applies the Original_Test body-to-
                          optical double-application correction from the old pipeline.

Usage:
    python3 reconstruct.py --session /path/to/session [options]

Options:
    --session           Path to session directory (required)
    --depth-trunc       Maximum depth in metres (default: 2.0)
    --depth-scale       Scale factor applied to depth values (default: 1.0)
    --voxel-size        Voxel downsampling size in metres, 0 to skip (default: 0.005)
    --bbox-min          Min x y z crop in base_link space
    --bbox-max          Max x y z crop in base_link space
    --output            Output .ply filename (default: reconstruction.ply in session dir)
    --no-visualise      Skip Open3D visualisation
    --skip              Comma-separated capture names to exclude
    --only              Comma-separated capture names to include exclusively
    --icp               Refine each capture against the accumulated cloud via ICP
    --icp-threshold     Max correspondence distance for ICP in metres (default: 0.01)
    --recompose         Mode B: recompose pose from pose_chain + session.json calibration
    --override-tf       Path to YAML file with replacement tool0->d435i_link transform
                        (only valid with --recompose)
    --legacy-pose       Legacy mode: read pose.yaml directly (for pre-capture.json scans)
"""

import os
import sys
import glob
import json
import yaml
import shutil
import argparse
import datetime
import subprocess
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d


# ==============================================================================
# TUNABLE CONSTANTS
# Edit these values to adjust reconstruction quality.
# See RECONSTRUCTION_TUNING_GUIDE.txt for a full explanation of each.
# ==============================================================================

# --- Per-capture Statistical Outlier Removal (SOR) ---
# Runs on each individual capture BEFORE merging. Removes isolated spikes.
SOR_NB_NEIGHBORS = 20    # How many neighbours to include in the mean distance check.
                          # Lower = more sensitive (removes more). Higher = less sensitive.
SOR_STD_RATIO    = 2.0   # Points further than (mean + STD_RATIO × std_dev) are removed.
                          # Lower = more aggressive. Higher = keeps more noisy points.

# --- Global Radius Outlier Removal (ROR) ---
# Runs on the MERGED cloud after voxel downsampling. Removes floating blobs.
ROR_NB_POINTS = 10       # A point is kept only if it has at least this many neighbours
                          # within ROR_RADIUS_M. Raise to kill larger floating clusters.
ROR_RADIUS_M  = 0.010    # Sphere radius in metres for the neighbour count check.
                          # Lower = stricter (removes more). Higher = looser.

# --- DBSCAN Cluster Filtering ---
# Runs after ROR on the merged cloud. Removes small isolated clusters (stray blobs).
DBSCAN_EPS_M           = 0.020  # Max distance between points in the same cluster (metres).
                                  # Raise if the object cloud is sparse/has gaps.
DBSCAN_MIN_POINTS      = 10     # Minimum neighbours within eps to form a core point.
DBSCAN_MIN_CLUSTER_PTS = 50     # Clusters with fewer than this many points are removed.
                                  # Set above the expected stray blob size.

# --- Normal Estimation ---
# Normals are computed after merging and are required by Poisson reconstruction.
NORMAL_RADIUS_M = 0.02   # KDTree search radius in metres. Should be ~2–3× voxel size.
                          # Too small = noisy normals. Too large = over-smoothed normals.
NORMAL_MAX_NN   = 30     # Maximum neighbours included in the normal fit.
                          # Raise for smoother normals on sparse clouds.
NORMAL_ORIENT_K = 20     # Neighbours used to propagate consistent normal orientation.
                          # Raise if normals are flipping (inside/outside confusion).

# ==============================================================================
# Argument Parsing
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge all RGBD captures in a session into a single point cloud.'
    )
    parser.add_argument('--session',     type=str,   required=True)
    parser.add_argument('--depth-trunc', type=float, default=2.0)
    parser.add_argument('--depth-scale', type=float, default=1.0)
    parser.add_argument('--voxel-size',  type=float, default=0.005)
    parser.add_argument('--output',      type=str,   default=None)
    parser.add_argument(
        '--models-dir', type=str, default=None, metavar='PATH',
        help='Override the models output directory (default: <repo>/models/<session_name>/).'
    )
    parser.add_argument('--no-visualise', action='store_true')
    parser.add_argument(
        '--bbox-min', type=float, nargs=3, default=None,
        metavar=('X', 'Y', 'Z'),
        help='Crop: min x y z in base_link space (metres).'
    )
    parser.add_argument(
        '--bbox-max', type=float, nargs=3, default=None,
        metavar=('X', 'Y', 'Z'),
        help='Crop: max x y z in base_link space (metres).'
    )
    parser.add_argument(
        '--skip', type=str, default=None, metavar='NAMES',
        help='Comma-separated capture names to exclude.'
    )
    parser.add_argument(
        '--only', type=str, default=None, metavar='NAMES',
        help='Comma-separated capture names to include exclusively.'
    )
    parser.add_argument(
        '--icp', action='store_true',
        help='Refine each capture against the accumulated cloud using point-to-plane ICP.'
    )
    parser.add_argument(
        '--icp-threshold', type=float, default=0.01, metavar='METRES',
        help='Max correspondence distance for ICP (default: 0.01 m).'
    )
    parser.add_argument(
        '--recompose', action='store_true',
        help='Mode B: recompose pose from pose_chain + session.json calibration chain.'
    )
    parser.add_argument(
        '--override-tf', type=str, default=None, metavar='YAML_PATH',
        help='Path to YAML file with replacement tool0->intermediate_frame transform. '
             'Only valid with --recompose.'
    )
    parser.add_argument(
        '--legacy-pose', action='store_true',
        help='Legacy mode: read pose.yaml directly (for pre-capture.json sessions). '
             'Applies the body-to-optical double-application correction used in the '
             'Original_Test session.'
    )
    parser.add_argument(
        '--workspace-config', type=str, default=None, metavar='YAML_PATH',
        help='Path to workspace.yaml. Auto-detected as <repo>/config/workspace.yaml '
             'when omitted.'
    )
    parser.add_argument(
        '--debug-bbox', action='store_true',
        help='Preview mode: reconstruct full uncropped cloud and overlay workspace bbox '
             'as a red wireframe in the viewer. Does not save output. '
             'Use to visually tune bounding_box values in workspace.yaml.'
    )

    mesh = parser.add_argument_group('Meshing (mesh_from_cloud.py)')
    mesh.add_argument(
        '--mesh', action='store_true',
        help='Run mesh_from_cloud.py automatically after the point cloud is saved.'
    )
    # Defaults here should match the TUNABLE CONSTANTS at the top of mesh_from_cloud.py
    mesh.add_argument('--mesh-smooth-passes',       type=int,   default=80,  metavar='N',
                      help='Taubin smooth passes passed to mesh_from_cloud.py (default: 80)')
    mesh.add_argument('--mesh-depth',               type=int,   default=7,   metavar='N',
                      help='Poisson depth passed to mesh_from_cloud.py (default: 7)')
    mesh.add_argument('--mesh-density-threshold',   type=float, default=0.20, metavar='F',
                      help='Density threshold passed to mesh_from_cloud.py (default: 0.20)')

    return parser.parse_args()


# ==============================================================================
# Scan format detection and loading
# ==============================================================================

def detect_scan_format(cap_dir: str) -> str:
    """Returns 'new' if capture.json exists, 'legacy' if pose.yaml exists."""
    if os.path.exists(os.path.join(cap_dir, 'capture.json')):
        return 'new'
    if os.path.exists(os.path.join(cap_dir, 'pose.yaml')):
        return 'legacy'
    raise FileNotFoundError(
        f'Cannot determine scan format for {cap_dir}: '
        f'neither capture.json nor pose.yaml found.'
    )


def load_capture_new(cap_dir: str) -> dict:
    paths = {
        'color':       os.path.join(cap_dir, 'color.png'),
        'depth':       os.path.join(cap_dir, 'depth.npy'),
        'cam_info':    os.path.join(cap_dir, 'camera_info.yaml'),
        'capture_json': os.path.join(cap_dir, 'capture.json'),
    }
    missing = [k for k, v in paths.items() if not os.path.exists(v)]
    if missing:
        raise FileNotFoundError(f'Missing files in {cap_dir}: {missing}')

    color = cv2.imread(paths['color'], cv2.IMREAD_COLOR)
    if color is None:
        raise RuntimeError(f'Failed to read: {paths["color"]}')
    depth = np.load(paths['depth'])
    with open(paths['cam_info']) as f:
        cam_info = yaml.safe_load(f)
    with open(paths['capture_json']) as f:
        capture_json = json.load(f)

    return {
        'format':       'new',
        'color':        color,
        'depth':        depth,
        'cam_info':     cam_info,
        'capture_json': capture_json,
    }


def load_capture_legacy(cap_dir: str) -> dict:
    paths = {
        'color':    os.path.join(cap_dir, 'color.png'),
        'depth':    os.path.join(cap_dir, 'depth.npy'),
        'cam_info': os.path.join(cap_dir, 'camera_info.yaml'),
        'pose':     os.path.join(cap_dir, 'pose.yaml'),
    }
    missing = [k for k, v in paths.items() if not os.path.exists(v)]
    if missing:
        raise FileNotFoundError(f'Missing files in {cap_dir}: {missing}')

    color = cv2.imread(paths['color'], cv2.IMREAD_COLOR)
    if color is None:
        raise RuntimeError(f'Failed to read: {paths["color"]}')
    depth = np.load(paths['depth'])
    with open(paths['cam_info']) as f:
        cam_info = yaml.safe_load(f)
    with open(paths['pose']) as f:
        pose = yaml.safe_load(f)

    return {
        'format':   'legacy',
        'color':    color,
        'depth':    depth,
        'cam_info': cam_info,
        'pose':     pose,
    }


def load_capture(cap_dir: str) -> dict:
    fmt = detect_scan_format(cap_dir)
    if fmt == 'new':
        return load_capture_new(cap_dir)
    return load_capture_legacy(cap_dir)


def load_session_json(session_dir: str) -> dict:
    path = os.path.join(session_dir, 'session.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'session.json not found in {session_dir}. '
            f'Mode B (--recompose) requires a session recorded with the new scan_node.'
        )
    with open(path) as f:
        return json.load(f)


def load_override_tf(yaml_path: str) -> dict:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    required = {'translation', 'rotation_xyzw'}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f'override-tf YAML missing keys: {missing}')
    return data


# ==============================================================================
# Math helpers
# ==============================================================================

def transform_dict_to_matrix(td: dict) -> np.ndarray:
    """Convert a translation+rotation_xyzw dict to a 4x4 homogeneous matrix."""
    t = td['translation']
    q = td['rotation_xyzw']
    x, y, z, w = float(q['x']), float(q['y']), float(q['z']), float(q['w'])
    R = np.array([
        [1-2*(y**2+z**2),   2*(x*y-z*w),     2*(x*z+y*w)],
        [2*(x*y+z*w),       1-2*(x**2+z**2), 2*(y*z-x*w)],
        [2*(x*z-y*w),       2*(y*z+x*w),     1-2*(x**2+y**2)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = [float(t['x']), float(t['y']), float(t['z'])]
    return T


def pose_to_matrix(pose_yaml: dict) -> np.ndarray:
    """Load a pose.yaml (legacy format) into a 4x4 matrix."""
    return transform_dict_to_matrix(pose_yaml)


def legacy_correction_matrix() -> np.ndarray:
    """
    Returns inv(T_{d435i_link → d435i_color_optical_frame}).

    In the Original_Test session, the TF chain accidentally applied the
    body-to-optical rotation (D) twice, so every saved pose satisfies:
        T_saved = T_correct × D
    Post-multiplying by inv(D) recovers T_correct.

    D rotation: roll=-π/2, pitch=0, yaw=-π/2  →  qx=-0.5, qy=0.5, qz=-0.5, qw=0.5
    D translation set to zero (physical offset <2 mm; unverified without hardware).
    To refine: run  ros2 run tf2_ros tf2_echo d435i_link d435i_color_optical_frame
    and update D[:3, 3] below.
    """
    qx, qy, qz, qw = -0.5, 0.5, -0.5, 0.5
    R = np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx**2+qy**2)],
    ], dtype=np.float64)
    D = np.eye(4, dtype=np.float64)
    D[:3, :3] = R
    D[:3, 3]  = [0.0, 0.0, 0.0]  # refine once tf2_echo values are available
    return np.linalg.inv(D)


# ==============================================================================
# Intrinsics
# ==============================================================================

def make_intrinsics(cam_info: dict) -> o3d.camera.PinholeCameraIntrinsic:
    k = cam_info['k']
    return o3d.camera.PinholeCameraIntrinsic(
        int(cam_info['width']), int(cam_info['height']),
        float(k[0]), float(k[4]), float(k[2]), float(k[5])
    )


# ==============================================================================
# Depth preprocessing
# ==============================================================================

def depth_to_metres(depth: np.ndarray, depth_trunc: float) -> np.ndarray:
    """
    Convert depth to float32 metres, zeroing saturated pixels and anything
    beyond depth_trunc before back-projection.
    """
    if depth.dtype == np.uint16:
        print('    depth dtype: uint16 — converting mm -> metres')
        saturated = depth == 65535
        n_sat = int(saturated.sum())
        if n_sat > 0:
            print(f'    zeroing {n_sat} saturated pixels (RealSense no-data marker)')
        depth_m = depth.astype(np.float32) / 1000.0
        depth_m[saturated] = 0.0
    elif depth.dtype == np.float32:
        print('    depth dtype: float32 — assuming already metres')
        depth_m = depth.copy()
    else:
        print(f'    depth dtype: {depth.dtype} — casting to float32')
        depth_m = depth.astype(np.float32)

    too_far = depth_m > depth_trunc
    n_far = int(too_far.sum())
    if n_far > 0:
        print(f'    zeroing {n_far} pixels beyond {depth_trunc}m in camera space')
    depth_m[too_far] = 0.0
    return depth_m


# ==============================================================================
# Bounding-box crop
# ==============================================================================

def crop_pcd(
    pcd: o3d.geometry.PointCloud,
    bbox_min: list,
    bbox_max: list,
) -> o3d.geometry.PointCloud:
    pts  = np.asarray(pcd.points)
    mask = (
        (pts[:, 0] >= bbox_min[0]) & (pts[:, 0] <= bbox_max[0]) &
        (pts[:, 1] >= bbox_min[1]) & (pts[:, 1] <= bbox_max[1]) &
        (pts[:, 2] >= bbox_min[2]) & (pts[:, 2] <= bbox_max[2])
    )
    return pcd.select_by_index(np.where(mask)[0])


def make_bbox_edge_cloud(
    bbox_min: list,
    bbox_max: list,
    n_per_edge: int = 80,
) -> o3d.geometry.PointCloud:
    """
    Sample n_per_edge points along each of the 12 bbox edges.
    Returns a PointCloud coloured solid red — works in any viewer including MeshLab.
    """
    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x0, y1, z0], [x1, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x0, y1, z1], [x1, y1, z1],
    ], dtype=float)
    edge_pairs = [
        (0, 1), (0, 2), (1, 3), (2, 3),
        (4, 5), (4, 6), (5, 7), (6, 7),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    segments = [np.linspace(corners[i], corners[j], n_per_edge) for i, j in edge_pairs]
    pts = np.vstack(segments)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 0.0], (len(pts), 1)))
    return pcd


def make_bbox_lineset(
    bbox_min: list,
    bbox_max: list,
    color: list | None = None,
) -> o3d.geometry.LineSet:
    """Return a 12-edge axis-aligned wireframe LineSet for the given bounding box."""
    if color is None:
        color = [1.0, 0.0, 0.0]
    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x0, y1, z0], [x1, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x0, y1, z1], [x1, y1, z1],
    ], dtype=float)
    edges = [
        [0, 1], [0, 2], [1, 3], [2, 3],
        [4, 5], [4, 6], [5, 7], [6, 7],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    ls = o3d.geometry.LineSet()
    ls.points  = o3d.utility.Vector3dVector(corners)
    ls.lines   = o3d.utility.Vector2iVector(edges)
    ls.colors  = o3d.utility.Vector3dVector([color] * len(edges))
    return ls


# ==============================================================================
# Workspace config
# ==============================================================================

def load_workspace_config(session_dir: str, override_path: str | None = None) -> dict | None:
    """
    Load config/workspace.yaml.

    Resolution order:
      1. override_path if provided
      2. <repo_root>/config/workspace.yaml  (session_dir = .../rs2/scans/<session>)

    Returns the parsed dict, or None if the file is absent (not an error).
    """
    if override_path is not None:
        cfg_path = Path(override_path)
    else:
        cfg_path = Path(session_dir).parent.parent / 'config' / 'workspace.yaml'

    if not cfg_path.exists():
        return None

    with open(cfg_path) as f:
        return yaml.safe_load(f)


def derive_effective_bbox(
    workspace: dict | None,
    cli_min: list | None,
    cli_max: list | None,
) -> tuple[list | None, list | None]:
    """
    Determine the effective bbox_min / bbox_max to use for cropping.

    Priority:
      1. Explicit --bbox-min / --bbox-max CLI args (full override)
      2. workspace.yaml bounding_box + computed z_min from bench calibration
      3. No crop (both return None)

    z_min = bench.z_m + bench.margin_m  (only when bench.z_m is not null)
    z_max comes from bounding_box.z_max.
    x/y bounds come from bounding_box section.
    """
    if cli_min is not None or cli_max is not None:
        return cli_min, cli_max

    if workspace is None:
        return None, None

    bb = workspace.get('bounding_box')
    bench = workspace.get('bench')
    if bb is None:
        return None, None

    try:
        x_min = float(bb['x_min'])
        x_max = float(bb['x_max'])
        y_min = float(bb['y_min'])
        y_max = float(bb['y_max'])
        z_max = float(bb['z_max'])
    except (KeyError, TypeError, ValueError):
        return None, None

    z_m = bench.get('z_m') if bench else None
    margin_m = float(bench.get('margin_m', 0.010)) if bench else 0.010

    if z_m is None:
        z_min = None
    else:
        z_min = float(z_m) + margin_m

    if z_min is None:
        return None, None

    return [x_min, y_min, z_min], [x_max, y_max, z_max]


# ==============================================================================
# Pose resolution
# ==============================================================================

def resolve_pose(
    data: dict,
    mode: str,
    session_calib: dict | None,
    override_tf: dict | None,
    legacy_corr: np.ndarray | None,
) -> np.ndarray:
    """
    Return a 4x4 homogeneous matrix T such that p_world = T × p_camera.

    mode='A'      — use pose_composed from capture.json directly
    mode='B'      — recompose from pose_chain + session calibration
    mode='legacy' — use pose.yaml, optionally apply legacy correction
    """
    if mode == 'legacy':
        T = pose_to_matrix(data['pose'])
        if legacy_corr is not None:
            T = T @ legacy_corr
        return T

    cj = data['capture_json']

    if mode == 'A':
        return transform_dict_to_matrix(cj['pose_composed'])

    # Mode B — recompose
    if session_calib is None:
        raise ValueError('Mode B requires session calibration (session.json).')

    calib = session_calib.get('calibration')
    if calib is None:
        reason = session_calib.get('calibration_unavailable_reason', 'unknown')
        raise ValueError(
            f'session.json has no calibration data (reason: {reason}). '
            f'Mode B is unavailable for this session.'
        )

    chain = cj.get('pose_chain', {})
    robot_ee  = cj.get('robot_ee_frame', 'tool0')
    fixed     = cj.get('fixed_frame',    'base_link')
    key_base_to_ee = f'{fixed}_to_{robot_ee}'

    if key_base_to_ee not in chain:
        raise ValueError(
            f'pose_chain missing {key_base_to_ee!r} in {cj.get("capture_name")}. '
            f'Was the pose_chain lookup successful at capture time?'
        )

    T_base_to_ee = transform_dict_to_matrix(chain[key_base_to_ee])

    # Calibration chain keys mirror what scan_node records
    ee_frame    = session_calib.get('robot_ee_frame',    cj.get('robot_ee_frame',    'tool0'))
    mid_frame   = session_calib.get('intermediate_frame', 'd435i_link')
    cam_frame_s = session_calib.get('camera_frame',       cj.get('camera_frame',    'd435i_color_optical_frame'))

    key_ee_to_mid  = f'{ee_frame}_to_{mid_frame}'
    key_mid_to_cam = f'{mid_frame}_to_{cam_frame_s}'

    if key_ee_to_mid not in calib:
        raise ValueError(
            f'Calibration key {key_ee_to_mid!r} not found in session.json. '
            f'Available: {list(calib.keys())}'
        )
    if key_mid_to_cam not in calib:
        raise ValueError(
            f'Calibration key {key_mid_to_cam!r} not found in session.json. '
            f'Available: {list(calib.keys())}'
        )

    ee_to_mid_dict = override_tf if override_tf is not None else calib[key_ee_to_mid]
    T_ee_to_mid    = transform_dict_to_matrix(ee_to_mid_dict)
    T_mid_to_cam   = transform_dict_to_matrix(calib[key_mid_to_cam])

    return T_base_to_ee @ T_ee_to_mid @ T_mid_to_cam


# ==============================================================================
# Per-capture point cloud
# ==============================================================================

def capture_to_pcd(
    data: dict,
    depth_scale: float,
    depth_trunc: float,
    bbox_min: list | None,
    bbox_max: list | None,
    mode: str,
    session_calib: dict | None,
    override_tf: dict | None,
    legacy_corr: np.ndarray | None,
) -> o3d.geometry.PointCloud | None:
    """Convert a single capture dict into an Open3D PointCloud in base_link frame."""

    depth_m = depth_to_metres(data['depth'], depth_trunc)

    valid = int((depth_m > 0).sum())
    total = depth_m.size
    print(f'    valid pixels after clamping: {valid} / {total} ({100.0*valid/total:.1f}%)')
    if valid > 0:
        print(f'    depth range: {float(depth_m[depth_m>0].min()):.3f}m'
              f' – {float(depth_m[depth_m>0].max()):.3f}m')

    intrinsics = make_intrinsics(data['cam_info'])
    print(f'    intrinsics: '
          f'fx={intrinsics.intrinsic_matrix[0,0]:.2f}  '
          f'fy={intrinsics.intrinsic_matrix[1,1]:.2f}  '
          f'cx={intrinsics.intrinsic_matrix[0,2]:.2f}  '
          f'cy={intrinsics.intrinsic_matrix[1,2]:.2f}')

    color_rgb = cv2.cvtColor(data['color'], cv2.COLOR_BGR2RGB)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_rgb),
        o3d.geometry.Image(depth_m),
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )

    pcd_cam = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
    n_cam = len(pcd_cam.points)
    print(f'    points in camera frame: {n_cam}')
    if n_cam == 0:
        print('    WARNING: empty point cloud — skipping.')
        return None

    T = resolve_pose(data, mode, session_calib, override_tf, legacy_corr)

    print(f'    pose: x={T[0,3]:.4f}  y={T[1,3]:.4f}  z={T[2,3]:.4f}')

    pcd_base = pcd_cam.transform(T)
    print(f'    points in base_link frame: {len(pcd_base.points)}')

    if bbox_min is not None and bbox_max is not None:
        before = len(pcd_base.points)
        pcd_base = crop_pcd(pcd_base, bbox_min, bbox_max)
        after  = len(pcd_base.points)
        print(f'    points after bbox crop: {after}  (removed {before - after})')
        if after == 0:
            print('    WARNING: bbox crop removed all points — check --bbox-min/max values.')
            return None

    # SOR — per-capture, after optional bbox crop
    n_before_sor = len(pcd_base.points)
    pcd_base, _ = pcd_base.remove_statistical_outlier(nb_neighbors=SOR_NB_NEIGHBORS, std_ratio=SOR_STD_RATIO)
    n_after_sor = len(pcd_base.points)
    print(f'    points after SOR : {n_after_sor}  (removed {n_before_sor - n_after_sor})')
    if n_after_sor == 0:
        print('    WARNING: SOR removed all points.')
        return None

    pts = np.asarray(pcd_base.points)
    if len(pts) > 0:
        print(f'    final bbox: '
              f'x=[{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]  '
              f'y=[{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]  '
              f'z=[{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]')

    return pcd_base


# ==============================================================================
# ICP refinement
# ==============================================================================

def icp_refine(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    threshold: float,
) -> tuple:
    """
    Refine source against target using point-to-plane ICP.
    Both clouds must be in the same coordinate frame (base_link).
    Returns (refined_cloud, rmse, n_correspondences).
    """
    for pcd in (source, target):
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
    reg = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=threshold,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    n_corr = len(reg.correspondence_set)
    if n_corr == 0:
        # No correspondences found — transformation is undefined; return source unchanged.
        return source, 0.0, 0
    refined = source.transform(reg.transformation)
    return refined, float(reg.inlier_rmse), n_corr


# ==============================================================================
# Provenance output
# ==============================================================================

def write_provenance(
    session_dir: str,
    models_dir: str,
    output_path: str,
    args,
    mode: str,
    captures_found: int,
    captures_processed: int,
    skipped: list,
    final_point_count: int,
) -> None:
    prov_path = os.path.join(models_dir, 'reconstruction.json')
    data = {
        'timestamp':           datetime.datetime.now().isoformat(timespec='seconds'),
        'session_dir':         session_dir,
        'models_dir':          models_dir,
        'reconstruction_mode': mode,
        'output_path':         output_path,
        'args': {
            'depth_trunc':   args.depth_trunc,
            'depth_scale':   args.depth_scale,
            'voxel_size':    args.voxel_size,
            'bbox_min':      args.bbox_min,
            'bbox_max':      args.bbox_max,
            'icp':           args.icp,
            'icp_threshold': args.icp_threshold,
            'recompose':     args.recompose,
            'override_tf':   args.override_tf,
            'legacy_pose':   args.legacy_pose,
            'skip':          args.skip,
            'only':          args.only,
        },
        'captures_found':     captures_found,
        'captures_processed': captures_processed,
        'captures_skipped':   skipped,
        'final_point_count':  final_point_count,
    }
    with open(prov_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'  provenance written: {prov_path}')


# ==============================================================================
# Helpers
# ==============================================================================

def section(title: str) -> None:
    print()
    print('=' * 60)
    print(f'  {title}')
    print('=' * 60)


def subsection(title: str) -> None:
    print(f'\n--- {title} ---')


def find_captures(session_dir: str) -> list:
    pattern = os.path.join(session_dir, 'capture_*')
    return sorted([p for p in glob.glob(pattern) if os.path.isdir(p)])


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()

    # --- Argument validation ---
    if args.override_tf and not args.recompose:
        print('ERROR: --override-tf requires --recompose.')
        sys.exit(1)
    if args.recompose and args.legacy_pose:
        print('ERROR: --recompose and --legacy-pose are mutually exclusive.')
        sys.exit(1)

    session_dir  = os.path.abspath(args.session)
    session_name = os.path.basename(session_dir)
    repo_root    = Path(session_dir).parent.parent

    if args.output:
        output_path = os.path.abspath(args.output)
        models_dir  = os.path.dirname(output_path)
    elif args.models_dir:
        models_dir  = os.path.abspath(args.models_dir)
        output_path = os.path.join(models_dir, 'reconstruction.ply')
    else:
        models_dir  = str(repo_root / 'reconstructed_scans' / session_name)
        output_path = os.path.join(models_dir, 'reconstruction.ply')

    workspace = load_workspace_config(session_dir, getattr(args, 'workspace_config', None))
    bbox_min, bbox_max = derive_effective_bbox(workspace, args.bbox_min, args.bbox_max)

    # In debug-bbox mode, save the workspace bbox for the wireframe but disable all cropping
    _wire_min: list | None = None
    _wire_max: list | None = None
    if args.debug_bbox:
        _wire_min, _wire_max = bbox_min, bbox_max
        bbox_min, bbox_max = None, None

    skip_set     = set(args.skip.split(',')) if args.skip else set()
    only_set     = set(args.only.split(',')) if args.only else set()

    # Determine reconstruction mode
    if args.recompose:
        mode = 'B'
    elif args.legacy_pose:
        mode = 'legacy'
    else:
        mode = 'A'

    section('CONFIGURATION')
    print(f'  session dir   : {session_dir}')
    print(f'  models dir    : {models_dir}')
    print(f'  output path   : {output_path}')
    print(f'  mode          : {mode}')
    print(f'  depth trunc   : {args.depth_trunc} m')
    print(f'  depth scale   : {args.depth_scale}')
    print(f'  voxel size    : {args.voxel_size} m {"(disabled)" if args.voxel_size == 0 else ""}')
    if workspace is not None:
        bench = workspace.get('bench', {})
        z_m = bench.get('z_m')
        margin = bench.get('margin_m', 0.010)
        z_min_display = f'{float(z_m) + float(margin):.4f} m' if z_m is not None else 'not calibrated'
        print(f'  workspace cfg : loaded  (bench z_m={z_m}, z_min={z_min_display})')
    else:
        print(f'  workspace cfg : not found — no workspace-based crop')
    print(f'  bbox min      : {bbox_min}')
    print(f'  bbox max      : {bbox_max}')
    print(f'  visualise     : {not args.no_visualise}')
    print(f'  only          : {sorted(only_set) or "all"}')
    print(f'  skip          : {sorted(skip_set) or "none"}')
    print(f'  icp           : {args.icp}  threshold={args.icp_threshold} m')
    if args.override_tf:
        print(f'  override-tf   : {args.override_tf}')

    if not os.path.isdir(session_dir):
        print(f'ERROR: directory does not exist: {session_dir}')
        sys.exit(1)

    # --- Load session.json for Mode B ---
    session_calib: dict | None = None
    if mode == 'B':
        section('LOADING SESSION CALIBRATION')
        session_data  = load_session_json(session_dir)
        session_calib = session_data
        calib = session_data.get('calibration')
        if calib is None:
            reason = session_data.get('calibration_unavailable_reason', 'unknown')
            print(f'ERROR: session.json has no calibration. Reason: {reason}')
            print('       Mode B unavailable. Use Mode A (default) or --legacy-pose.')
            sys.exit(1)
        print(f'  calibration keys: {list(calib.keys())}')

    # --- Load override-tf ---
    override_tf: dict | None = None
    if args.override_tf:
        section('LOADING OVERRIDE TF')
        override_tf = load_override_tf(args.override_tf)
        print(f'  translation: {override_tf["translation"]}')
        print(f'  rotation:    {override_tf["rotation_xyzw"]}')

    # --- Legacy correction matrix ---
    legacy_corr: np.ndarray | None = None
    if mode == 'legacy':
        legacy_corr = legacy_correction_matrix()
        print('  legacy correction matrix loaded (Original_Test body-to-optical fix).')

    section('FINDING CAPTURES')
    captures = find_captures(session_dir)
    if not captures:
        print(f'ERROR: no capture_XXXX dirs found in {session_dir}')
        sys.exit(1)
    print(f'  found {len(captures)} capture(s):')
    for c in captures:
        marker = '  [SKIP]' if os.path.basename(c) in skip_set else ''
        print(f'    {os.path.basename(c)}{marker}')

    section('PROCESSING CAPTURES')
    clouds: list[o3d.geometry.PointCloud] = []
    skipped: list[str] = []

    for i, cap_dir in enumerate(captures):
        cap_name = os.path.basename(cap_dir)
        if cap_name in skip_set or (only_set and cap_name not in only_set):
            print(f'\n--- Capture {i+1}/{len(captures)}: {cap_name} [SKIPPED] ---')
            skipped.append(cap_name)
            continue

        subsection(f'Capture {i+1}/{len(captures)}: {cap_name}')

        try:
            fmt = detect_scan_format(cap_dir)

            if fmt == 'legacy' and mode in ('A', 'B'):
                print(f'    WARNING: legacy scan format detected but mode={mode}. '
                      f'Use --legacy-pose for this session.')
                skipped.append(cap_name)
                continue
            if fmt == 'new' and mode == 'legacy':
                print(f'    WARNING: new scan format but --legacy-pose specified. '
                      f'Processing as Mode A instead.')
                effective_mode = 'A'
            else:
                effective_mode = mode

            data = load_capture(cap_dir)
            print(f'    format: {fmt}  |  color: {data["color"].shape}  '
                  f'depth: {data["depth"].shape} dtype={data["depth"].dtype}')

            pcd = capture_to_pcd(
                data,
                depth_scale=args.depth_scale,
                depth_trunc=args.depth_trunc,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                mode=effective_mode,
                session_calib=session_calib,
                override_tf=override_tf,
                legacy_corr=legacy_corr,
            )

            if pcd is not None:
                if args.icp and len(clouds) > 0:
                    reference = clouds[0]
                    for prev in clouds[1:]:
                        reference = reference + prev
                    if args.voxel_size > 0:
                        reference = reference.voxel_down_sample(args.voxel_size)
                    pcd, rmse, n_corr = icp_refine(pcd, reference, args.icp_threshold)
                    print(f'    ICP: rmse={rmse:.4f}m  correspondences={n_corr}')
                    # Re-apply bbox crop: ICP can shift points across the boundary
                    if bbox_min is not None and bbox_max is not None and n_corr > 0:
                        pcd = crop_pcd(pcd, bbox_min, bbox_max)
                clouds.append(pcd)
                print(f'    added ({len(pcd.points)} pts)')
            else:
                skipped.append(cap_name)

        except Exception as e:
            print(f'    ERROR: {e}')
            skipped.append(cap_name)

    section('MERGING')
    print(f'  processed : {len(clouds)} / {len(captures)}')
    if skipped:
        print(f'  skipped   : {skipped}')
    if not clouds:
        print('ERROR: no valid clouds to merge. Exiting.')
        sys.exit(1)
    merged = clouds[0]
    for pcd in clouds[1:]:
        merged += pcd
    print(f'  total points after merge: {len(merged.points)}')

    # ------------------------------------------------------------------
    # Debug-bbox early exit: show full cloud + wireframe, skip saving
    # ------------------------------------------------------------------
    if args.debug_bbox:
        section('DEBUG BBOX PREVIEW')
        if args.voxel_size > 0:
            merged = merged.voxel_down_sample(voxel_size=args.voxel_size)
            print(f'  cloud downsampled to {len(merged.points)} points')
        if _wire_min is None or _wire_max is None:
            print('  WARNING: no bounding box configured in workspace.yaml.')
            print('           Set bounding_box values in config/workspace.yaml first.')
        else:
            print(f'  bbox min : {[round(v, 4) for v in _wire_min]}')
            print(f'  bbox max : {[round(v, 4) for v in _wire_max]}')

            # --- Always save a MeshLab-compatible fallback PLY ---
            edge_cloud  = make_bbox_edge_cloud(_wire_min, _wire_max)
            debug_cloud = merged + edge_cloud
            debug_path  = os.path.join(session_dir, 'debug_bbox.ply')
            o3d.io.write_point_cloud(debug_path, debug_cloud)
            print()
            print(f'  Saved debug PLY (bbox edges in red):')
            print(f'    {debug_path}')
            print(f'  Open with:  meshlab {debug_path}')
            print()
            print('  RED POINTS  = workspace bounding box edges')
            print('  Other points = full uncropped scene (no crop applied)')
            print('  Points that fall OUTSIDE the red box are removed in a normal run.')
            print()

            # --- Also try the Open3D live viewer ---
            print('  Attempting Open3D viewer...')
            print('  (If it fails on Wayland, use the MeshLab file above instead)')
            print('  Viewer controls: left-drag=rotate  right-drag=pan  scroll=zoom')
            print('  Close the viewer window to exit.')
            bbox_wire = make_bbox_lineset(_wire_min, _wire_max)
            o3d.visualization.draw_geometries(  # type: ignore[attr-defined]
                [merged, bbox_wire],
                window_name='BBox Debug — close to exit',
                width=1280,
                height=720,
            )
        print('\nDone (debug mode — output not saved).\n')
        return

    section('DOWNSAMPLING')
    if args.voxel_size > 0:
        before = len(merged.points)
        merged = merged.voxel_down_sample(voxel_size=args.voxel_size)
        after  = len(merged.points)
        print(f'  {before} -> {after} points  ({100.0*(before-after)/before:.1f}% reduction)')
    else:
        print('  skipping (voxel-size=0)')

    section('RADIUS OUTLIER REMOVAL')
    before_ror = len(merged.points)
    merged, _ = merged.remove_radius_outlier(nb_points=ROR_NB_POINTS, radius=ROR_RADIUS_M)
    after_ror = len(merged.points)
    if before_ror > 0:
        print(f'  {before_ror} -> {after_ror} points  ({100.0*(before_ror-after_ror)/before_ror:.1f}% removed)')
    else:
        print('  skipping (empty cloud)')

    section('CLUSTER FILTERING')
    labels = np.array(merged.cluster_dbscan(eps=DBSCAN_EPS_M, min_points=DBSCAN_MIN_POINTS, print_progress=False))
    if labels.max() >= 0:
        counts = np.bincount(labels[labels >= 0])
        before_cluster = len(merged.points)
        # Keep all clusters large enough to be part of the object; discard small stray blobs.
        keep_labels = np.where(counts >= DBSCAN_MIN_CLUSTER_PTS)[0]
        removed_labels = np.where(counts < DBSCAN_MIN_CLUSTER_PTS)[0]
        keep_mask = np.isin(labels, keep_labels)
        keep = np.where(keep_mask)[0]
        merged = merged.select_by_index(keep)
        kept_pts   = sum(counts[l] for l in keep_labels)
        removed_pts = before_cluster - len(merged.points)
        print(f'  {len(counts)} cluster(s) found, {len(keep_labels)} kept (>={DBSCAN_MIN_CLUSTER_PTS} pts), {len(removed_labels)} removed')
        print(f'  {before_cluster} -> {len(merged.points)} points  ({removed_pts} removed)')
    else:
        print('  no clusters found — skipping')

    section('NORMAL ESTIMATION')
    merged.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS_M, max_nn=NORMAL_MAX_NN)
    )
    merged.orient_normals_consistent_tangent_plane(k=NORMAL_ORIENT_K)
    # Ensure normals point outward: average dot product with centroid-to-point vector should be positive
    pts     = np.asarray(merged.points)
    normals = np.asarray(merged.normals)
    centroid = pts.mean(axis=0)
    outward_dot = (normals * (pts - centroid)).sum(axis=1).mean()
    if outward_dot < 0:
        merged.normals = o3d.utility.Vector3dVector(-normals)
        print('  normals flipped to point outward (were inward on average)')
    print(f'  normals estimated ({len(merged.points)} points)')

    section('SAVING')
    if os.path.exists(models_dir):
        shutil.rmtree(models_dir)
        print(f'  cleared previous models dir: {models_dir}')
    os.makedirs(models_dir, exist_ok=True)
    print(f'  writing to: {output_path}')
    success = o3d.io.write_point_cloud(output_path, merged)
    if success:
        print(f'  saved ({os.path.getsize(output_path)/1024/1024:.2f} MB)')
    else:
        print('  ERROR: write failed.')
        sys.exit(1)

    write_provenance(
        session_dir=session_dir,
        models_dir=models_dir,
        output_path=output_path,
        args=args,
        mode=mode,
        captures_found=len(captures),
        captures_processed=len(clouds),
        skipped=skipped,
        final_point_count=len(merged.points),
    )

    section('SUMMARY')
    print(f'  captures found     : {len(captures)}')
    print(f'  captures processed : {len(clouds)}')
    print(f'  captures skipped   : {len(skipped)}')
    print(f'  final point count  : {len(merged.points)}')
    print(f'  output             : {output_path}')

    pts = np.asarray(merged.points)
    if len(pts) > 0:
        centroid = pts.mean(axis=0)
        print(f'  object centroid    : x={centroid[0]:.4f}  y={centroid[1]:.4f}  z={centroid[2]:.4f}  (base_link frame)')

    section('VISUALISATION')
    if not args.no_visualise:
        print('  opening Open3D viewer... (close window to exit)')
        o3d.visualization.draw_geometries(  # type: ignore[attr-defined]
            [merged],
            window_name='Reconstruction',
            width=1280,
            height=720,
        )
    else:
        print('  skipped (--no-visualise)')

    if args.mesh:
        section('MESHING')
        mesh_script = str(Path(__file__).resolve().parent / 'mesh_from_cloud.py')
        mesh_cmd = [
            sys.executable, mesh_script,
            '--cloud',               output_path,
            '--smooth-passes',       str(args.mesh_smooth_passes),
            '--depth',               str(args.mesh_depth),
            '--density-threshold',   str(args.mesh_density_threshold),
            '--format',              'obj',
            '--no-visualise',
        ]
        print(f'  cloud  : {output_path}')
        print(f'  depth  : {args.mesh_depth}   smooth-passes : {args.mesh_smooth_passes}'
              f'   density-threshold : {args.mesh_density_threshold}')
        print(f'  running mesh_from_cloud.py...')
        result = subprocess.run(mesh_cmd, env=os.environ.copy())
        if result.returncode != 0:
            print(f'  WARNING: mesh_from_cloud.py exited with code {result.returncode}')
        else:
            print('  meshing complete.')

    print('\nDone.\n')


if __name__ == '__main__':
    main()
