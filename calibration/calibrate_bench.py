#!/usr/bin/env python3
# 41069 Robotics Studio 2
# Authors: Jayden Hazell (24953364), Lachlan Partridge (24964557), Tomas Klimes (24773804)
# Fits a RANSAC plane to an empty-bench scan and writes the bench Z height to workspace.yaml.
"""
calibrate_bench.py

Fits a horizontal plane to an empty-bench scan and writes the bench surface
Z height (in base_link frame) to config/workspace.yaml.

Run once against an empty-bench scan before scanning objects:
    python3 scripts/calibration/calibrate_bench.py --session scans/empty_bench

The script applies the workspace bounding_box x/y limits from workspace.yaml
before RANSAC so that only the area directly above the scan workspace is used
for plane fitting. This prevents walls, floor, or robot links from being fitted.

After the run, workspace.yaml will have bench.z_m set. The next reconstruct.py
invocation will automatically compute z_min = bench.z_m + bench.margin_m and
use it as the lower z crop bound.
"""

import argparse
import datetime
import os
import sys
import yaml
from pathlib import Path

import numpy as np
import open3d as o3d

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / 'helpers'))
import reconstruct as R


# ==============================================================================
# Args
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Fit bench plane from empty-bench scan and calibrate workspace.yaml.'
    )
    p.add_argument('--session', required=True,
                   help='Path to empty-bench scan session directory.')
    p.add_argument('--workspace-config', default=None,
                   help='Override path to workspace.yaml (auto-detected when omitted).')
    p.add_argument('--distance-threshold', type=float, default=0.005,
                   help='RANSAC inlier tolerance in metres (default: 0.005).')
    p.add_argument('--ransac-n', type=int, default=3,
                   help='RANSAC minimal sample size (default: 3).')
    p.add_argument('--num-iterations', type=int, default=1000,
                   help='RANSAC iterations (default: 1000).')
    p.add_argument('--depth-trunc', type=float, default=2.0,
                   help='Max depth in metres for reconstruction (default: 2.0).')
    p.add_argument('--depth-scale', type=float, default=1.0)
    p.add_argument('--no-visualise', action='store_true',
                   help='Skip Open3D visualisation.')
    return p.parse_args()


# ==============================================================================
# Helpers
# ==============================================================================

def find_workspace_yaml(session_dir: str, override: str | None) -> Path:
    if override:
        p = Path(override)
    else:
        p = Path(session_dir).parent.parent / 'config' / 'workspace.yaml'
    if not p.exists():
        print(f'ERROR: workspace.yaml not found: {p}')
        print('       Create it first by running reconstruct.py once (it auto-generates the file).')
        sys.exit(1)
    return p


def build_merged_cloud(
    session_dir: str,
    depth_trunc: float,
    depth_scale: float,
) -> o3d.geometry.PointCloud:
    caps = R.find_captures(session_dir)
    if not caps:
        print(f'ERROR: no captures found in {session_dir}')
        sys.exit(1)

    print(f'  found {len(caps)} capture(s)')
    clouds = []

    for cap_dir in caps:
        name = os.path.basename(cap_dir)
        try:
            fmt = R.detect_scan_format(cap_dir)
            if fmt != 'new':
                print(f'  {name}: legacy format — skipping')
                continue
            data = R.load_capture(cap_dir)
            pcd = R.capture_to_pcd(
                data,
                depth_scale=depth_scale,
                depth_trunc=depth_trunc,
                bbox_min=None,
                bbox_max=None,
                mode='A',
                session_calib=None,
                override_tf=None,
                legacy_corr=None,
            )
            if pcd is not None:
                clouds.append(pcd)
                print(f'  {name}: {len(pcd.points)} pts')
            else:
                print(f'  {name}: empty cloud — skipped')
        except Exception as e:
            print(f'  {name}: ERROR — {e}')

    if not clouds:
        print('ERROR: no valid clouds reconstructed.')
        sys.exit(1)

    merged = clouds[0]
    for c in clouds[1:]:
        merged += c
    return merged


def crop_xy(
    pcd: o3d.geometry.PointCloud,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
) -> o3d.geometry.PointCloud:
    pts = np.asarray(pcd.points)
    mask = (
        (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) &
        (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max)
    )
    return pcd.select_by_index(np.where(mask)[0])


def write_workspace_yaml(cfg_path: Path, workspace: dict) -> None:
    """Write workspace.yaml preserving comments and formatting."""
    bench = workspace.get('bench', {})
    bb = workspace.get('bounding_box', {})

    z_m = bench.get('z_m')
    z_m_str = 'null' if z_m is None else f'{float(z_m):.6f}'
    session_str = bench.get('calibration_session') or 'null'
    ts_str = bench.get('calibration_timestamp') or 'null'
    margin = bench.get('margin_m', 0.010)

    content = (
        '# AUTO-GENERATED SECTION — do not edit bench.z_m or bench.calibration_* manually\n'
        '# USER-EDITABLE SECTION — set bounding_box values to surround your object area\n'
        'bench:\n'
        f'  z_m: {z_m_str}\n'
        f'  margin_m: {margin}\n'
        f'  calibration_session: {session_str}\n'
        f'  calibration_timestamp: {ts_str}\n'
        'bounding_box:\n'
        f'  x_min: {bb.get("x_min", -0.30)}\n'
        f'  x_max:  {bb.get("x_max",  0.30)}\n'
        f'  y_min:  {bb.get("y_min",  0.10)}\n'
        f'  y_max:  {bb.get("y_max",  0.50)}\n'
        f'  z_max:  {bb.get("z_max",  0.60)}\n'
        '# z_min is NEVER stored. Computed at runtime as: bench.z_m + bench.margin_m\n'
    )
    with open(cfg_path, 'w') as f:
        f.write(content)


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()
    session_dir = os.path.abspath(args.session)

    print('=' * 60)
    print('  BENCH CALIBRATION')
    print('=' * 60)

    if not os.path.isdir(session_dir):
        print(f'ERROR: directory not found: {session_dir}')
        sys.exit(1)

    cfg_path = find_workspace_yaml(session_dir, args.workspace_config)
    with open(cfg_path) as f:
        workspace = yaml.safe_load(f)

    print(f'  session     : {session_dir}')
    print(f'  workspace   : {cfg_path}')

    bb = workspace.get('bounding_box', {})
    x_min = float(bb.get('x_min', -0.30))
    x_max = float(bb.get('x_max',  0.30))
    y_min = float(bb.get('y_min',  0.10))
    y_max = float(bb.get('y_max',  0.50))
    print(f'  XY crop     : x=[{x_min}, {x_max}]  y=[{y_min}, {y_max}]')
    print()

    # ------------------------------------------------------------------
    # Build merged cloud from all captures (no crop yet)
    # ------------------------------------------------------------------
    print('--- Building merged cloud ---')
    merged = build_merged_cloud(session_dir, args.depth_trunc, args.depth_scale)
    print(f'  total before crop : {len(merged.points)} points')

    # ------------------------------------------------------------------
    # Crop to workspace XY to isolate bench area
    # ------------------------------------------------------------------
    cropped = crop_xy(merged, x_min, x_max, y_min, y_max)
    n_cropped = len(cropped.points)
    print(f'  after XY crop     : {n_cropped} points')

    if n_cropped < 100:
        print('ERROR: too few points after XY crop.')
        print('       Check workspace.yaml bounding_box x/y values match the scan area.')
        sys.exit(1)

    # ------------------------------------------------------------------
    # RANSAC plane fit
    # ------------------------------------------------------------------
    print()
    print('--- Fitting bench plane (RANSAC) ---')
    plane_model, inliers = cropped.segment_plane(
        distance_threshold=args.distance_threshold,
        ransac_n=args.ransac_n,
        num_iterations=args.num_iterations,
    )
    a, b, c, d = [float(v) for v in plane_model]

    # Orient normal upward (+Z)
    if c < 0:
        a, b, c, d = -a, -b, -c, -d

    bench_z = -d / c
    tilt_deg = float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))
    n_in = len(inliers)
    frac = n_in / n_cropped

    print(f'  normal        : [{a:.4f}, {b:.4f}, {c:.4f}]')
    print(f'  tilt          : {tilt_deg:.2f}° from horizontal')
    print(f'  inliers       : {n_in} / {n_cropped} ({100 * frac:.1f}%)')
    print(f'  bench Z       : {bench_z:.6f} m  (in base_link frame)')

    if frac < 0.30:
        print('  WARNING: inlier fraction <30% — fit may not be the table surface.')
        print('           Consider re-running with a smaller --distance-threshold.')
    if tilt_deg > 5.0:
        print(f'  WARNING: plane tilted {tilt_deg:.2f}° — verify table is level and robot base is stable.')

    # ------------------------------------------------------------------
    # Write to workspace.yaml
    # ------------------------------------------------------------------
    if 'bench' not in workspace:
        workspace['bench'] = {}
    workspace['bench']['z_m'] = round(bench_z, 6)
    workspace['bench']['calibration_session'] = os.path.basename(session_dir)
    workspace['bench']['calibration_timestamp'] = datetime.datetime.now().isoformat(timespec='seconds')

    write_workspace_yaml(cfg_path, workspace)

    margin = float(workspace['bench'].get('margin_m', 0.010))
    z_min_eff = bench_z + margin

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print('=' * 60)
    print('  RESULT')
    print('=' * 60)
    print(f'  bench z_m    : {bench_z:.6f} m')
    print(f'  margin_m     : {margin * 1000:.0f} mm')
    print(f'  z_min (eff)  : {z_min_eff:.6f} m  ← objects below this are cropped')
    print(f'  written to   : {cfg_path}')

    # ------------------------------------------------------------------
    # Visualise
    # ------------------------------------------------------------------
    if not args.no_visualise:
        print()
        inlier_cloud  = cropped.select_by_index(inliers)
        outlier_cloud = cropped.select_by_index(inliers, invert=True)
        inlier_cloud.paint_uniform_color([0.0, 0.8, 0.0])
        outlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])
        print('  green = bench inliers | grey = outliers')
        print('  close window to exit')
        o3d.visualization.draw_geometries(  # type: ignore[attr-defined]
            [inlier_cloud, outlier_cloud],
            window_name='Bench Calibration',
            width=1280,
            height=720,
        )

    print()
    print('Done.')


if __name__ == '__main__':
    main()
