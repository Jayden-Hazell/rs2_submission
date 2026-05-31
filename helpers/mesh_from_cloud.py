#!/usr/bin/env python3
# 41069 Robotics Studio 2
# Authors: Jayden Hazell (24953364), Lachlan Partridge (24964557), Tomas Klimes (24773804)
# Converts a merged point cloud into a repaired, smoothed, multi-format 3D mesh.
"""
mesh_from_cloud.py

Converts a merged point cloud (reconstruction.ply from reconstruct.py) into a
coloured 3D mesh with repair, smoothing, and multi-format export.

Pipeline:
  1. Load point cloud (with normals and per-vertex colour from reconstruct.py)
  2. Poisson (default) or BPA surface reconstruction → coloured mesh
  3. Post-Poisson density filter to remove fringe artefacts
  4. Open3D mesh cleanup (degenerate/duplicate geometry, non-manifold edges)
  5. Plane cut at bench z_min (Option B: flat underside)
  6. PyMeshLab hole fill and non-manifold repair
  7. Taubin smoothing (volume-preserving)
  8. Quadric-error decimation (optional)
  9. Export: PLY / OBJ / STL / GLB
 10. Provenance JSON

Usage:
    python3 helpers/mesh_from_cloud.py \\
        --cloud reconstructed_scans/<name>/reconstruction.ply [options]

Options:
    --cloud             Input point cloud PLY (required)
    --algorithm         poisson|bpa (default: poisson)
    --depth             Poisson octree depth 6-12 (default: 9). Higher = more detail, slower.
    --density-threshold Remove vertices below this density percentile after Poisson (default: 0.1)
    --bpa-radii         Ball-pivoting radii in metres (default: 0.005 0.01 0.02)
    --cap-bottom        Add flat cap at bench z_min — closes the unscanned underside (default: on)
    --no-cap-bottom     Disable flat bottom cap
    --z-min             Override cap height in metres (auto-detected from workspace.yaml if omitted)
    --repair            Run PyMeshLab hole fill + non-manifold repair (default: on)
    --no-repair         Disable PyMeshLab repair
    --smooth-passes     Taubin smoothing iterations (default: 5, 0 to skip)
    --decimate-target   Target triangle count (0 = no decimation)
    --format            Output format(s): ply obj stl glb (default: ply)
    --output-dir        Override output directory (default: same folder as --cloud)
    --no-visualise      Skip Open3D viewer
    --workspace-config  Override path to workspace.yaml
"""

import os
import sys
import json
import yaml
import argparse
import datetime
import numpy as np
from pathlib import Path

import open3d as o3d
import trimesh
import trimesh.visual.color
import pymeshlab


# ==============================================================================
# TUNABLE CONSTANTS
# Edit these values to adjust mesh quality.
# See RECONSTRUCTION_TUNING_GUIDE.txt for a full explanation of each.
# ==============================================================================

# --- Poisson surface reconstruction ---
POISSON_DEPTH             = 7    # Octree depth. Higher = more detail but slower and noisier.
                                  # 7 = smooth (good for simple objects with basic geometry)
                                  # 8 = moderate detail, resolves ~3mm features including scan noise
                                  # 9+ = high detail, but resolves scan noise as real geometry (lumpy)
POISSON_DENSITY_THRESHOLD = 0.20 # Fraction (0–1) of low-confidence vertices removed after Poisson.
                                  # These are vertices where Poisson invented surface with little data.
                                  # Raise (0.3+) to cut more fringe/skirt. Lower (0.05) to keep more.
                                  # Kept at 0.20: higher values punch holes in the mesh on sparse clouds.

# --- Smoothing ---
SMOOTH_PASSES = 80               # Taubin smoothing iterations. Volume-preserving (won't shrink mesh).
                                  # 0 = no smoothing. 20 = moderate. 40 = strong. 80 = very strong.
                                  # Raise to reduce bumpy noise. Lower to preserve sharp real edges.

# --- Hole filling ---
HOLE_MAX_BOUNDARY_EDGES = 1000   # Max boundary edge count of a hole that will be filled.
                                  # Raise to fill larger holes. Lower to leave large openings intact.
                                  # The flat bottom cap hole is always large — keep this value high.

# ==============================================================================
# Argument parsing
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Mesh a point cloud with repair and multi-format export.'
    )
    parser.add_argument(
        '--cloud', required=True,
        help='Input point cloud PLY (e.g. reconstructed_scans/<name>/reconstruction.ply).'
    )

    alg = parser.add_argument_group('Reconstruction')
    alg.add_argument('--algorithm', choices=['poisson', 'bpa'], default='poisson')
    alg.add_argument(
        '--depth', type=int, default=POISSON_DEPTH,
        help=f'Poisson octree depth 6-12 (default: {POISSON_DEPTH}). Higher = more detail / slower.'
    )
    alg.add_argument(
        '--density-threshold', type=float, default=POISSON_DENSITY_THRESHOLD,
        help=f'Post-Poisson: remove vertices below this density percentile (default: {POISSON_DENSITY_THRESHOLD}).'
    )
    alg.add_argument(
        '--bpa-radii', type=float, nargs='+', default=[0.005, 0.01, 0.02],
        help='BPA ball radii in metres (default: 0.005 0.01 0.02).'
    )

    cap = parser.add_argument_group('Bottom cap')
    cap.add_argument(
        '--cap-bottom', action='store_true', default=True,
        help='Add a flat cap at bench z_min to close the unscanned underside (default: on).'
    )
    cap.add_argument('--no-cap-bottom', dest='cap_bottom', action='store_false')
    cap.add_argument(
        '--z-min', type=float, default=None,
        help='Override cap height in metres. Auto-detected from workspace.yaml if omitted.'
    )

    post = parser.add_argument_group('Post-processing')
    post.add_argument(
        '--repair', action='store_true', default=True,
        help='Run PyMeshLab hole fill and non-manifold repair (default: on).'
    )
    post.add_argument('--no-repair', dest='repair', action='store_false')
    post.add_argument(
        '--smooth-passes', type=int, default=SMOOTH_PASSES,
        help=f'Taubin smoothing iterations (default: {SMOOTH_PASSES}, 0 to skip).'
    )
    post.add_argument(
        '--decimate-target', type=int, default=0,
        help='Target triangle count after quadric decimation (0 = disabled).'
    )

    out = parser.add_argument_group('Output')
    out.add_argument(
        '--format', nargs='+', default=['obj'],
        choices=['ply', 'obj', 'stl', 'glb'],
        help='Output format(s) (default: obj).'
    )
    out.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory (default: same folder as --cloud).'
    )
    out.add_argument('--no-visualise', action='store_true')
    out.add_argument(
        '--workspace-config', type=str, default=None,
        help='Path to workspace.yaml. Auto-detected from repo root if omitted.'
    )

    return parser.parse_args()


# ==============================================================================
# Helpers
# ==============================================================================

def section(title: str) -> None:
    print()
    print('=' * 60)
    print(f'  {title}')
    print('=' * 60)


def load_z_min(
    cloud_path: str,
    override_yaml: str | None,
    override_z: float | None,
) -> float | None:
    """
    Determine z_min for the flat bottom cap.
    Priority: --z-min CLI > workspace.yaml bench calibration > None (cap disabled).

    cloud_path is at reconstructed_scans/<name>/reconstruction.ply,
    so repo root is 3 levels up.
    """
    if override_z is not None:
        return override_z

    if override_yaml:
        cfg_path = Path(override_yaml)
    else:
        cfg_path = Path(cloud_path).parent.parent.parent / 'config' / 'workspace.yaml'

    if not cfg_path.exists():
        return None

    with open(cfg_path) as f:
        ws = yaml.safe_load(f)

    bench = ws.get('bench', {})
    z_m = bench.get('z_m')
    margin = float(bench.get('margin_m', 0.010))

    if z_m is None:
        return None
    return float(z_m) + margin


# ==============================================================================
# Conversion utilities
# ==============================================================================

def o3d_mesh_to_trimesh(mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    vertices = np.asarray(mesh.vertices)
    faces    = np.asarray(mesh.triangles)

    if mesh.has_vertex_colors():
        rgb   = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
        alpha = np.full((len(rgb), 1), 255, dtype=np.uint8)
        colors = np.hstack([rgb, alpha])
    else:
        colors = None

    return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)


def trimesh_to_pymeshlab(tm: trimesh.Trimesh) -> pymeshlab.MeshSet:
    # Ensure we have ColorVisuals (not TextureVisuals)
    if not isinstance(tm.visual, trimesh.visual.color.ColorVisuals):
        tm.visual = tm.visual.to_color()

    v = np.asarray(tm.vertices, dtype=np.float64)
    f = np.asarray(tm.faces,    dtype=np.int32)

    vc_uint8 = np.asarray(tm.visual.vertex_colors, dtype=np.uint8)  # Nx4 RGBA
    vc = vc_uint8.astype(np.float64) / 255.0                         # Nx4 float [0,1]

    pm_mesh = pymeshlab.Mesh(vertex_matrix=v, face_matrix=f, v_color_matrix=vc)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pm_mesh)
    return ms


def pymeshlab_to_trimesh(ms: pymeshlab.MeshSet) -> trimesh.Trimesh:
    m  = ms.current_mesh()
    v  = m.vertex_matrix()
    f  = m.face_matrix()
    vc = m.vertex_color_matrix()                          # Nx4 RGBA float [0,1]
    colors_uint8 = (vc * 255).astype(np.uint8)
    return trimesh.Trimesh(vertices=v, faces=f, vertex_colors=colors_uint8, process=False)


def pymeshlab_to_o3d(ms: pymeshlab.MeshSet) -> o3d.geometry.TriangleMesh:
    m   = ms.current_mesh()
    v   = m.vertex_matrix()
    f   = m.face_matrix()
    vc  = m.vertex_color_matrix()[:, :3]  # RGB float [0,1]
    out = o3d.geometry.TriangleMesh()
    out.vertices      = o3d.utility.Vector3dVector(v)
    out.triangles     = o3d.utility.Vector3iVector(f)
    out.vertex_colors = o3d.utility.Vector3dVector(vc)
    out.compute_vertex_normals()
    return out


# ==============================================================================
# Main
# ==============================================================================

def main():
    args = parse_args()

    cloud_path = os.path.abspath(args.cloud)
    if not os.path.exists(cloud_path):
        print(f'ERROR: cloud file not found: {cloud_path}')
        sys.exit(1)

    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.dirname(cloud_path)
    os.makedirs(output_dir, exist_ok=True)

    z_min: float | None = None
    if args.cap_bottom:
        z_min = load_z_min(cloud_path, args.workspace_config, args.z_min)

    # ------------------------------------------------------------------
    section('CONFIGURATION')
    # ------------------------------------------------------------------
    print(f'  cloud             : {cloud_path}')
    print(f'  output dir        : {output_dir}')
    print(f'  algorithm         : {args.algorithm}')
    if args.algorithm == 'poisson':
        print(f'  poisson depth     : {args.depth}')
        print(f'  density threshold : {args.density_threshold:.0%} percentile')
    else:
        print(f'  bpa radii         : {args.bpa_radii} m')
    if args.cap_bottom:
        cap_str = f'z_min = {z_min:.4f} m' if z_min is not None else 'DISABLED (z_min not found)'
        print(f'  bottom cap        : {cap_str}')
    else:
        print('  bottom cap        : off (--no-cap-bottom)')
    print(f'  repair            : {args.repair}')
    print(f'  smooth passes     : {args.smooth_passes}')
    print(f'  decimate target   : {args.decimate_target if args.decimate_target > 0 else "disabled"}')
    print(f'  formats           : {args.format}')

    # ------------------------------------------------------------------
    section('LOADING POINT CLOUD')
    # ------------------------------------------------------------------
    pcd = o3d.io.read_point_cloud(cloud_path)
    print(f'  points     : {len(pcd.points)}')
    print(f'  has colors : {pcd.has_colors()}')
    print(f'  has normals: {pcd.has_normals()}')

    if len(pcd.points) == 0:
        print('ERROR: point cloud is empty.')
        sys.exit(1)

    if not pcd.has_normals():
        print('  normals missing — estimating...')
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)
        print('  normals estimated.')

    # ------------------------------------------------------------------
    section('SURFACE RECONSTRUCTION')
    # ------------------------------------------------------------------
    if args.algorithm == 'poisson':
        print(f'  running Poisson reconstruction (depth={args.depth})...')
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=args.depth
        )
        print(f'  raw mesh : {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles')

        densities_np = np.asarray(densities)
        threshold_val = np.quantile(densities_np, args.density_threshold)
        vertices_to_remove = densities_np < threshold_val
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print(
            f'  after density filter ({args.density_threshold:.0%} percentile): '
            f'{len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles'
        )
    else:
        print(f'  running Ball-Pivoting (radii={args.bpa_radii} m)...')
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(args.bpa_radii)
        )
        print(f'  mesh : {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles')

    if len(mesh.triangles) == 0:
        print('ERROR: reconstruction produced no triangles. '
              'Try --depth with a lower value, or switch to --algorithm bpa.')
        sys.exit(1)

    # ------------------------------------------------------------------
    section('OPEN3D MESH CLEANUP')
    # ------------------------------------------------------------------
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    print(f'  {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles')

    # ------------------------------------------------------------------
    section('CONVERTING TO TRIMESH')
    # ------------------------------------------------------------------
    tm = o3d_mesh_to_trimesh(mesh)
    print(f'  {len(tm.vertices)} vertices, {len(tm.faces)} faces')

    # ------------------------------------------------------------------
    # Option B: flat bottom cap
    # Cut the mesh at z_min so the unscanned underside is replaced with a
    # clean planar face instead of Poisson's invented geometry.
    # ------------------------------------------------------------------
    if args.cap_bottom and z_min is not None:
        section('BOTTOM CAP (plane cut at z_min)')
        n_before = len(tm.faces)
        print(f'  cutting at z = {z_min:.4f} m — keeping everything above ...')
        # slice_mesh_plane keeps the half-space where dot(normal, p - origin) >= 0
        # normal=[0,0,1], origin=[0,0,z_min]  →  keeps z >= z_min
        tm = trimesh.intersections.slice_mesh_plane(
            tm,
            plane_normal=[0.0, 0.0, 1.0],
            plane_origin=[0.0, 0.0, z_min],
        )
        print(f'  faces after cut  : {len(tm.faces)}  (was {n_before})')
        print(f'  open boundary at z={z_min:.4f} m will be capped by PyMeshLab repair.')
    elif args.cap_bottom:
        print()
        print('  WARNING: cap-bottom requested but z_min could not be determined.')
        print('           Run calibrate_bench.py first, or pass --z-min <metres>.')

    # ------------------------------------------------------------------
    section('PYMESHLAB REPAIR + SMOOTHING')
    # ------------------------------------------------------------------
    ms = trimesh_to_pymeshlab(tm)
    print(f'  loaded: {ms.current_mesh().vertex_number()} vertices, '
          f'{ms.current_mesh().face_number()} faces')

    if args.repair:
        print('  repairing non-manifold edges...')
        ms.meshing_repair_non_manifold_edges()
        print('  repairing non-manifold vertices...')
        ms.meshing_repair_non_manifold_vertices()
        print(f'  filling holes (max boundary length: {HOLE_MAX_BOUNDARY_EDGES} edges)...')
        ms.meshing_close_holes(maxholesize=HOLE_MAX_BOUNDARY_EDGES)
        print(f'  after repair: {ms.current_mesh().vertex_number()} vertices, '
              f'{ms.current_mesh().face_number()} faces')
    else:
        print('  skipped (--no-repair)')

    if args.smooth_passes > 0:
        print(f'  Taubin smoothing ({args.smooth_passes} passes)...')
        ms.apply_coord_taubin_smoothing(stepsmoothnum=args.smooth_passes)
    else:
        print('  smoothing skipped (--smooth-passes 0)')

    if args.decimate_target > 0:
        current = ms.current_mesh().face_number()
        print(f'  decimating {current} → {args.decimate_target} triangles...')
        ms.simplification_quadric_edge_collapse_decimation(targetfacenum=args.decimate_target)
        print(f'  after decimation: {ms.current_mesh().face_number()} triangles')

    # ------------------------------------------------------------------
    section('EXPORTING')
    # ------------------------------------------------------------------
    # Name output files after the scan folder (e.g. Black_Box.ply)
    stem = os.path.basename(os.path.dirname(cloud_path))
    outputs: dict[str, str] = {}

    # Convert once for trimesh-based exports (OBJ, GLB, STL)
    tm_final: trimesh.Trimesh | None = None

    for fmt in args.format:
        out_path = os.path.join(output_dir, f'{stem}.{fmt}')
        if fmt == 'ply':
            # PLY preserves vertex colours natively
            ms.save_current_mesh(out_path)
        elif fmt == 'obj':
            # Write OBJ manually — trimesh's and PyMeshLab's exporters both trip
            # MeshLab's buggy OBJ importer assertion. Hand-written format with an
            # explicit group line and per-vertex normals (vn) works universally
            # (MeshLab, Fusion 360, Blender, etc.).
            m = ms.current_mesh()
            verts   = m.vertex_matrix()
            faces   = m.face_matrix()
            normals = m.vertex_normal_matrix()
            with open(out_path, 'w') as obj_f:
                obj_f.write('g default\n')
                for v in verts:
                    obj_f.write(f'v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n')
                for n in normals:
                    obj_f.write(f'vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n')
                for face in faces:
                    i0, i1, i2 = face[0]+1, face[1]+1, face[2]+1
                    obj_f.write(f'f {i0}//{i0} {i1}//{i1} {i2}//{i2}\n')
        else:
            # GLB / STL via trimesh with full colour support
            if tm_final is None:
                tm_final = pymeshlab_to_trimesh(ms)
            tm_final.export(out_path)
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f'  {fmt.upper():4s}  →  {out_path}  ({size_mb:.2f} MB)')
        outputs[fmt] = out_path

    # ------------------------------------------------------------------
    section('PROVENANCE')
    # ------------------------------------------------------------------
    prov = {
        'timestamp':          datetime.datetime.now().isoformat(timespec='seconds'),
        'cloud_path':         cloud_path,
        'output_dir':         output_dir,
        'algorithm':          args.algorithm,
        'poisson_depth':      args.depth if args.algorithm == 'poisson' else None,
        'density_threshold':  args.density_threshold if args.algorithm == 'poisson' else None,
        'bpa_radii':          args.bpa_radii if args.algorithm == 'bpa' else None,
        'cap_bottom':         args.cap_bottom,
        'z_min_used':         z_min,
        'repair':             args.repair,
        'smooth_passes':      args.smooth_passes,
        'decimate_target':    args.decimate_target,
        'formats':            args.format,
        'outputs':            outputs,
        'final_vertices':     ms.current_mesh().vertex_number(),
        'final_faces':        ms.current_mesh().face_number(),
    }
    prov_path = os.path.join(output_dir, f'{stem}.json')
    with open(prov_path, 'w') as f:
        json.dump(prov, f, indent=2)
    print(f'  {prov_path}')

    # ------------------------------------------------------------------
    section('SUMMARY')
    # ------------------------------------------------------------------
    print(f'  input points  : {len(pcd.points)}')
    print(f'  output verts  : {ms.current_mesh().vertex_number()}')
    print(f'  output faces  : {ms.current_mesh().face_number()}')
    for fmt, path in outputs.items():
        print(f'  {fmt.upper():4s}          : {path}')

    if not args.no_visualise:
        section('VISUALISATION')
        print('  opening Open3D viewer... (close window to exit)')
        vis_mesh = pymeshlab_to_o3d(ms)
        o3d.visualization.draw_geometries(  # type: ignore[attr-defined]
            [vis_mesh],
            window_name='Mesh',
            width=1280,
            height=720,
        )
    else:
        print()
        print('  visualisation skipped (--no-visualise)')

    print('\nDone.\n')


if __name__ == '__main__':
    main()
