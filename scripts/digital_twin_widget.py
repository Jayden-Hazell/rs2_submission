# 41069 Robotics Studio 2
# Authors: Jayden Hazell (24953364), Lachlan Partridge (24964557), Tomas Klimes (24773804)
# Embedded pyqtgraph 3D viewer that renders the UR3e URDF model and tracks live TF transforms.
"""
digital_twin_widget.py

Embedded pyqtgraph digital twin for the ARISS UR3e system.

This widget:
- processes the same rs2_world.urdf.xacro used by the robot system;
- loads every URDF visual mesh into the embedded 3D viewer;
- follows each URDF link using TF from robot_state_publisher;
- supports extra fixed visual links, such as the RealSense D435i mount.

The digital twin is intentionally visual-only. It does not perform collision
checking, planning, or control.
"""

from __future__ import annotations

# Standard library
import os
import xml.etree.ElementTree as ET
from typing import Optional

# Third-party
import numpy as np
import pyqtgraph.opengl as gl
import rclpy
import tf2_ros
import trimesh
import xacro
from ament_index_python.packages import get_package_share_directory
from PySide6.QtCore import QTimer
from PySide6.QtGui import QMatrix4x4
from PySide6.QtWidgets import QVBoxLayout, QWidget


class DigitalTwinWidget(QWidget):
    """URDF/TF based digital twin viewer for the UR3e system."""

    UPDATE_PERIOD_MS = 50
    FIXED_FRAME = "world"
    ROBOT_PACKAGE = "rs2"
    ROBOT_XACRO = os.path.join("urdf", "rs2_world.urdf.xacro")

    def __init__(self, ros_node, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self.node = ros_node
        self.fixed_frame = self.FIXED_FRAME

        # Each entry stores one visual mesh and the TF link it follows.
        self.mesh_items: dict[str, dict[str, object]] = {}

        # Used for resolving relative mesh paths inside the URDF/xacro.
        self.urdf_dir = ""

        self._build_view()
        self._setup_tf()
        self._load_robot_model()
        self._start_update_timer()

    # ------------------------------------------------------------------
    # Initial setup
    # ------------------------------------------------------------------
    def _build_view(self) -> None:
        """Create the embedded 3D viewport and reference grid."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=2.0, elevation=25, azimuth=45)
        layout.addWidget(self.view)

        grid = gl.GLGridItem()
        grid.scale(0.5, 0.5, 0.5)
        self.view.addItem(grid)

    def _setup_tf(self) -> None:
        """Create the TF buffer used to track URDF links in real time."""
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self.node,
            spin_thread=False,
        )

    def _load_robot_model(self) -> None:
        """Find and load the rs2 robot xacro from the installed package share."""
        try:
            package_share = get_package_share_directory(self.ROBOT_PACKAGE)
            xacro_path = os.path.join(package_share, self.ROBOT_XACRO)

            self.node.get_logger().info(f"Digital twin loading: {xacro_path}")
            self.load_xacro(xacro_path)

        except Exception as exc:
            self.node.get_logger().error(f"Digital twin failed to locate robot model: {exc}")

    def _start_update_timer(self) -> None:
        """Start the Qt timer that refreshes mesh transforms from TF."""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_robot)
        self.timer.start(self.UPDATE_PERIOD_MS)

    # ------------------------------------------------------------------
    # URDF/xacro loading
    # ------------------------------------------------------------------
    def load_xacro(self, xacro_path: str) -> None:
        """Process a xacro file and load all visual mesh links."""
        self.urdf_dir = os.path.dirname(xacro_path)

        try:
            doc = xacro.process_file(xacro_path)
            root = ET.fromstring(doc.toxml())
            self._load_robot_root(root)

        except Exception as exc:
            self.node.get_logger().error(f"Digital twin failed to process xacro: {exc}")

    def _load_robot_root(self, root: ET.Element) -> None:
        """Load every visual mesh found in the processed URDF."""
        self.mesh_items.clear()

        for link in root.findall("link"):
            link_name = link.attrib.get("name", "")

            # A URDF link may contain multiple visual elements. Each visual gets
            # its own GL mesh item but follows the same TF frame.
            for visual_index, visual in enumerate(link.findall("visual")):
                mesh_info = self._extract_mesh_info(visual)
                if mesh_info is None:
                    continue

                mesh_path, mesh_scale, visual_origin = mesh_info
                mesh_item = self._create_mesh_item(link_name, mesh_path, mesh_scale)

                if mesh_item is None:
                    continue

                self.view.addItem(mesh_item)

                item_key = f"{link_name}:{visual_index}"
                self.mesh_items[item_key] = {
                    "link_name": link_name,
                    "item": mesh_item,
                    "origin_matrix": visual_origin,
                }

        self.node.get_logger().info(
            f"Digital twin loaded {len(self.mesh_items)} visual mesh item(s)."
        )

    def _extract_mesh_info(self, visual: ET.Element) -> Optional[tuple[str, np.ndarray, np.ndarray]]:
        """Read mesh filename, scale, and visual origin from one URDF visual tag."""
        geometry = visual.find("geometry")
        if geometry is None:
            return None

        mesh = geometry.find("mesh")
        if mesh is None:
            return None

        filename = mesh.attrib.get("filename", "").strip()
        if not filename:
            return None

        mesh_path = self._resolve_resource_path(filename)
        if not os.path.exists(mesh_path):
            self.node.get_logger().warn(f"Digital twin missing mesh: {mesh_path}")
            return None

        mesh_scale = self._parse_scale(mesh.attrib.get("scale", "1 1 1"))
        visual_origin = self._parse_origin(visual.find("origin"))

        return mesh_path, mesh_scale, visual_origin

    def _create_mesh_item(
        self,
        link_name: str,
        mesh_path: str,
        mesh_scale: np.ndarray,
    ) -> Optional[gl.GLMeshItem]:
        """Load one mesh file and convert it into a pyqtgraph GL mesh item."""
        try:
            loaded_mesh = trimesh.load(mesh_path, force="mesh")

            # Some formats, especially DAE, may load as a scene containing
            # multiple geometries. Merge them so pyqtgraph can render one item.
            if isinstance(loaded_mesh, trimesh.Scene):
                geometries = list(loaded_mesh.geometry.values())
                if not geometries:
                    self.node.get_logger().warn(f"Digital twin empty mesh scene: {mesh_path}")
                    return None
                loaded_mesh = trimesh.util.concatenate(geometries)

            vertices = np.asarray(loaded_mesh.vertices, dtype=float)
            faces = np.asarray(loaded_mesh.faces, dtype=np.uint32)

            if vertices.size == 0 or faces.size == 0:
                self.node.get_logger().warn(f"Digital twin empty mesh: {mesh_path}")
                return None

            vertices = vertices * mesh_scale

            colour = np.asarray(self._colour_for_link(link_name), dtype=float)
            face_colours = np.tile(colour, (faces.shape[0], 1))

            mesh_data = gl.MeshData(
                vertexes=vertices,
                faces=faces,
                faceColors=face_colours,
            )

            mesh_item = gl.GLMeshItem(
                meshdata=mesh_data,
                smooth=False,
                drawEdges=False,
                drawFaces=True,
                shader="shaded",
                glOptions="opaque",
            )

            # Fallback for pyqtgraph versions that do not fully respect
            # per-face colours.
            mesh_item.setColor(tuple(colour))
            return mesh_item

        except Exception as exc:
            self.node.get_logger().warn(f"Digital twin failed to load {mesh_path}: {exc}")
            return None

    def _resolve_resource_path(self, filename: str) -> str:
        """Resolve URDF mesh paths into absolute file paths."""
        filename = os.path.expandvars(os.path.expanduser(filename.strip()))

        if filename.startswith("package://"):
            remainder = filename[len("package://"):]
            package_name, relative_path = remainder.split("/", 1)
            package_share = get_package_share_directory(package_name)
            return os.path.join(package_share, relative_path)

        if filename.startswith("file://"):
            return filename[len("file://"):]

        if filename.startswith("$(find "):
            package_name = filename.split("$(find ")[1].split(")")[0]
            relative_path = filename.split(")", 1)[1].lstrip("/")
            package_share = get_package_share_directory(package_name)
            return os.path.join(package_share, relative_path)

        if os.path.isabs(filename):
            return filename

        return os.path.join(self.urdf_dir, filename)

    # ------------------------------------------------------------------
    # Colour handling
    # ------------------------------------------------------------------
    def _colour_for_link(self, link_name: str) -> tuple[float, float, float, float]:
        """Return a simple RGBA colour based on the URDF link name.

        pyqtgraph does not reliably apply URDF material colours for every mesh
        type, so colours are applied here in code. This keeps the UR3e visually
        consistent while still allowing the camera mount to be coloured later.
        """
        clean_name = self._strip_prefix(link_name)

        ur_light_links = {
            "base_link",
            "base_link_inertia",
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
        }

        if clean_name == "workbench":
            return (0.35, 0.35, 0.35, 1.0)

        if clean_name in ur_light_links:
            return (0.88, 0.88, 0.86, 1.0)

        if clean_name == "tool0":
            return (0.20, 0.20, 0.20, 1.0)

        if "collision" in clean_name:
            return (0.0, 1.0, 0.0, 0.25)

        if (
            "d435" in clean_name
            or "camera" in clean_name
            or "realsense" in clean_name
        ):
            return (0.12, 0.12, 0.12, 1.0)

        return (0.75, 0.75, 0.75, 1.0)

    def _strip_prefix(self, link_name: str) -> str:
        """Remove common tf_prefix text by matching known UR link suffixes."""
        known_suffixes = [
            "base_link_inertia",
            "base_link",
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
            "tool0",
            "workbench",
        ]

        for suffix in known_suffixes:
            if link_name.endswith(suffix):
                return suffix

        return link_name

    # ------------------------------------------------------------------
    # TF update loop
    # ------------------------------------------------------------------
    def update_robot(self) -> None:
        """Update every loaded mesh pose using the latest available TF."""
        for data in self.mesh_items.values():
            link_name = data["link_name"]
            mesh_item = data["item"]
            origin_matrix = data["origin_matrix"]

            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    self.fixed_frame,
                    str(link_name),
                    rclpy.time.Time(),
                )

                world_to_link = self._transform_msg_to_matrix(tf_msg.transform)
                final_matrix = world_to_link @ origin_matrix
                self._apply_matrix(mesh_item, final_matrix)

            except Exception:
                # TF can be unavailable during startup. Silently skip the frame
                # to avoid flooding the terminal while controllers initialise.
                continue

    def _apply_matrix(self, mesh_item: gl.GLMeshItem, matrix_np: np.ndarray) -> None:
        """Apply a 4x4 numpy transform matrix to a pyqtgraph mesh item."""
        qmat = QMatrix4x4(
            float(matrix_np[0, 0]), float(matrix_np[0, 1]), float(matrix_np[0, 2]), float(matrix_np[0, 3]),
            float(matrix_np[1, 0]), float(matrix_np[1, 1]), float(matrix_np[1, 2]), float(matrix_np[1, 3]),
            float(matrix_np[2, 0]), float(matrix_np[2, 1]), float(matrix_np[2, 2]), float(matrix_np[2, 3]),
            float(matrix_np[3, 0]), float(matrix_np[3, 1]), float(matrix_np[3, 2]), float(matrix_np[3, 3]),
        )

        mesh_item.resetTransform()
        mesh_item.applyTransform(qmat, local=False)

    # ------------------------------------------------------------------
    # Transform parsing and maths helpers
    # ------------------------------------------------------------------
    def _parse_origin(self, origin_element: Optional[ET.Element]) -> np.ndarray:
        """Convert a URDF origin tag into a 4x4 transform matrix."""
        if origin_element is None:
            return np.eye(4)

        xyz = self._parse_vector(origin_element.attrib.get("xyz", "0 0 0"))
        rpy = self._parse_vector(origin_element.attrib.get("rpy", "0 0 0"))
        return self._make_transform(xyz, rpy)

    def _parse_scale(self, text: str) -> np.ndarray:
        """Parse URDF mesh scale. Supports either one value or xyz values."""
        values = self._parse_vector(text)

        if len(values) == 1:
            return np.array([values[0], values[0], values[0]], dtype=float)

        if len(values) == 3:
            return values

        return np.array([1.0, 1.0, 1.0], dtype=float)

    def _parse_vector(self, text: str) -> np.ndarray:
        """Parse a whitespace-separated numeric vector from URDF text."""
        try:
            return np.array([float(value) for value in text.split()], dtype=float)
        except ValueError:
            return np.array([0.0, 0.0, 0.0], dtype=float)

    def _transform_msg_to_matrix(self, transform) -> np.ndarray:
        """Convert a geometry_msgs Transform into a 4x4 numpy matrix."""
        translation = np.array(
            [
                transform.translation.x,
                transform.translation.y,
                transform.translation.z,
            ],
            dtype=float,
        )

        rotation = self._quat_to_rot(
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        )

        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        return matrix

    def _quat_to_rot(self, x: float, y: float, z: float, w: float) -> np.ndarray:
        """Convert quaternion xyzw into a 3x3 rotation matrix."""
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        return np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
            ],
            dtype=float,
        )

    def _make_transform(self, xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
        """Build a 4x4 transform from URDF xyz and roll-pitch-yaw values."""
        roll, pitch, yaw = rpy

        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)

        rotation_x = np.array(
            [
                [1, 0, 0],
                [0, cr, -sr],
                [0, sr, cr],
            ],
            dtype=float,
        )

        rotation_y = np.array(
            [
                [cp, 0, sp],
                [0, 1, 0],
                [-sp, 0, cp],
            ],
            dtype=float,
        )

        rotation_z = np.array(
            [
                [cy, -sy, 0],
                [sy, cy, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )

        matrix = np.eye(4)
        matrix[:3, :3] = rotation_z @ rotation_y @ rotation_x
        matrix[:3, 3] = xyz
        return matrix