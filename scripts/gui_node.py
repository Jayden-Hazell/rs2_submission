#!/usr/bin/env python3
# 41069 Robotics Studio 2
# Authors: Jayden Hazell (24953364), Lachlan Partridge (24964557), Tomas Klimes (24773804)
# PySide6 operator GUI that publishes scan commands and displays robot status, camera feeds, and reconstructed models.
"""
gui_node.py

PySide6 + ROS2 GUI node for the ARISS UR3e scanning system.

Main functions:
- Publish scan commands, scan settings, scan name, manual viewpoints, and emergency stop requests.
- Display robot status, scan progress, coverage, camera feed, depth overlay, point cloud, and reconstructed models.
- Bridge ROS2 callbacks into Qt signals so the GUI can update safely from the ROS executor thread.

ROS2 publishers:
    /gui/settings                       std_msgs/String   JSON payload: resolution and speed
    /gui/viewpoint                      std_msgs/String
    /gui/scan_name                      std_msgs/String
    /gui/start_scan                     std_msgs/Bool
    /gui/pause_scan                     std_msgs/Bool
    /gui/emergency_stop                 std_msgs/Bool
    /gui/move_to_pre_scan_position      std_msgs/Bool
    /gui/rescan_section                 std_msgs/Bool
    /gui/sample_object                  std_msgs/Bool
    /heartbeat/gui                      std_msgs/Empty

ROS2 subscribers:
    /control/status                              std_msgs/String
    /movement/progress                           std_msgs/Float32
    /scan/coverage                               std_msgs/Float32
    /system/log                                  std_msgs/String
    /camera/d435i/color/image_raw                sensor_msgs/Image
    /camera/d435i/aligned_depth_to_color/image_raw sensor_msgs/Image
    /camera/d435i/depth/color/points             sensor_msgs/PointCloud2
"""

from __future__ import annotations

# Stdlib
import json
import os
import shutil
import signal
import sys
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
# Third-party: math/geometry/vision
import cv2
import numpy as np
import trimesh

# Third-party: ROS 2
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Bool, Float32, String, Empty

# Third-party: Qt / GUI
import pyqtgraph.opengl as gl
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QStackedLayout,
    QTabBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QLineEdit,
)

# Local
from digital_twin_widget import DigitalTwinWidget
@dataclass
class ScanSettings:
    """Scan settings published to the movement/perception stack."""
    resolution: str = "high"
    speed: str = "medium"


class GuiBridge(QObject):
    """Qt signal bridge so ROS callbacks can update the GUI safely."""
    status_received = Signal(str)
    progress_received = Signal(float)
    coverage_received = Signal(float)
    camera_received = Signal(QImage)
    depth_received = Signal(QImage)
    pointcloud_received = Signal(object)
    log_received = Signal(str)


class GuiRosNode(Node):
    def __init__(self, bridge: GuiBridge) -> None:
        super().__init__("gui_node")
        self.bridge = bridge

        # Publishers used by the GUI.
        self.settings_pub = self.create_publisher(String, "/gui/settings", 10)
        self.viewpoint_pub = self.create_publisher(String, "/gui/viewpoint", 10)
        self.scan_filename_pub = self.create_publisher(String, "/gui/scan_name", 10)
        self.start_scan_pub = self.create_publisher(Bool, '/gui/start_scan', 10)
        self.pause_scan_pub = self.create_publisher(Bool, '/gui/pause_scan', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, "/gui/emergency_stop", 10)
        self.move_to_pre_scan_position_pub = self.create_publisher(Bool, '/gui/move_to_pre_scan_position', 10)
        self.rescan_section_pub = self.create_publisher(Bool, "/gui/rescan_section", 10)
        self.sample_object_pub = self.create_publisher(Bool, "/gui/sample_object", 10)

        # Subscribers used by the GUI.
        self.create_subscription(String, "/system/log", self._log_callback, 10)
        self.create_subscription(String, "/control/status", self._status_callback, 10)
        self.create_subscription(Float32, "/movement/progress", self._progress_callback, 10)
        self.create_subscription(Float32, "/scan/coverage", self._coverage_callback, 10)
        self.create_subscription(Image, "/camera/d435i/color/image_raw", self._camera_callback, qos_profile_sensor_data,)
        self.create_subscription(PointCloud2, "/camera/d435i/depth/color/points", self._pointcloud_callback, qos_profile_sensor_data,)
        self.create_subscription(Image, "/camera/d435i/aligned_depth_to_color/image_raw", self._depth_callback, qos_profile_sensor_data,)
        
        # Heartbeat used by main_control to confirm the GUI is alive.
        self.heartbeat_pub = self.create_publisher(
            Empty,
            '/heartbeat/gui',
            10,
        )

        self.heartbeat_timer = self.create_timer(
            0.5,
            self._publish_heartbeat,
        )

        self.get_logger().info("GUI node ready.")

    # ---------------------------------------------------------------------
    # Publish helpers
    # ---------------------------------------------------------------------

    def _publish_heartbeat(self) -> None:
        self.heartbeat_pub.publish(Empty())

    def publish_start_scan(self) -> None:
        msg = Bool()
        msg.data = True
        self.start_scan_pub.publish(msg)
        self.get_logger().info("Published: start_scan")

    def publish_pause_scan(self) -> None:
        msg = Bool()
        msg.data = True
        self.pause_scan_pub.publish(msg)
        self.get_logger().info("Published: pause_scan")

    def publish_move_to_pre_scan(self) -> None:
        msg = Bool()
        msg.data = True
        self.move_to_pre_scan_position_pub.publish(msg)
        self.get_logger().info("Published: move_to_pre_scan_position")

    def publish_emergency_stop(self) -> None:
        msg = Bool()
        msg.data = True
        self.emergency_stop_pub.publish(msg)
        self.get_logger().info("Published: emergency_stop")

    def publish_rescan_section(self) -> None:
        msg = Bool()
        msg.data = True
        self.rescan_section_pub.publish(msg)
        self.get_logger().info("Published: rescan_section")
    
    def publish_sample_object(self) -> None:
        msg = Bool()
        msg.data = True
        self.sample_object_pub.publish(msg)
        self.get_logger().info("Published: sample_object")

    def publish_viewpoint(self, viewpoint: str) -> None:
        msg = String()
        msg.data = viewpoint
        self.viewpoint_pub.publish(msg)
        self.get_logger().info(f"Published viewpoint: {viewpoint}")

    def publish_settings(self, settings: ScanSettings) -> None:
        msg = String()
        msg.data = json.dumps(asdict(settings))
        self.settings_pub.publish(msg)
        self.get_logger().info(f"Published settings: {msg.data}")
    
    def publish_scan_filename(self, filename: str) -> None:
        msg = String()
        msg.data = filename
        self.scan_filename_pub.publish(msg)
        self.get_logger().info(f"Published scan filename: {filename}")

    # ---------------------------------------------------------------------
    # Subscriber callbacks
    # ---------------------------------------------------------------------
    def _status_callback(self, msg: String) -> None:
        self.bridge.status_received.emit(msg.data)

    def _progress_callback(self, msg: Float32) -> None:
        self.bridge.progress_received.emit(float(msg.data))

    def _depth_callback(self, msg: Image) -> None:
        image = ros_depth_to_qimage(msg)
        if image is not None:
            self.bridge.depth_received.emit(image)

    def _pointcloud_callback(self, msg: PointCloud2) -> None:
        points = []

        try:
            for p in pc2.read_points(
                msg,
                field_names=("x", "y", "z"),
                skip_nans=True,
            ):
                points.append([p[0], p[1], p[2]])

            if not points:
                return

            points_np = np.asarray(points, dtype=np.float32)

            # Downsample so the GUI does not lag
            if len(points_np) > 8000:
                step = max(1, len(points_np) // 8000)
                points_np = points_np[::step]

            self.bridge.pointcloud_received.emit(points_np)

        except Exception as exc:
            self.get_logger().warn(f"Point cloud conversion failed: {exc}")

    def _coverage_callback(self, msg: Float32) -> None:
        self.bridge.coverage_received.emit(float(msg.data))

    def _camera_callback(self, msg: Image) -> None:
        image = ros_image_to_qimage(msg)
        if image is not None:
            self.bridge.camera_received.emit(image)

    def _log_callback(self, msg: String) -> None:
        self.bridge.log_received.emit(msg.data)

class SectionFrame(QFrame):
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("sectionFrame")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("sectionTitle")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setMinimumHeight(44)
        self.title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(self.title_label)

        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(2, 2, 2, 2)
        self.content_layout.setSpacing(10)
        layout.addLayout(self.content_layout)


class OptionSelector(QWidget):
    """Reusable row of mutually exclusive option buttons."""
    selection_changed = Signal(str)

    def __init__(self, label_text: str, options: list[str], default: str) -> None:
        super().__init__()
        self.buttons: dict[str, QPushButton] = {}
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        label = QLabel(label_text)
        label.setObjectName("rowLabel")
        layout.addWidget(label)

        layout.addStretch()

        for option in options:
            button = QPushButton(option.capitalize())
            button.setCheckable(True)
            button.setObjectName("optionButton")
            self.group.addButton(button)
            self.buttons[option] = button
            layout.addWidget(button)
            button.clicked.connect(lambda checked=False, value=option: self.selection_changed.emit(value))

        if default in self.buttons:
            self.buttons[default].setChecked(True)

    def selected(self) -> str:
        for value, button in self.buttons.items():
            if button.isChecked():
                return value
        return next(iter(self.buttons.keys()))

    def set_selected(self, value: str) -> None:
        if value in self.buttons:
            self.buttons[value].setChecked(True)


class ScannerMainWindow(QMainWindow):
    def __init__(self, ros_node: GuiRosNode, bridge: GuiBridge) -> None:
        super().__init__()
        self.ros_node = ros_node
        self.bridge = bridge
        self.settings = ScanSettings()
        self.system_state = "idle"

        self.setWindowTitle("UR3e Scanner GUI")
        self.resize(1400, 800)
        self.setMinimumSize(1000, 620)

        self._build_ui()
        self._connect_bridge_signals()
        self._apply_stylesheet()
        self._publish_settings()

    # ------------------------------------------------------------------
    # UI build
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(16)

        self.left_panel = self._build_left_panel()
        self.right_panel = self._build_right_panel()

        root.addWidget(self.left_panel, 1)
        root.addWidget(self.right_panel, 2)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        outer_layout = QVBoxLayout(panel)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(12)

        self.mode_tab_bar = QTabBar()
        self.mode_tab_bar.setObjectName("modeTabBar")
        self.mode_tab_bar.setDrawBase(False)
        self.mode_tab_bar.addTab("Normal Mode")
        self.mode_tab_bar.addTab("Advanced Mode")
        self.mode_tab_bar.setCurrentIndex(0)
        self.mode_tab_bar.currentChanged.connect(self._switch_left_mode)
        outer_layout.addWidget(self.mode_tab_bar)

        self.left_mode_stack = QStackedLayout()
        outer_layout.addLayout(self.left_mode_stack, 1)

        self.normal_mode_panel = self._build_normal_mode_panel()
        self.advanced_mode_panel = self._build_advanced_mode_panel()

        self.left_mode_stack.addWidget(self.normal_mode_panel)
        self.left_mode_stack.addWidget(self.advanced_mode_panel)
        self.left_mode_stack.setCurrentIndex(0)

        return panel
    
    def _build_normal_mode_panel(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Status + buttons section.
        self.status_label = QLabel("IDLE")
        self.status_label.setObjectName("sectionTitle")  # reuse styling
        self.status_label.setAlignment(Qt.AlignCenter)

        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.estop_button = QPushButton("Emergency Stop")
        self.move_to_pre_scan_button = QPushButton("Move to Home")

        self.start_button.clicked.connect(self._send_start_scan)
        self.pause_button.clicked.connect(self._send_pause_scan)
        self.estop_button.clicked.connect(self._send_emergency_stop)
        self.estop_button.setObjectName("dangerButton")
        self.move_to_pre_scan_button.clicked.connect(self._send_move_to_pre_scan)

        status_grid = QGridLayout()
        status_grid.setContentsMargins(0, 0, 0, 0)
        status_grid.setSpacing(10)

        status_grid.addWidget(self.status_label, 0, 0, 1, 2)
        status_grid.addWidget(self.start_button, 1, 0)
        status_grid.addWidget(self.pause_button, 1, 1)
        status_grid.addWidget(self.move_to_pre_scan_button, 2, 0, 1, 2)
        status_grid.addWidget(self.estop_button, 3, 0, 1, 2)

        layout.addLayout(status_grid, 0, 0, 1, 2)

        # Settings section.
        settings_section = SectionFrame("Parameters")
        self.resolution_selector = OptionSelector("Resolution", ["high", "medium", "low"], "high")
        self.speed_selector = OptionSelector("Speed", ["high", "medium", "low"], "medium")

        self.resolution_selector.selection_changed.connect(self._settings_changed)
        self.speed_selector.selection_changed.connect(self._settings_changed)

        settings_section.content_layout.addWidget(self.resolution_selector)
        settings_section.content_layout.addWidget(self.speed_selector)
        layout.addWidget(settings_section, 1, 0, 1, 1)

        # Scan metrics section.
        metrics_section = SectionFrame("Metrics")

        metrics_grid = QGridLayout()
        metrics_grid.setContentsMargins(0, 0, 0, 0)
        metrics_grid.setHorizontalSpacing(8)
        metrics_grid.setVerticalSpacing(8)

        coverage_label = QLabel("Coverage")
        coverage_label.setObjectName("metricName")

        self.coverage_value = QLabel("0 %")
        self.coverage_value.setObjectName("metricValue")

        file_name_title = QLabel("File Name")
        file_name_title.setObjectName("metricName")

        self.scan_name_input = QLineEdit()
        self.scan_name_input.setPlaceholderText("Enter file name")
        self.scan_name_input.setMaxLength(64)
        self.scan_name_input.textChanged.connect(self._enforce_ascii_filename)

        metrics_grid.addWidget(coverage_label, 0, 0)
        metrics_grid.addWidget(self.coverage_value, 0, 1)

        metrics_grid.addWidget(file_name_title, 1, 0, 1, 2)

        metrics_grid.addWidget(self.scan_name_input, 2, 0, 1, 2)

        metrics_section.content_layout.addLayout(metrics_grid)

        layout.addWidget(metrics_section, 1, 1, 1, 1)

        # Warning box.
        warning_section = SectionFrame("System Log")
        self.warning_box = QTextEdit()
        self.warning_box.setReadOnly(True)
        self.warning_box.setMinimumHeight(120)
        warning_section.content_layout.addWidget(self.warning_box)
        layout.addWidget(warning_section, 2, 0, 1, 2)

        layout.setRowStretch(0, 0)
        layout.setRowStretch(1, 0)
        layout.setRowStretch(2, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

        return panel
    
    def _build_advanced_mode_panel(self) -> QWidget:
        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Viewpoint control section.
        viewpoint_section = SectionFrame("Manual Control")

        self.top_view_button = QPushButton("Top")
        self.front_view_button = QPushButton("Front")
        self.back_view_button = QPushButton("Back")
        self.left_view_button = QPushButton("Left")
        self.right_view_button = QPushButton("Right")
        self.advanced_estop_button = QPushButton("Emergency Stop")
        self.advanced_rescan_button = QPushButton("Rescan Current Section")
        self.advanced_sample_button = QPushButton("Sample Object")

        self.advanced_estop_button.setObjectName("dangerButton")

        viewpoint_buttons = [
            (self.top_view_button, "top"),
            (self.front_view_button, "front"),
            (self.back_view_button, "back"),
            (self.left_view_button, "left"),
            (self.right_view_button, "right"),
        ]

        for button, viewpoint in viewpoint_buttons:
            button.setObjectName("viewpointButton")
            button.clicked.connect(
                lambda checked=False, selected_viewpoint=viewpoint: self._send_viewpoint(selected_viewpoint)
            )

        self.advanced_estop_button.clicked.connect(self._send_emergency_stop)
        self.advanced_rescan_button.clicked.connect(self._send_rescan_section)
        self.advanced_sample_button.clicked.connect(self._send_sample_object)

        viewpoint_grid = QGridLayout()
        viewpoint_grid.setContentsMargins(0, 0, 0, 0)
        viewpoint_grid.setHorizontalSpacing(10)
        viewpoint_grid.setVerticalSpacing(10)

        # Direction layout:
        #          Front
        #   Left    Top    Right
        #          Back
        viewpoint_grid.addWidget(self.front_view_button, 0, 1)
        viewpoint_grid.addWidget(self.left_view_button, 1, 0)
        viewpoint_grid.addWidget(self.top_view_button, 1, 1)
        viewpoint_grid.addWidget(self.right_view_button, 1, 2)
        viewpoint_grid.addWidget(self.back_view_button, 2, 1)
        viewpoint_grid.addWidget(self.advanced_estop_button, 3, 0, 1, 3)
        viewpoint_grid.addWidget(self.advanced_rescan_button, 4, 0, 1, 3)
        viewpoint_grid.addWidget(self.advanced_sample_button, 5, 0, 1, 3)
        viewpoint_section.content_layout.addLayout(viewpoint_grid)
        layout.addWidget(viewpoint_section, 0, 0, 1, 2)

        warning_section = SectionFrame("Error Log")
        self.advanced_warning_box = QTextEdit()
        self.advanced_warning_box.setReadOnly(True)
        self.advanced_warning_box.setMinimumHeight(120)
        warning_section.content_layout.addWidget(self.advanced_warning_box)

        layout.addWidget(warning_section, 1, 0, 1, 2)

        layout.setRowStretch(0, 0)
        layout.setRowStretch(1, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

        return panel


    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Top bar.
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(12)

        progress_container = QFrame()
        progress_container.setMinimumHeight(40)
        progress_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        progress_layout = QGridLayout(progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(0)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        overlay_widget = QWidget()
        overlay_widget.setAttribute(Qt.WA_TransparentForMouseEvents)
        overlay_widget.setStyleSheet("background: transparent;")

        overlay_layout = QHBoxLayout(overlay_widget)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        overlay_layout.setAlignment(Qt.AlignCenter)

        self.progress_overlay = QLabel("Progress: 0%")
        self.progress_overlay.setObjectName("progressOverlay")
        self.progress_overlay.setAlignment(Qt.AlignCenter)
        self.progress_overlay.setStyleSheet("background: transparent;")

        overlay_layout.addWidget(self.progress_overlay)

        progress_layout.addWidget(self.progress_bar, 0, 0)
        progress_layout.addWidget(overlay_widget, 0, 0)

        top_bar.addWidget(progress_container, 1)
        layout.addLayout(top_bar)

        # Tab controls.
        self.tab_bar = QTabBar()
        self.tab_bar.setDrawBase(False)
        self.tab_bar.addTab("Simulation")
        self.tab_bar.addTab("Camera")
        self.tab_bar.addTab("Model")
        self.tab_bar.currentChanged.connect(self._switch_view)
        layout.addWidget(self.tab_bar)

        # Stacked views.
        view_container = QFrame()
        view_container.setObjectName("viewFrame")
        self.view_stack = QStackedLayout(view_container)
        self.view_stack.setContentsMargins(0, 0, 0, 0)

        self.simulation_view = self._build_simulation_view()
        self.camera_view = self._build_camera_view()
        self.model_tab_view = self._build_model_view()

        self.view_stack.addWidget(self.simulation_view)
        self.view_stack.addWidget(self.camera_view)
        self.view_stack.addWidget(self.model_tab_view)

        layout.addWidget(view_container, 1)
        return panel

    def _build_simulation_view(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.simulation_widget = DigitalTwinWidget(self.ros_node)
        self.simulation_widget.setMinimumSize(500, 350)
        layout.addWidget(self.simulation_widget, 1)

        return widget

    def _build_camera_view(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.camera_mode_tab_bar = QTabBar()
        self.camera_mode_tab_bar.setDrawBase(False)
        self.camera_mode_tab_bar.addTab("Camera Feed")
        self.camera_mode_tab_bar.addTab("Depth Overlay")
        self.camera_mode_tab_bar.addTab("Point Cloud")
        self.camera_mode_tab_bar.currentChanged.connect(self._switch_camera_view)
        layout.addWidget(self.camera_mode_tab_bar)

        self.camera_stack_container = QFrame()
        self.camera_view_stack = QStackedLayout(self.camera_stack_container)
        self.camera_view_stack.setContentsMargins(0, 0, 0, 0)

        # RGB feed
        rgb_widget = QWidget()
        rgb_layout = QVBoxLayout(rgb_widget)
        rgb_layout.setContentsMargins(0, 0, 0, 0)

        self.camera_label = QLabel("RGB camera feed waiting...")
        self.camera_label.setObjectName("cameraLabel")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(500, 350)
        self.camera_label.setScaledContents(False)

        rgb_layout.addWidget(self.camera_label, 1)

        # Depth overlay feed
        overlay_widget = QWidget()
        overlay_layout = QVBoxLayout(overlay_widget)
        overlay_layout.setContentsMargins(0, 0, 0, 0)

        self.overlay_label = QLabel("Depth overlay feed waiting...")
        self.overlay_label.setObjectName("cameraLabel")
        self.overlay_label.setAlignment(Qt.AlignCenter)
        self.overlay_label.setMinimumSize(500, 350)
        self.overlay_label.setScaledContents(False)

        overlay_layout.addWidget(self.overlay_label, 1)

        # 3D point cloud feed
        pointcloud_widget = QWidget()
        pointcloud_layout = QVBoxLayout(pointcloud_widget)
        pointcloud_layout.setContentsMargins(0, 0, 0, 0)

        self.pointcloud_view = gl.GLViewWidget()

        self.pointcloud_view.setCameraPosition(
            distance=1.0,
            elevation=-90,
            azimuth=-90
        )

        self.pointcloud_grid = gl.GLGridItem()
        self.pointcloud_grid.scale(0.25, 0.25, 0.25)
        self.pointcloud_view.addItem(self.pointcloud_grid)

        self.pointcloud_item = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            size=2,
            pxMode=True,
        )
        self.pointcloud_view.addItem(self.pointcloud_item)

        pointcloud_layout.addWidget(self.pointcloud_view, 1)

        self.camera_view_stack.addWidget(rgb_widget)
        self.camera_view_stack.addWidget(overlay_widget)
        self.camera_view_stack.addWidget(pointcloud_widget)

        layout.addWidget(self.camera_stack_container, 1)

        self.latest_rgb_image = None
        self.latest_depth_image = None

        return widget

    def _build_model_view(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # Hardcoded to the source tree — same approach as scan_node.py.
        # __file__ in a ROS2 node points to the install copy, not the source.
        self.scans_folder = os.path.join(
            os.path.expanduser("~"), "ros2_ws", "src", "rs2", "reconstructed_scans"
        )
        self._model_paths: dict[str, str] = {}

        selector_row = QHBoxLayout()
        self.model_selector = QComboBox()
        self.model_selector.setObjectName("modelSelector")
        self.model_selector.addItem("Select Model")
        self.model_selector.currentIndexChanged.connect(self._model_selected)
        selector_row.addWidget(self.model_selector, 1)

        self.model_selector.showPopup = self._refresh_model_selector_popup

        self.download_btn = QPushButton("Download")
        self.download_btn.setFixedWidth(150)
        self.download_btn.clicked.connect(self._download_selected_model)
        selector_row.addWidget(self.download_btn)

        layout.addLayout(selector_row)

        self.model_view = gl.GLViewWidget()
        self.model_view.setCameraPosition(
            distance=1.0,
            elevation=25,
            azimuth=45,
        )

        self.model_grid = gl.GLGridItem()
        self.model_grid.scale(0.1, 0.1, 0.1)
        self.model_view.addItem(self.model_grid)

        layout.addWidget(self.model_view, 1)

        self.model_mesh_item = None
        self._refresh_model_list()

        return widget
    
    def _refresh_model_selector_popup(self) -> None:
        current_text = self.model_selector.currentText()

        self._refresh_model_list()

        if current_text in self._model_paths:
            self.model_selector.setCurrentText(current_text)

        QComboBox.showPopup(self.model_selector)

    def _switch_left_mode(self, index: int) -> None:
        self.left_mode_stack.setCurrentIndex(index)

    def _enforce_ascii_filename(self, text: str) -> None:
        ascii_text = ''.join(c for c in text if ord(c) < 128)

        if ascii_text != text:
            cursor_pos = self.scan_name_input.cursorPosition()

            self.scan_name_input.blockSignals(True)
            self.scan_name_input.setText(ascii_text)
            self.scan_name_input.setCursorPosition(max(0, cursor_pos - 1))
            self.scan_name_input.blockSignals(False)

    def _refresh_model_list(self) -> None:
        self.model_selector.blockSignals(True)
        self.model_selector.clear()
        self.model_selector.addItem("Select Model")
        self._model_paths.clear()

        if not os.path.isdir(self.scans_folder):
            self._append_warning(f"[WARNING] Reconstructed scans folder not found: {self.scans_folder}")
            self.model_selector.blockSignals(False)
            return

        for scan_name in sorted(os.listdir(self.scans_folder)):
            scan_dir = os.path.join(self.scans_folder, scan_name)
            if not os.path.isdir(scan_dir):
                continue
            obj_path = os.path.join(scan_dir, f"{scan_name}.obj")
            if os.path.exists(obj_path):
                self._model_paths[scan_name] = obj_path
                self.model_selector.addItem(scan_name)

        self.model_selector.blockSignals(False)

    def _model_selected(self, index: int) -> None:
        if index <= 0:
            return
        scan_name = self.model_selector.currentText()
        mesh_path = self._model_paths.get(scan_name)
        if mesh_path:
            self._load_model_mesh(mesh_path)

    def _load_model_mesh(self, mesh_path: str) -> None:
        try:
            if self.model_mesh_item is not None:
                self.model_view.removeItem(self.model_mesh_item)
                self.model_mesh_item = None

            mesh = trimesh.load(mesh_path, force="mesh")

            vertices = np.asarray(mesh.vertices)

            # Centre and scale the mesh so different scan sizes fit the viewer.
            vertices = vertices - vertices.mean(axis=0)
            max_dim = np.max(np.linalg.norm(vertices, axis=1))

            if max_dim > 0:
                vertices = vertices * (0.2 / max_dim)

            faces = np.asarray(mesh.faces)
            vertices = vertices - vertices.mean(axis=0)

            mesh_data = gl.MeshData(
                vertexes=vertices,
                faces=faces,
            )

            self.model_mesh_item = gl.GLMeshItem(
                meshdata=mesh_data,
                smooth=True,
                drawFaces=True,
                drawEdges=True,
                edgeColor=(0.2, 0.2, 0.2, 1),
            )

            self.model_view.addItem(self.model_mesh_item)

        except Exception as exc:
            self._append_warning(f"[ERROR] Model viewer failed to load mesh: {exc}")

    def _download_selected_model(self) -> None:
        scan_name = self.model_selector.currentText()
        src = self._model_paths.get(scan_name)
        if not src:
            self._append_warning("[WARNING] No model selected to download.")
            return
        dest_dir = Path.home() / "Downloads"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / Path(src).name
        try:
            shutil.copy2(src, dest)
            self._append_warning(f"[INFO] Downloaded {Path(src).name} to {dest_dir}")
        except Exception as exc:
            self._append_warning(f"[ERROR] Download failed: {exc}")

    # ------------------------------------------------------------------
    # Signal hookups
    # ------------------------------------------------------------------
    def _connect_bridge_signals(self) -> None:
        self.bridge.status_received.connect(self._update_status)
        self.bridge.progress_received.connect(self._update_progress)
        self.bridge.coverage_received.connect(self._update_coverage)
        self.bridge.camera_received.connect(self._update_camera)
        self.bridge.depth_received.connect(self._update_depth_overlay)
        self.bridge.pointcloud_received.connect(self._update_pointcloud)
        self.bridge.log_received.connect(self._append_warning)

    # ------------------------------------------------------------------
    # UI interactions
    # ------------------------------------------------------------------

    def _send_viewpoint(self, viewpoint: str) -> None:
        allowed_states = {"idle", "pre_scan_position", "auto: paused"}
        if self.system_state not in allowed_states:
            self._append_warning(
                "[WARNING] Manual move rejected: system must be Idle, Pre-Scan Position, or Paused."
            )
            return
        self.ros_node.publish_viewpoint(viewpoint)

    def _send_start_scan(self) -> None:
        self._publish_scan_filename()
        self.ros_node.publish_start_scan()

    def _send_pause_scan(self) -> None:
        self.ros_node.publish_pause_scan()

    def _send_rescan_section(self) -> None:
        if self.system_state != "idle":
            self._append_warning("[WARNING] Advanced command rejected: system must be Idle.")
            return
        self.ros_node.publish_rescan_section()

    def _send_sample_object(self) -> None:
        if self.system_state != "idle":
            self._append_warning("[WARNING] Advanced command rejected: system must be Idle.")
            return
        self.ros_node.publish_sample_object()

    def _send_emergency_stop(self) -> None:
        self.ros_node.publish_emergency_stop()

    def _send_move_to_pre_scan(self) -> None:
        self.ros_node.publish_move_to_pre_scan()


    def _settings_changed(self, *_args) -> None:
        """Publish scan parameter changes when the operator updates the GUI."""
        self.settings.resolution = self.resolution_selector.selected()
        self.settings.speed = self.speed_selector.selected()
        self._publish_settings()

    def _publish_settings(self) -> None:
        self.ros_node.publish_settings(self.settings)

    def _switch_view(self, index: int) -> None:
        self.view_stack.setCurrentIndex(index)

    def _switch_camera_view(self, index: int) -> None:
        self.camera_view_stack.setCurrentIndex(index)

    # ------------------------------------------------------------------
    # UI updates from ROS2
    # ------------------------------------------------------------------
    def _update_status(self, status: str) -> None:
        previous_state = self.system_state
        self.system_state = status.strip().lower()
        if hasattr(self, "status_label"):
            self.status_label.setText(status)
        if status.strip() == "AUTO: RECONSTRUCTING_SCAN" and previous_state != self.system_state:
            self.scan_name_input.clear()

    def _publish_scan_filename(self) -> None:
        filename = self.scan_name_input.text().strip()
        self.ros_node.publish_scan_filename(filename)

    def _append_warning(self, warning: str) -> None:
        if hasattr(self, "warning_box"):
            self.warning_box.append(warning)
        if hasattr(self, "advanced_warning_box"):
            self.advanced_warning_box.append(warning)

    def _update_progress(self, value: float) -> None:
        bounded = max(0, min(100, int(round(value))))
        self.progress_bar.setValue(bounded)
        self.progress_overlay.setText(f"Progress: {bounded}%")

    def _update_coverage(self, value: float) -> None:
        bounded = max(0, min(100, int(round(value))))
        self.coverage_value.setText(f"{bounded} %")

    def _update_camera(self, image: QImage) -> None:
        if image.isNull():
            return

        self.latest_rgb_image = image.copy()

        scaled = image.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.camera_label.setPixmap(QPixmap.fromImage(scaled))

        self._refresh_depth_overlay()

    def _update_depth_overlay(self, image: QImage) -> None:
        if image.isNull():
            return

        self.latest_depth_image = image.copy()
        self._refresh_depth_overlay()


    def _refresh_depth_overlay(self) -> None:
        if self.latest_rgb_image is None or self.latest_depth_image is None:
            return

        rgb = self.latest_rgb_image.copy()
        depth = self.latest_depth_image.scaled(
            rgb.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        painter = QPainter(rgb)
        painter.setOpacity(0.45)
        painter.drawImage(0, 0, depth)
        painter.end()

        scaled = rgb.scaled(
            self.overlay_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        self.overlay_label.setPixmap(QPixmap.fromImage(scaled))

    def _update_pointcloud(self, points: np.ndarray) -> None:
        if points is None or len(points) == 0:
            return

        self.pointcloud_item.setData(pos=points)


    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
    
        current_camera = self.camera_label.pixmap()
        if current_camera is not None and not current_camera.isNull():
            self.camera_label.setPixmap(
                current_camera.scaled(
                    self.camera_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

    
    def _apply_stylesheet(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #e8e8e8;
                color: #202020;
                font-family: Arial;
                font-size: 15px;
            }
            QMenuBar, QMenu {
                background: #e8e8e8;
                color: #202020;
            }
            #sectionFrame {
                background: transparent;
                border: none;
            }
            #sectionTitle {
                background: #1f1f22;
                color: #f2f2f2;
                border-radius: 12px;
                padding: 10px;
                font-size: 16px;
                font-weight: 600;
            }
            #statusValue {
                background: #ffffff;
                border: 1px solid #cfcfcf;
                border-radius: 10px;
                padding: 10px;
                font-size: 18px;
                font-weight: 600;
            }
            QPushButton {
                background: #1f1f22;
                color: #f2f2f2;
                border: none;
                border-radius: 12px;
                padding: 12px;
                font-size: 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #2c2c30;
            }
            QPushButton:pressed {
                background: #17171a;
            }
            QPushButton:checked {
                background: #2f2f34;
            }
            #optionButton {
                background: transparent;
                color: #202020;
                border: 2px solid transparent;
                border-radius: 12px;
                padding: 4px 8px;
                font-size: 14px;
                font-weight: 500;
            }
            #optionButton:hover {
                background: transparent;
                border: 2px solid #8a8a8a;
            }
            #optionButton:pressed {
                background: transparent;
                border: 2px solid #4a4a4a;
            }
            #optionButton:checked {
                background: #1f1f22;
                color: #f2f2f2;
                border: 2px solid #1f1f22;
            }
            #dangerButton {
                background: #7b1f1f;
            }
            #viewpointButton {
                min-height: 58px;
            }
            #dangerButton:hover {
                background: #942626;
            }
            #rowLabel, #metricName, #topLabel {
                font-size: 15px;
                font-weight: 500;
            }
            #metricValue, #topValue {
                font-size: 15px;
                font-weight: 600;
            }
            #progressOverlay {
                color: #f2f2f2;
                font-size: 15px;
                font-weight: 600;
                background: transparent;
            }
            QProgressBar {
                background: #7a7a7a;
                border: none;
                border-radius: 10px;
                min-height: 18px;
            }
            QProgressBar::chunk {
                background: #1f1f22;
                border-radius: 10px;
            }
            QTabBar {
                border: none;
            }

            QTabBar::tab-bar {
                alignment: left;
            }

            QTabWidget::pane {
                border: none;
            }
            QTabBar::tab {
                background: transparent;
                color: #202020;
                border: 2px solid transparent;
                padding: 10px 18px;
                margin-right: 6px;
                border-radius: 12px;
                font-size: 15px;
                font-weight: 500;
            }
            QTabBar::tab:hover {
                background: transparent;
                border: 2px solid #8a8a8a;
            }

            QTabBar::tab:selected {
                background: #1f1f22;
                color: #f2f2f2;
                border: 2px solid #1f1f22;
            }
            #viewFrame {
                background: #1f1f22;
                border-radius: 24px;
            }
            #viewTitle {
                color: #f2f2f2;
                font-size: 18px;
                font-weight: 600;
            }
            #bigPlaceholder {
                color: #dddddd;
                font-size: 17px;
                border: 1px dashed #4e4e4e;
                border-radius: 18px;
            }
            #cameraLabel {
                color: #f2f2f2;
                background: #151518;
                border: 1px solid #404040;
                border-radius: 18px;
                font-size: 18px;
                font-weight: 600;
            }
            QTextEdit {
                background: #ffffff;
                border: 1px solid #d0d0d0;
                border-radius: 12px;
                padding: 8px;
            }
            #modeTabBar::tab {
                background: transparent;
                color: #202020;
                border: 2px solid transparent;
                border-radius: 12px;
                padding: 6px 12px;
                font-size: 15px;
                font-weight: 500;
                margin-right: 6px;
            }

            #modeTabBar::tab:hover {
                background: transparent;
                border: 2px solid #8a8a8a;
            }

            #modeTabBar::tab:selected {
                background: #1f1f22;
                color: #f2f2f2;
                border: 2px solid #1f1f22;
            }

            """
        )


def ros_image_to_qimage(msg: Image) -> Optional[QImage]:
    """
    Convert a ROS2 sensor_msgs/Image into a QImage.

    Supported encodings in this prototype:
    - rgb8
    - bgr8
    - rgba8
    - bgra8
    - mono8
    """
    width = msg.width
    height = msg.height
    encoding = msg.encoding.lower()
    data = bytes(msg.data)

    if width <= 0 or height <= 0:
        return None

    if encoding == "rgb8":
        image = QImage(data, width, height, msg.step, QImage.Format_RGB888)
        return image.copy()
    if encoding == "bgr8":
        image = QImage(data, width, height, msg.step, QImage.Format_BGR888)
        return image.copy()
    if encoding == "rgba8":
        image = QImage(data, width, height, msg.step, QImage.Format_RGBA8888)
        return image.copy()
    if encoding == "bgra8":
        image = QImage(data, width, height, msg.step, QImage.Format_ARGB32)
        return image.copy()
    if encoding == "mono8":
        image = QImage(data, width, height, msg.step, QImage.Format_Grayscale8)
        return image.copy()

    return None

def ros_depth_to_qimage(msg: Image) -> Optional[QImage]:
    width = msg.width
    height = msg.height
    encoding = msg.encoding.lower()

    if width <= 0 or height <= 0:
        return None

    if encoding == "16uc1":
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(height, width)
        valid = depth[depth > 0]

        if valid.size == 0:
            return None

        min_depth = np.percentile(valid, 2)
        max_depth = np.percentile(valid, 98)

        depth_clipped = np.clip(depth, min_depth, max_depth)
        depth_norm = ((depth_clipped - min_depth) / (max_depth - min_depth + 1e-6) * 255).astype(np.uint8)

    elif encoding == "32fc1":
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape(height, width)
        valid = depth[np.isfinite(depth) & (depth > 0)]

        if valid.size == 0:
            return None

        min_depth = np.percentile(valid, 2)
        max_depth = np.percentile(valid, 98)

        depth_clipped = np.clip(depth, min_depth, max_depth)
        depth_norm = ((depth_clipped - min_depth) / (max_depth - min_depth + 1e-6) * 255).astype(np.uint8)

    else:
        return None

    colour_depth = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    colour_depth = cv2.cvtColor(colour_depth, cv2.COLOR_BGR2RGB)

    qimage = QImage(
        colour_depth.data,
        width,
        height,
        3 * width,
        QImage.Format_RGB888,
    )

    return qimage.copy()


def main(args=None) -> None:
    rclpy.init(args=args)

    app = QApplication(sys.argv)
    app.setApplicationName("UR3e Scanner GUI")

    bridge = GuiBridge()
    node = GuiRosNode(bridge)
    window = ScannerMainWindow(node, bridge)
    window.show()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    executor_thread = threading.Thread(
        target=executor.spin,
        daemon=True,
    )
    executor_thread.start()

    def request_shutdown(*_args):
        app.quit()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    try:
        app.exec()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.remove_node(node)
        except Exception:
            pass

        try:
            executor.shutdown(timeout_sec=1.0)
        except Exception:
            pass

        try:
            node.destroy_node()
        except Exception:
            pass

        try:
            rclpy.shutdown()
        except Exception:
            pass

        if executor_thread.is_alive():
            executor_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()