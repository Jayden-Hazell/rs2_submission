#!/usr/bin/env python3
# 41069 Robotics Studio 2
# Authors: Jayden Hazell (24953364), Lachlan Partridge (24964557), Tomas Klimes (24773804)
# Captures synchronised RGB-D frames from the RealSense camera and launches 3D reconstruction.
"""
Scan node for the UR3e scanning system.

Manages a multi-step boot sequence that verifies camera topics and TF availability,
then captures synchronised RGB-D frames on command. Each capture bundle (colour image,
depth array, camera info, TF pose, and joint state) is saved to a timestamped session
directory. After scanning, triggers the external reconstruct.py pipeline as a subprocess.
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import subprocess
from enum import Enum, auto
from datetime import datetime
from pathlib import Path

import cv2
import yaml
import numpy as np

from typing import cast

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from std_msgs.msg import Empty, String, Float32
from sensor_msgs.msg import Image, CameraInfo, JointState

from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, TransformException  # pyright: ignore[reportAttributeAccessIssue]
from message_filters import Subscriber, ApproximateTimeSynchronizer

# --------------------------------------------------
# Tunable constants
# --------------------------------------------------

BOOT_TIMEOUT_SEC             = 5.0    # Max time to wait for each boot check
CALIBRATION_TIMEOUT_SEC      = 10.0   # Max time to wait for calibration TF snapshot at boot
CAPTURE_TIMEOUT_SEC          = 10.0   # Max time to wait for a capture to succeed
CAPTURE_MAX_RETRIES          = 10     # Max capture attempts before going to ERROR
CAPTURE_RETRY_DELAY_SEC      = 0.5    # Delay between capture retries
MAX_FRAME_AGE_SEC            = 0.75   # Reject frames older than this
SYNC_SLOP_SEC                = 0.08   # ApproximateTimeSynchronizer slop
SYNC_QUEUE_SIZE              = 10     # ApproximateTimeSynchronizer queue size
JPEG_QUALITY                 = 95     # JPEG quality for color saves (unused if PNG)
SAVE_DEPTH_NPY               = True   # Save raw depth as .npy
SAVE_DEPTH_PNG               = True   # Save colourised depth as .png
VELOCITY_STABLE_THRESHOLD    = 0.01   # rad/s — below this all joints are considered settled

# --------------------------------------------------
# Enumerations
# --------------------------------------------------

class ScanState(Enum):
    BOOTING           = auto()
    IDLE              = auto()
    SCANNING          = auto()
    FINISHED_SCANNING = auto()
    RECONSTRUCTING    = auto()
    ERROR             = auto()


class RequestedCommand(Enum):
    NONE        = auto()
    SCAN        = auto()
    RESET       = auto()
    ERROR       = auto()
    RECONSTRUCT = auto()


# --------------------------------------------------
# Boot sub-states (used internally during BOOTING)
# --------------------------------------------------

class BootStep(Enum):
    CAMERA      = auto()
    SAVE_DIR    = auto()
    TF          = auto()
    CALIBRATION = auto()  # attempts calibration TF snapshot; non-blocking on failure
    DONE        = auto()


class ScanNode(Node):
    def __init__(self) -> None:
        super().__init__('scan_node')

        # ------------------------------------------
        # Parameters
        # ------------------------------------------

        self.declare_parameter('auto_reconstruct',   True)
        self.declare_parameter('use_fake_hardware',  False)
        self.declare_parameter('fixed_frame',        'base_link')
        self.declare_parameter('camera_frame',       'd435i_color_optical_frame')
        self.declare_parameter('robot_ee_frame',     'tool0')
        self.declare_parameter('intermediate_frame', 'd435i_link')
        self.declare_parameter('color_topic',        '/camera/d435i/color/image_raw')
        self.declare_parameter('depth_topic',        '/camera/d435i/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic',  '/camera/d435i/color/camera_info')
        self.declare_parameter(
            'save_root',
            os.path.join(os.path.expanduser('~'), 'ros2_ws', 'src', 'rs2', 'raw_scans')
        )
        # Reconstruction parameters — passed directly to reconstruct.py
        self.declare_parameter('recon_voxel_size',    0.003)
        self.declare_parameter('recon_icp',           True)
        self.declare_parameter('recon_icp_threshold', 0.005)

        self.auto_reconstruct    = bool(self.get_parameter('auto_reconstruct').value)
        self.use_fake_hardware   = bool(self.get_parameter('use_fake_hardware').value)
        self.fixed_frame         = cast(str, self.get_parameter('fixed_frame').value)
        self.camera_frame        = cast(str, self.get_parameter('camera_frame').value)
        self.robot_ee_frame      = cast(str, self.get_parameter('robot_ee_frame').value)
        self.intermediate_frame  = cast(str, self.get_parameter('intermediate_frame').value)
        self.color_topic         = cast(str, self.get_parameter('color_topic').value)
        self.depth_topic         = cast(str, self.get_parameter('depth_topic').value)
        self.camera_info_topic   = cast(str, self.get_parameter('camera_info_topic').value)
        self.save_root           = cast(str, self.get_parameter('save_root').value)
        self.recon_voxel_size    = float(self.get_parameter('recon_voxel_size').value)
        self.recon_icp           = bool(self.get_parameter('recon_icp').value)
        self.recon_icp_threshold = float(self.get_parameter('recon_icp_threshold').value)

        # ------------------------------------------
        # State machine variables
        # ------------------------------------------

        self.state:             ScanState        = ScanState.BOOTING
        self.requested_command: RequestedCommand  = RequestedCommand.NONE

        # Boot tracking
        self.boot_step:              BootStep    = BootStep.CAMERA
        self.boot_step_start:        float       = time.monotonic()
        self.calibration_step_start: float | None = None

        # Session / capture tracking
        self.session_dir:       str | None = None
        self.session_id:        str | None = None
        self._last_session_dir: str | None = None  # saved by RESET so RECONSTRUCT can find it
        self.pending_scan_name: str | None = None
        self.capture_index: int = 0
        self.session_valid_fractions: list[float] = []
        self._reconstruction_process: subprocess.Popen | None = None

        # Capture attempt tracking
        self.capture_in_progress:   bool         = False
        self.capture_attempt_count: int          = 0
        self.capture_start_time:    float | None = None
        self.last_retry_time:       float | None = None

        # Latest synced frame from camera (written by _synced_callback, read by _attempt_capture)
        # Single-threaded executor — no Lock needed; snapshot at read site for clarity.
        self.latest: dict | None = None

        # Joint state (written by _joint_callback, read by _attempt_capture)
        self.current_joint_names:      list[str]        = []
        self.current_joint_positions:  dict[str, float] = {}
        self.current_joint_velocities: dict[str, float] = {}

        # Calibration TF snapshot written once during CALIBRATION boot step
        self.calibration_snapshot:          dict | None  = None
        self.calibration_unavailable_reason: str | None  = None

        # ------------------------------------------
        # CV bridge & TF
        # ------------------------------------------

        self.bridge      = CvBridge()
        self.tf_buffer   = Buffer(cache_time=Duration(seconds=10))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ------------------------------------------
        # Publishers
        # ------------------------------------------

        self.heartbeat_pub = self.create_publisher(Empty,   '/heartbeat/scan',  10)
        self.status_pub    = self.create_publisher(String,  '/scan/status',     10)
        self.error_pub     = self.create_publisher(String,  '/system/log',      10)
        self.coverage_pub  = self.create_publisher(Float32, '/scan/coverage',   10)

        # ------------------------------------------
        # Subscribers
        # ------------------------------------------

        self.scan_command_sub = self.create_subscription(
            String,
            '/control/scan_command',
            self._scan_command_callback,
            10,
        )

        self.scan_name_sub = self.create_subscription(
            String,
            '/gui/scan_name',
            self._scan_name_callback,
            10,
        )

        self.joint_states_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_callback,
            10,
        )

        # Camera subscribers (always created; data only flows on real hardware)
        self.color_sub = Subscriber(self, Image,      self.color_topic)
        self.depth_sub = Subscriber(self, Image,      self.depth_topic)
        self.info_sub  = Subscriber(self, CameraInfo, self.camera_info_topic)

        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub],
            queue_size=SYNC_QUEUE_SIZE,
            slop=SYNC_SLOP_SEC,
        )
        self.sync.registerCallback(self._synced_callback)

        # ------------------------------------------
        # Timers
        # ------------------------------------------

        self.heartbeat_timer          = self.create_timer(0.5, self._publish_heartbeat)
        self.state_machine_timer      = self.create_timer(0.1, self.state_machine_tick)
        self._startup_status_count    = 0
        self._startup_status_limit    = 20  # 20 x 0.5s = 10 seconds
        self.startup_status_timer     = self.create_timer(0.5, self._publish_startup_status)

        # ------------------------------------------
        # Logging
        # ------------------------------------------

        self.get_logger().info(f'Scan node started. use_fake_hardware={self.use_fake_hardware}')
        self.get_logger().info(f'Initial state: {self.state.name}')
        self._publish_status()

    # ==========================================================
    # State Management
    # ==========================================================

    def set_state(self, new_state: ScanState) -> None:
        if self.state == new_state:
            return
        self.get_logger().info(f'State change: {self.state.name} -> {new_state.name}')
        if new_state == ScanState.FINISHED_SCANNING:
            self._try_rename_session()
        self.state = new_state
        self._publish_status()

    def _enter_error(self, reason: str) -> None:
        self.get_logger().error(f'Scan node ERROR: {reason}')
        msg = String()
        msg.data = f'[ERROR] {reason}'
        self.error_pub.publish(msg)
        self.set_state(ScanState.ERROR)

    # ==========================================================
    # State Machine Tick
    # ==========================================================

    def state_machine_tick(self) -> None:    
        if self.state != ScanState.BOOTING:
            if self.requested_command == RequestedCommand.RESET:
                self.requested_command = RequestedCommand.NONE
                self._reset_session()
                self.set_state(ScanState.IDLE)

            if self.requested_command == RequestedCommand.ERROR:
                self.requested_command = RequestedCommand.NONE
                self._enter_error('Received ERROR command.')
                self.set_state(ScanState.ERROR)        

        if self.state == ScanState.BOOTING:
            self._handle_booting()
        elif self.state == ScanState.IDLE:
            self._handle_idle()
        elif self.state == ScanState.SCANNING:
            self._handle_scanning()
        elif self.state == ScanState.FINISHED_SCANNING:
            self._handle_finished_scanning()
        elif self.state == ScanState.RECONSTRUCTING:
            self._handle_reconstructing()
        elif self.state == ScanState.ERROR:
            self._handle_error()

    # ==========================================================
    # State Handlers
    # ==========================================================

    # ----------------------------------------------------------
    # BOOTING
    # ----------------------------------------------------------

    def _handle_booting(self) -> None:
        # Fake hardware: skip all checks and go straight to IDLE
        if self.use_fake_hardware:
            self.get_logger().info('Fake hardware: skipping boot checks.')
            self.set_state(ScanState.IDLE)
            return

        now = time.monotonic()

        # ---- Step 1: Camera topic publishing ----
        if self.boot_step == BootStep.CAMERA:
            if self.latest is not None:
                self.get_logger().info('Boot check passed: camera is publishing.')
                self.boot_step = BootStep.SAVE_DIR
                self.boot_step_start = now
                return

            elapsed = now - self.boot_step_start
            if elapsed >= BOOT_TIMEOUT_SEC:
                self._enter_error(
                    f'Boot failed: camera not publishing on {self.color_topic} '
                    f'after {BOOT_TIMEOUT_SEC}s.'
                )
            return

        # ---- Step 2: save_root writable ----
        if self.boot_step == BootStep.SAVE_DIR:
            try:
                os.makedirs(self.save_root, exist_ok=True)
                if not os.access(self.save_root, os.W_OK):
                    raise PermissionError(f'{self.save_root} is not writable.')
                self.get_logger().info(f'Boot check passed: save_root writable ({self.save_root}).')
                self.boot_step = BootStep.TF
                self.boot_step_start = now
            except Exception as e:
                self._enter_error(f'Boot failed: save_root check error: {e}')
            return

        # ---- Step 3: Composed TF available ----
        if self.boot_step == BootStep.TF:
            try:
                self.tf_buffer.lookup_transform(
                    self.fixed_frame,
                    self.camera_frame,
                    Time(),
                    Duration(nanoseconds=int(0.1 * 1e9)),
                )
                self.get_logger().info(
                    f'Boot check passed: TF {self.fixed_frame} -> {self.camera_frame} available.'
                )
                self.boot_step = BootStep.CALIBRATION
                self.calibration_step_start = now
            except TransformException:
                elapsed = now - self.boot_step_start
                if elapsed >= BOOT_TIMEOUT_SEC:
                    self._enter_error(
                        f'Boot failed: TF {self.fixed_frame} -> {self.camera_frame} '
                        f'not available after {BOOT_TIMEOUT_SEC}s.'
                    )
            return

        # ---- Step 4: Calibration chain snapshot (non-blocking — warn and continue) ----
        if self.boot_step == BootStep.CALIBRATION:
            assert self.calibration_step_start is not None
            elapsed = now - self.calibration_step_start

            try:
                tf_ee_to_intermediate = self.tf_buffer.lookup_transform(
                    self.robot_ee_frame,
                    self.intermediate_frame,
                    Time(),
                    Duration(nanoseconds=int(0.1 * 1e9)),
                )
                tf_intermediate_to_camera = self.tf_buffer.lookup_transform(
                    self.intermediate_frame,
                    self.camera_frame,
                    Time(),
                    Duration(nanoseconds=int(0.1 * 1e9)),
                )
                self.calibration_snapshot = {
                    f'{self.robot_ee_frame}_to_{self.intermediate_frame}':
                        self._transform_to_dict(tf_ee_to_intermediate),
                    f'{self.intermediate_frame}_to_{self.camera_frame}':
                        self._transform_to_dict(tf_intermediate_to_camera),
                }
                self.calibration_unavailable_reason = None
                self.get_logger().info(
                    f'Boot check passed: calibration snapshot captured '
                    f'({self.robot_ee_frame} -> {self.intermediate_frame} -> {self.camera_frame}).'
                )
            except TransformException as e:
                if elapsed < CALIBRATION_TIMEOUT_SEC:
                    return  # keep retrying

                reason = (
                    f'TransformException after {CALIBRATION_TIMEOUT_SEC}s: {e}. '
                    f'Recomposition (Mode B) will be unavailable for this session.'
                )
                self.calibration_snapshot = None
                self.calibration_unavailable_reason = reason
                self.get_logger().warn(f'Boot: calibration snapshot failed — {reason}')

            self.boot_step = BootStep.DONE
            return

        # ---- All checks passed ----
        if self.boot_step == BootStep.DONE:
            self.get_logger().info('All boot checks passed. Transitioning to IDLE.')
            self.set_state(ScanState.IDLE)

    # ----------------------------------------------------------
    # IDLE
    # ----------------------------------------------------------

    def _handle_idle(self) -> None:
        if self.requested_command == RequestedCommand.SCAN:
            self.requested_command = RequestedCommand.NONE
            self.set_state(ScanState.SCANNING)

    # ----------------------------------------------------------
    # SCANNING
    # ----------------------------------------------------------

    def _handle_scanning(self) -> None:

        # ---- Fake hardware: simulate a short delay then finish ----
        if self.use_fake_hardware:
            if not self.capture_in_progress:
                self.capture_in_progress = True
                self.capture_start_time  = time.monotonic()
                return

            if self.capture_start_time is not None and time.monotonic() - self.capture_start_time >= 1.0:
                self.capture_in_progress   = False
                self.capture_start_time    = None
                self.capture_attempt_count = 0
                self.set_state(ScanState.FINISHED_SCANNING)
            return

        # ---- Real hardware ----

        # Create session directory on first capture of this session
        if self.session_dir is None:
            self._create_session_dir()
            if self.session_dir is None:
                # _create_session_dir already called _enter_error
                return

        now = time.monotonic()

        # Start a new capture attempt
        if not self.capture_in_progress:
            self.capture_in_progress   = True
            self.capture_attempt_count = 0
            self.capture_start_time    = now
            self.last_retry_time       = now
            self.get_logger().info('Starting capture attempt 1...')
            self._attempt_capture()
            return

        if self.capture_start_time is None:
            self.get_logger().warn('capture_start_time is None in _handle_scanning')
            self.capture_in_progress = False
            return

        # Check for max retries
        if self.capture_attempt_count > CAPTURE_MAX_RETRIES:
            self.capture_in_progress = False
            self._enter_error(
                f'Capture failed after {CAPTURE_MAX_RETRIES} retries.'
            )
            return

        # Retry after delay
        if self.last_retry_time is not None and now - self.last_retry_time >= CAPTURE_RETRY_DELAY_SEC:
            self.last_retry_time = now
            self.capture_attempt_count += 1
            self.get_logger().warn(
                f'Retrying capture (attempt {self.capture_attempt_count + 1})...'
            )
            self._attempt_capture()

    # ----------------------------------------------------------
    # FINISHED_SCANNING
    # ----------------------------------------------------------

    def _handle_finished_scanning(self) -> None:
        if self.requested_command == RequestedCommand.SCAN:
            self.requested_command = RequestedCommand.NONE
            self.set_state(ScanState.SCANNING)

    # ----------------------------------------------------------
    # RECONSTRUCTING
    # ----------------------------------------------------------

    def _handle_reconstructing(self) -> None:
        if self.use_fake_hardware:
                self.set_state(ScanState.IDLE)
                return
        retcode = self._reconstruction_process.poll()
        if retcode is not None:
            if retcode != 0:
                self.get_logger().warn(f'Reconstruction process exited with code {retcode}')
            self._reconstruction_process = None
            self.set_state(ScanState.IDLE)

    # ----------------------------------------------------------
    # ERROR
    # ----------------------------------------------------------

    def _handle_error(self) -> None:
        # Sit in ERROR until a RESET command is received (handled at top of tick)
        pass

    # ==========================================================
    # Capture Logic
    # ==========================================================

    def _attempt_capture(self) -> None:
        """Try to capture a single RGB-D frame and save the full bundle."""

        # --- Immutable acquisition snapshot ---
        # Single-threaded executor guarantees no concurrent callback can mutate
        # self.latest between this copy and the end of this method. The shallow
        # copy is defence-in-depth: if the executor type ever changes, the
        # snapshot still isolates this attempt from subsequent _synced_callback
        # invocations that replace self.latest entirely.
        if self.latest is None:
            self.get_logger().warn('Capture attempt: no synced frame available yet.')
            return

        snapshot = dict(self.latest)

        age = (self.get_clock().now() - snapshot['received_time']).nanoseconds / 1e9
        if age > MAX_FRAME_AGE_SEC:
            self.get_logger().warn(f'Capture attempt: frame stale ({age:.3f}s old).')
            return

        color_msg       = snapshot['color_msg']
        depth_msg       = snapshot['depth_msg']
        camera_info_msg = snapshot['camera_info_msg']
        color           = snapshot['color']
        depth           = snapshot['depth']

        # --- Joint state snapshot ---
        joint_names      = list(self.current_joint_names)
        joint_positions  = [self.current_joint_positions.get(n, 0.0) for n in joint_names]
        joint_velocities = [self.current_joint_velocities.get(n, 0.0) for n in joint_names]
        velocities_available = len(self.current_joint_velocities) > 0
        if velocities_available and joint_names:
            stable = all(abs(v) < VELOCITY_STABLE_THRESHOLD for v in joint_velocities)
        else:
            stable = None

        # --- TF lookups ---
        stamp = Time.from_msg(color_msg.header.stamp)
        try:
            tf_composed = self.tf_buffer.lookup_transform(
                self.fixed_frame,
                self.camera_frame,
                stamp,
                Duration(nanoseconds=int(0.5 * 1e9)),
            )
        except TransformException as e:
            self.get_logger().warn(f'Capture attempt: TF lookup failed: {e}')
            return

        # Pose chain: base → end-effector (dynamic per-capture; non-blocking on failure)
        tf_base_to_ee: object | None = None
        try:
            tf_base_to_ee = self.tf_buffer.lookup_transform(
                self.fixed_frame,
                self.robot_ee_frame,
                stamp,
                Duration(nanoseconds=int(0.5 * 1e9)),
            )
        except TransformException as e:
            self.get_logger().warn(
                f'Capture: pose_chain {self.fixed_frame}->{self.robot_ee_frame} lookup failed: {e}'
            )

        # --- Depth quality ---
        depth_quality = self._compute_depth_quality(depth)

        # --- Timestamp delta ---
        color_ns = color_msg.header.stamp.sec * 1_000_000_000 + color_msg.header.stamp.nanosec
        depth_ns = depth_msg.header.stamp.sec * 1_000_000_000 + depth_msg.header.stamp.nanosec
        timestamp_delta_ms = abs(color_ns - depth_ns) / 1_000_000.0

        # --- Build capture directory ---
        capture_name = f'capture_{self.capture_index:04d}'

        if self.session_dir is None:
            self._create_session_dir()
        if self.session_dir is None:
            return

        session_dir  = self.session_dir
        capture_dir  = os.path.join(session_dir, capture_name)
        os.makedirs(capture_dir, exist_ok=True)

        color_path         = os.path.join(capture_dir, 'color.png')
        depth_npy_path     = os.path.join(capture_dir, 'depth.npy')
        depth_png_path     = os.path.join(capture_dir, 'depth_visualization.png')
        camera_info_path   = os.path.join(capture_dir, 'camera_info.yaml')
        capture_json_path  = os.path.join(capture_dir, 'capture.json')

        try:
            cv2.imwrite(color_path, color, [cv2.IMWRITE_PNG_COMPRESSION, 3])

            if SAVE_DEPTH_NPY:
                np.save(depth_npy_path, depth)

            if SAVE_DEPTH_PNG:
                depth_vis = self._depth_to_visualization(depth)
                cv2.imwrite(depth_png_path, depth_vis, [cv2.IMWRITE_PNG_COMPRESSION, 3])

            self._save_camera_info(camera_info_msg, camera_info_path)

            self._save_capture_json(
                color_msg=color_msg,
                depth_msg=depth_msg,
                camera_info_msg=camera_info_msg,
                tf_composed=tf_composed,
                tf_base_to_ee=tf_base_to_ee,
                depth_quality=depth_quality,
                timestamp_delta_ms=timestamp_delta_ms,
                joint_names=joint_names,
                joint_positions=joint_positions,
                joint_velocities=joint_velocities,
                velocities_available=velocities_available,
                stable=stable,
                path=capture_json_path,
                capture_name=capture_name,
                capture_dir=capture_dir,
                depth_dtype=str(depth.dtype),
            )
        except Exception as e:
            self.get_logger().warn(f'Capture attempt: file save failed: {e}')
            return

        # Success
        self.capture_index        += 1
        self.capture_in_progress   = False
        self.capture_attempt_count = 0
        self.capture_start_time    = None
        self.last_retry_time       = None

        self.session_valid_fractions.append(depth_quality['valid_fraction'])
        coverage_msg = Float32()
        coverage_msg.data = float(
            sum(self.session_valid_fractions) / len(self.session_valid_fractions) * 100.0
        )
        self.coverage_pub.publish(coverage_msg)

        self.get_logger().info(f'Capture saved: {capture_dir}')
        self.set_state(ScanState.FINISHED_SCANNING)

    # ==========================================================
    # Session Management
    # ==========================================================

    def _create_session_dir(self) -> None:
        session_id  = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_dir = os.path.join(self.save_root, session_id)
        try:
            os.makedirs(session_dir, exist_ok=True)
            self.session_dir = session_dir
            self.session_id  = session_id
            self.get_logger().info(f'Session directory created: {session_dir}')
            self._write_session_json(session_dir, session_id)
        except Exception as e:
            self._enter_error(f'Failed to create session directory: {e}')

    def _write_session_json(self, session_dir: str, session_id: str) -> None:
        path = os.path.join(session_dir, 'session.json')
        if os.path.exists(path):
            # Immutable — never overwrite an existing session.json
            self.get_logger().warn(f'session.json already exists, not overwriting: {path}')
            return

        data: dict = {
            'format_version':    1,
            'session_id':        session_id,
            'session_dir':       session_dir,
            'fixed_frame':       self.fixed_frame,
            'camera_frame':      self.camera_frame,
            'robot_ee_frame':    self.robot_ee_frame,
            'intermediate_frame': self.intermediate_frame,
            'calibration':       self.calibration_snapshot,
            'calibration_unavailable_reason': self.calibration_unavailable_reason,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        self.get_logger().info(
            f'session.json written '
            f'(calibration: {"available" if self.calibration_snapshot else "unavailable"})'
        )

    def _reset_session(self) -> None:
        self.get_logger().info('Session reset.')
        # Save session_dir so an immediately following RECONSTRUCT command can
        # still locate the completed session on disk.  session_dir itself is
        # cleared so the next scan always creates a fresh directory via
        # _create_session_dir() rather than appending to the previous session.
        self._last_session_dir        = self.session_dir
        self.session_dir              = None
        self.session_id               = None
        self.pending_scan_name        = None
        self.capture_index            = 0
        self.session_valid_fractions  = []
        self._reconstruction_process  = None
        self.capture_in_progress      = False
        self.capture_attempt_count    = 0
        self.capture_start_time       = None
        self.last_retry_time          = None

    # ==========================================================
    # Camera Sync Callback
    # ==========================================================

    def _synced_callback(
        self,
        color_msg: Image,
        depth_msg: Image,
        camera_info_msg: CameraInfo,
    ) -> None:
        try:
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        if color_msg.header.frame_id != self.camera_frame:
            self.get_logger().warn(
                f'Color frame_id {color_msg.header.frame_id} != '
                f'configured camera_frame {self.camera_frame}'
            )

        self.latest = {
            'color_msg':       color_msg,
            'depth_msg':       depth_msg,
            'camera_info_msg': camera_info_msg,
            'color':           color,
            'depth':           depth,
            'received_time':   self.get_clock().now(),
        }

    # ==========================================================
    # Joint State Callback
    # ==========================================================

    def _joint_callback(self, msg: JointState) -> None:
        self.current_joint_names = list(msg.name)
        self.current_joint_positions = dict(zip(msg.name, msg.position))
        if len(msg.velocity) == len(msg.name):
            self.current_joint_velocities = dict(zip(msg.name, msg.velocity))

    # ==========================================================
    # Helpers
    # ==========================================================

    def _transform_to_dict(self, tf) -> dict:
        """Serialise a TransformStamped to a plain dict."""
        t = tf.transform.translation
        q = tf.transform.rotation
        return {
            'source_frame': tf.child_frame_id,
            'target_frame': tf.header.frame_id,
            'stamp': {
                'sec':     int(tf.header.stamp.sec),
                'nanosec': int(tf.header.stamp.nanosec),
            },
            'translation': {
                'x': float(t.x),
                'y': float(t.y),
                'z': float(t.z),
            },
            'rotation_xyzw': {
                'x': float(q.x),
                'y': float(q.y),
                'z': float(q.z),
                'w': float(q.w),
            },
        }

    def _compute_depth_quality(self, depth: np.ndarray) -> dict:
        """Compute depth validity statistics from a raw depth array (uint16 mm or float32 m)."""
        total   = int(depth.size)
        valid_m = (depth > 0)
        valid_n = int(valid_m.sum())
        if valid_n > 0:
            if depth.dtype == np.uint16:
                depth_m = depth.astype(np.float32) / 1000.0
            else:
                depth_m = depth.astype(np.float32)
            valid_vals = depth_m[valid_m]
            median_m = float(np.median(valid_vals))
            min_m    = float(valid_vals.min())
            max_m    = float(valid_vals.max())
        else:
            median_m = 0.0
            min_m    = 0.0
            max_m    = 0.0
        return {
            'total_pixels':    total,
            'valid_pixels':    valid_n,
            'valid_fraction':  round(valid_n / total, 4) if total > 0 else 0.0,
            'median_depth_m':  round(median_m, 4),
            'min_depth_m':     round(min_m, 4),
            'max_depth_m':     round(max_m, 4),
        }

    def _depth_to_visualization(self, depth: np.ndarray) -> np.ndarray:
        depth_float = depth.astype(np.float32)
        valid = depth_float > 0

        if not np.any(valid):
            return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

        min_val = np.percentile(depth_float[valid], 2)
        max_val = np.percentile(depth_float[valid], 98)

        if max_val <= min_val:
            max_val = min_val + 1.0

        depth_clipped = np.clip(depth_float, min_val, max_val)
        depth_norm    = ((depth_clipped - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
        depth_norm[~valid] = 0
        return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    def _save_camera_info(self, camera_info_msg: CameraInfo, path: str) -> None:
        def to_float_list(seq):
            return [float(x) for x in seq]

        data = {
            'header': {
                'stamp': {
                    'sec':     int(camera_info_msg.header.stamp.sec),
                    'nanosec': int(camera_info_msg.header.stamp.nanosec),
                },
                'frame_id': str(camera_info_msg.header.frame_id),
            },
            'height':            int(camera_info_msg.height),
            'width':             int(camera_info_msg.width),
            'distortion_model':  str(camera_info_msg.distortion_model),
            'd':                 to_float_list(camera_info_msg.d),
            'k':                 to_float_list(camera_info_msg.k),
            'r':                 to_float_list(camera_info_msg.r),
            'p':                 to_float_list(camera_info_msg.p),
            'binning_x':         int(camera_info_msg.binning_x),
            'binning_y':         int(camera_info_msg.binning_y),
        }
        with open(path, 'w') as f:
            yaml.safe_dump(data, f, sort_keys=False)

    def _save_capture_json(
        self,
        *,
        color_msg,
        depth_msg,
        camera_info_msg,
        tf_composed,
        tf_base_to_ee,
        depth_quality: dict,
        timestamp_delta_ms: float,
        joint_names: list,
        joint_positions: list,
        joint_velocities: list,
        velocities_available: bool,
        stable: bool | None,
        path: str,
        capture_name: str,
        capture_dir: str,
        depth_dtype: str,
    ) -> None:
        pose_chain: dict = {}
        if tf_base_to_ee is not None:
            pose_chain[f'{self.fixed_frame}_to_{self.robot_ee_frame}'] = \
                self._transform_to_dict(tf_base_to_ee)

        data = {
            'format_version': 2,
            'capture_name':   capture_name,
            'capture_dir':    capture_dir,
            'session_dir':    self.session_dir,
            'fixed_frame':    self.fixed_frame,
            'camera_frame':   self.camera_frame,
            'robot_ee_frame': self.robot_ee_frame,
            'timestamps': {
                'color': {
                    'sec':     int(color_msg.header.stamp.sec),
                    'nanosec': int(color_msg.header.stamp.nanosec),
                },
                'depth': {
                    'sec':     int(depth_msg.header.stamp.sec),
                    'nanosec': int(depth_msg.header.stamp.nanosec),
                },
                'camera_info': {
                    'sec':     int(camera_info_msg.header.stamp.sec),
                    'nanosec': int(camera_info_msg.header.stamp.nanosec),
                },
                'timestamp_delta_color_depth_ms': round(timestamp_delta_ms, 3),
            },
            'image_size': {
                'width':  int(color_msg.width),
                'height': int(color_msg.height),
            },
            'depth_dtype':    depth_dtype,
            'depth_quality':  depth_quality,
            'robot_state': {
                'joint_names':           joint_names,
                'positions_rad':         [round(v, 8) for v in joint_positions],
                'velocities_rad_s':      [round(v, 8) for v in joint_velocities],
                'velocities_available':  velocities_available,
                'stable':                stable,
                'stability_threshold_rad_s': VELOCITY_STABLE_THRESHOLD,
            },
            'pose_composed': self._transform_to_dict(tf_composed),
            'pose_chain':    pose_chain,
            'calibration_source': 'session.json',
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    # ==========================================================
    # Command Callback
    # ==========================================================

    def _scan_command_callback(self, msg: String) -> None:
        command = msg.data
        self.get_logger().info(f'Received scan command: {command}')

        if command == 'SCAN':
            self.requested_command = RequestedCommand.SCAN
        elif command == 'RESET':
            self.requested_command = RequestedCommand.RESET
        elif command == 'ERROR':
            self.requested_command = RequestedCommand.ERROR
        elif command == 'RECONSTRUCT':
            self._launch_reconstruction()
        else:
            self.get_logger().warn(f'Unknown scan command: {command}')

    def _scan_name_callback(self, msg: String) -> None:
        name = msg.data.strip()
        if name:
            self.pending_scan_name = name
            self.get_logger().info(f'Scan name received: "{name}"')
        else:
            self.pending_scan_name = None
            self.get_logger().info('Scan name cleared — will use timestamp default.')

    @staticmethod
    def _sanitise_scan_name(name: str) -> str:
        name = name.strip().replace(' ', '_')
        name = re.sub(r'[^\w\-]', '', name)
        return name[:64]

    def _try_rename_session(self) -> None:
        if self.pending_scan_name is None or self.session_dir is None:
            return
        sanitised = self._sanitise_scan_name(self.pending_scan_name)
        self.pending_scan_name = None
        if not sanitised:
            self.get_logger().warn('Scan name empty after sanitisation — keeping timestamp name.')
            return
        new_dir = os.path.join(self.save_root, sanitised)
        if os.path.exists(new_dir):
            suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
            new_dir = f'{new_dir}_{suffix}'
            self.get_logger().warn(
                f'Scan name "{sanitised}" already exists — using "{os.path.basename(new_dir)}".'
            )
        try:
            os.rename(self.session_dir, new_dir)
            self.get_logger().info(f'Session renamed to: {new_dir}')
            self.session_dir = new_dir
        except OSError as e:
            self.get_logger().error(f'Failed to rename session directory: {e}')

    def _launch_reconstruction(self) -> None:
        if not self.auto_reconstruct:
            self.get_logger().info('auto_reconstruct disabled — skipping reconstruction.')
            return
        # Fake hardware never creates a session dir; enter RECONSTRUCTING so the
        # main_control handshake can complete, then _handle_reconstructing() will
        # immediately transition back to IDLE.
        if self.use_fake_hardware:
            self.set_state(ScanState.RECONSTRUCTING)
            return
        # RECONSTRUCT normally arrives after a RESET, so session_dir has been
        # cleared and the path was saved in _last_session_dir.  Fall back to
        # session_dir for the rare case where RECONSTRUCT arrives without a
        # prior RESET (e.g. manual invocation).
        target_dir = self._last_session_dir or self.session_dir
        if target_dir is None:
            self.get_logger().warn('RECONSTRUCT received but no session directory available — skipping.')
            return
        self._last_session_dir = None  # consumed
        script = str(Path(__file__).resolve().parent.parent / 'helpers' / 'reconstruct.py')
        # Derive models-dir as reconstructed_scans/<session_name> alongside raw_scans/
        session_name = os.path.basename(target_dir)
        recon_root   = os.path.join(os.path.dirname(self.save_root), 'reconstructed_scans')
        models_dir   = os.path.join(recon_root, session_name)
        cmd = [
            sys.executable, script,
            '--session',        target_dir,
            '--models-dir',     models_dir,
            '--voxel-size',     str(self.recon_voxel_size),
            '--icp-threshold',  str(self.recon_icp_threshold),
            '--mesh',
            '--no-visualise',
        ]
        if self.recon_icp:
            cmd.append('--icp')
        self.get_logger().info(f'Launching reconstruction: {" ".join(cmd)}')
        self._reconstruction_process = subprocess.Popen(cmd, env=os.environ.copy())
        self.set_state(ScanState.RECONSTRUCTING)

    # ==========================================================
    # Publishers
    # ==========================================================

    def _publish_status(self) -> None:
        msg = String()
        msg.data = self.state.name
        self.status_pub.publish(msg)

    def _publish_heartbeat(self) -> None:
        self.heartbeat_pub.publish(Empty())

    def _publish_startup_status(self) -> None:
        if self._startup_status_count >= self._startup_status_limit:
            self.startup_status_timer.cancel()
            return
        self._publish_status()
        self._startup_status_count += 1


# ==========================================================
# Entry Point
# ==========================================================

def main(args=None) -> None:
    rclpy.init(args=args)
    node = ScanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
