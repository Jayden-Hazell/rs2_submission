#!/usr/bin/env python3
# 41069 Robotics Studio 2
# Authors: Jayden Hazell (24953364), Lachlan Partridge (24964557), Tomas Klimes (24773804)
# Controls UR3e robot joint trajectories, manages scan waypoints, and handles auto/manual movement states.
"""
Movement node for the UR3e scanning system.

Owns all robot motion: moves to the pre-scan home position, executes the
auto-scan waypoint sequence, and services manual viewpoint requests. Uses a
background watchdog thread to detect when each joint target is reached,
missed, or faulted.
"""

from __future__ import annotations

import csv
import json
import math
import threading
import time
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from dataclasses import dataclass
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float32, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class MovementState(Enum):
    BOOTING = auto()
    IDLE = auto()
    MOVE_PRE_SCAN_POSITION = auto()
    PRE_SCAN_POSITION = auto()
    AUTO_EXECUTING = auto()
    MANUAL_EXECUTING = auto()
    PAUSED = auto()
    ERROR = auto()

class AutoExecutionSubState(Enum):
    MOVING_TO_WAYPOINT = auto()
    WAITING_FOR_SCAN = auto()
    FINISHED = auto()

class ManualMovementState(Enum):
    IDLE = auto()
    MOVING_TO_WAYPOINT = auto()
    FINISHED = auto()

class MotionStatus(Enum):
    IN_PROGRESS = auto()
    REACHED = auto()
    TIMEOUT = auto()
    FAULT = auto()
    INTERRUPTED = auto()

class RequestedCommand(Enum):
    NONE = auto()
    MOVE_PRE_SCAN_POSITION = auto()
    CONTINUE = auto()
    PAUSE = auto()
    IDLE = auto()
    ERROR = auto()
    EMERGENCY_STOP = auto()


_MANUAL_POSITIONS_CSV = """\
name,shoulder_pan_joint,shoulder_lift_joint,elbow_joint,wrist_1_joint,wrist_2_joint,wrist_3_joint
Top,-0.8293,-2.0747,4.5011,-4.7372,0.7967,5.5078
Front,-0.8293,-2.0747,4.5011,-4.7372,0.7967,5.5078
Back,-1.1137,-2.6392,-0.4107,-5.0318,-2.7092,-1.0041
Left,-2.4351,-1.0646,-2.4313,1.1567,2.9891,-0.5513
Right,-2.5413,-0.7462,3.9479,2.4918,-5.7831,-2.1616
"""

class MovementNode(Node):
    def __init__(self) -> None:
        super().__init__('movement_node')

        # ------------------------------------------
        # Variables
        # ------------------------------------------

        self.state: MovementState = MovementState.BOOTING  # Current top-level state
        self.auto_substate: AutoExecutionSubState | None = None  # Current auto-execution sub-state
        self.requested_command: RequestedCommand = RequestedCommand.NONE
        self.emergency_stop_latched: bool = False

        # Launch state publisher 
        self.startup_status_publish_count: int = 0
        self.startup_status_publish_limit: int = 20  # 20 x 0.5s = 10 seconds

        # Speed Variables
        self.speed_to_move_time = {
            "low": 6.0,
            "medium": 3.0,
            "high": 1.5,
        }

        self.scan_speed = "medium"
        self.scan_move_time_sec = self.speed_to_move_time[self.scan_speed]

        self.scan_resolution = "high"
        self.resolution_to_fraction = {
            "high": 1.0,
            "medium": 0.75,
            "low": 0.50,
        }

        self.watchdog_extra_time_sec = 5.0

        self.home_move_time_sec = 3.0

        # ------------------------------------------
        # Publishers
        # ------------------------------------------

        self.heartbeat_pub = self.create_publisher(
            Empty,
            '/heartbeat/movement',
            10,
        )
        self.status_pub = self.create_publisher(
            String,
            '/movement/status',
            10,
        )
        self.progress_pub = self.create_publisher(
            Float32,
            '/movement/progress',
            10,
        )

        # ------------------------------------------
        # Subscribers
        # ------------------------------------------
        self.movement_command_sub = self.create_subscription(
            String,
            '/control/movement_command',
            self._movement_command_callback,
            10,
        )

        self.movement_speed_sub = self.create_subscription(
            String,
            "/control/movement_speed",
            self._movement_speed_callback,
            10,
        )

        # ------------------------------------------
        # Timers
        # ------------------------------------------
        self.heartbeat_timer = self.create_timer(
            0.5,
            self._publish_heartbeat,
        )
        self.state_machine_timer = self.create_timer(
            0.1,
            self.state_machine_tick,
        )
        self.startup_status_timer = self.create_timer(
            0.5,
            self._publish_startup_status,
        )

        # ------------------------------------------
        # Logging
        # ------------------------------------------
        self.get_logger().info(f'Initial state: {self.state.name}')
        self.get_logger().info('Movement node started.')

        self._publish_status()

        # ------------------------------------------
        # Joint control setup
        # ------------------------------------------
        self.callback_group = ReentrantCallbackGroup()
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.current_joints: Dict[str, float] = {}

        # Home config
        self.home_position = [-1.57, -1.57, -1.57, -1.57, 1.57, 0.00]

        self.joint_reach_tolerance = 0.05
        self.home_timeout_sec = 8.0

        # Broad software sanity limits for the UR joints.
        # These are intentionally robot-level limits, not scan-shape limits.
        # The old narrow limits rejected valid hand-recorded poses such as
        # shoulder_lift_joint=-3.433 rad.
        two_pi = 2.0 * math.pi
        self.safe_joint_limits = {
            "shoulder_pan_joint": (-two_pi, two_pi),
            "shoulder_lift_joint": (-two_pi, two_pi),
            "elbow_joint":         (-two_pi, two_pi),
            "wrist_1_joint":       (-two_pi, two_pi),
            "wrist_2_joint":       (-two_pi, two_pi),
            "wrist_3_joint":       (-two_pi, two_pi),
        }


        # ------------------------------------------
        # End-effector workspace safety limits
        # ------------------------------------------
        self.enable_fk_workspace_check = True

        # X/Y allow 0.75 m in each direction from the robot base
        self.ee_min_x = -0.75
        self.ee_max_x =  0.75

        self.ee_min_y = -0.75
        self.ee_max_y =  0.75

        
        # Minimum safe height is 0.03 m above the table.
        self.ee_min_z = 0.03

        # Top of 1 m cube measured from robot base
        self.ee_max_z = 1.00

        # Scan path configuration.
        # NOTE: the CSV is NOT loaded in __init__. It is checked/loaded in
        # the BOOTING state so startup success/failure is handled by the
        # state machine instead of crashing the node constructor.
        # Expected file: <rs2 package>/joints/autoScan.csv
        # Waypoint order in the CSV should match self.joint_names:
        # shoulder_pan_joint, shoulder_lift_joint, elbow_joint,
        # wrist_1_joint, wrist_2_joint, wrist_3_joint
        self.scan_waypoint_file: Optional[str] = None
        self.all_scan_waypoints: List[List[float]] = []
        self.scan_waypoints: List[List[float]] = []
        self.current_waypoint_index: int = 0
        self.boot_completed: bool = False
        self.paused_resume_state: Optional[MovementState] = None
        self.paused_resume_auto_substate: Optional[AutoExecutionSubState] = None

        self.manual_positions: Dict[str, List[float]] = {}
        self.pending_manual_target: Optional[str] = None
        self.manual_sub_state: ManualMovementState = ManualMovementState.IDLE

        self.motion_status: MotionStatus = MotionStatus.IN_PROGRESS
        self.motion_status_lock = threading.Lock()
        self.watchdog_thread: Optional[threading.Thread] = None

        # Publishers
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            "/scaled_joint_trajectory_controller/joint_trajectory",
            10,
        )

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_callback,
            10,
        )
    # ------------------------------------------
    # Scan waypoint loading
    # ------------------------------------------
    def _resolve_scan_waypoint_file(self) -> str:
        """Find joints/autoScan.csv in the package/source tree."""
        relative_path = Path("joints") / "autoScan.csv"

        candidates: List[Path] = []

        # Supports the literal path requested by the team if it exists.
        candidates.append(Path("/joints") / "autoScan.csv")

        # Works when the file is installed into the rs2 package share directory.
        try:
            candidates.append(Path(get_package_share_directory("rs2")) / relative_path)
        except Exception as exc:
            self.get_logger().warn(f"Could not resolve rs2 package share directory: {exc}")

        # Works when running from the source tree with --symlink-install.
        current_file = Path(__file__).resolve()
        for parent in current_file.parents:
            candidates.append(parent / relative_path)

        # Useful when launching from the workspace/package root.
        candidates.append(Path.cwd() / relative_path)

        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)

        searched = "\n  - ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "Could not find joints/autoScan.csv. Searched:\n  - " + searched
        )

    def load_scan_waypoints_from_csv(self, csv_path: str) -> List[List[float]]:
        """Load scan waypoints from CSV in self.joint_names order.

        Preferred CSV format:
            shoulder_pan_joint,shoulder_lift_joint,elbow_joint,wrist_1_joint,wrist_2_joint,wrist_3_joint
            0.0,-1.57,0.0,-1.57,0.0,0.0

        Also supports the recorder format:
            timestamp,name1,name2,...,name6,value1,value2,...,value6
        """
        waypoints: List[List[float]] = []

        with open(csv_path, "r", newline="") as csv_file:
            rows = list(csv.reader(csv_file))

        if not rows:
            raise ValueError(f"Scan waypoint CSV is empty: {csv_path}")

        for row_number, row in enumerate(rows, start=1):
            if not row or all(cell.strip() == "" for cell in row):
                continue

            row = [cell.strip() for cell in row]

            # Skip a normal header row if present.
            if row[0] == self.joint_names[0] or row[0].lower() in ("timestamp", "time"):
                continue

            try:
                if len(row) == len(self.joint_names):
                    positions = [float(value) for value in row]

                elif len(row) >= 13:
                    # Recorder format: timestamp, 6 joint names, 6 joint values.
                    recorded_joint_names = row[1:7]
                    recorded_joint_values = [float(value) for value in row[7:13]]
                    recorded_map = dict(zip(recorded_joint_names, recorded_joint_values))
                    positions = [recorded_map[name] for name in self.joint_names]

                else:
                    raise ValueError(
                        f"expected 6 columns or recorder-format 13 columns, got {len(row)}"
                    )

            except Exception as exc:
                raise ValueError(
                    f"Invalid scan waypoint CSV row {row_number}: {row}. Error: {exc}"
                ) from exc

            if not self.is_waypoint_safe(positions):
                raise ValueError(
                    f"Scan waypoint row {row_number} is outside configured robot limits: {positions}"
                )

            waypoints.append(positions)

        if not waypoints:
            raise ValueError(f"No valid scan waypoints loaded from {csv_path}")

        return waypoints
    

    def apply_resolution_to_waypoints(self) -> None:
        """Apply GUI resolution setting to choose how many scan waypoints are used."""
        if not self.all_scan_waypoints:
            self.scan_waypoints = []
            self.current_waypoint_index = 0
            return

        fraction = self.resolution_to_fraction.get(self.scan_resolution, 1.0)
        total_waypoints = len(self.all_scan_waypoints)
        desired_count = max(1, int(round(total_waypoints * fraction)))

        if desired_count >= total_waypoints:
            self.scan_waypoints = list(self.all_scan_waypoints)
        else:
            indices = np.linspace(
                0,
                total_waypoints - 1,
                desired_count,
                dtype=int,
            )
            self.scan_waypoints = [
                self.all_scan_waypoints[int(index)]
                for index in indices
            ]

        self.current_waypoint_index = 0
        self.get_logger().info(
            f"Resolution updated: {self.scan_resolution}. "
            f"Using {len(self.scan_waypoints)}/{len(self.all_scan_waypoints)} scan waypoints."
        )
        self.publish_progress(completed_waypoints=0)

        

    def _ensure_manual_position_file(self) -> str:
        """Find joints/manualPosition.csv, creating it from defaults if absent."""
        relative_path = Path("joints") / "manualPosition.csv"
        candidates: List[Path] = []

        try:
            candidates.append(Path(get_package_share_directory("rs2")) / relative_path)
        except Exception as exc:
            self.get_logger().warn(f"Could not resolve rs2 package share directory: {exc}")

        current_file = Path(__file__).resolve()
        for parent in current_file.parents:
            candidates.append(parent / relative_path)

        candidates.append(Path.cwd() / relative_path)

        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)

        for candidate in candidates:
            if candidate.parent.is_dir():
                candidate.write_text(_MANUAL_POSITIONS_CSV)
                self.get_logger().warn(
                    f"manualPosition.csv not found — wrote defaults to {candidate}"
                )
                return str(candidate)

        raise FileNotFoundError(
            "Could not find or create joints/manualPosition.csv. Searched: "
            + ", ".join(str(c) for c in candidates)
        )

    def _load_manual_positions(self, csv_path: str) -> Dict[str, List[float]]:
        """Load named manual positions from CSV, keyed by lowercase name."""
        positions: Dict[str, List[float]] = {}

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("name", "").strip().lower()
                if not name:
                    continue
                try:
                    joint_values = [float(row[j]) for j in self.joint_names]
                except (KeyError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid row for position '{name}' in {csv_path}: {exc}"
                    ) from exc
                if not self.is_waypoint_safe(joint_values):
                    raise ValueError(
                        f"Manual position '{name}' is outside robot limits: {joint_values}"
                    )
                positions[name] = joint_values

        if not positions:
            raise ValueError(f"No valid manual positions loaded from {csv_path}")

        return positions

    # ------------------------------------------
    # State Management
    # ------------------------------------------

    def set_state(self, new_state: MovementState) -> None:
        if self.state == new_state:
            return

        self.get_logger().info(
            f'State change: {self.state.name} -> {new_state.name}'
        )
        self.state = new_state

        if new_state != MovementState.AUTO_EXECUTING:
            self.auto_substate = None

        if new_state != MovementState.MANUAL_EXECUTING:
            self.manual_sub_state = ManualMovementState.IDLE

        self._publish_status()

    def set_auto_substate(
        self,
        new_substate: AutoExecutionSubState | None,
    ) -> None:
        if self.state != MovementState.AUTO_EXECUTING:
            self.get_logger().warn(
                'Attempted to set AUTO_EXECUTING sub-state while not in AUTO_EXECUTING.'
            )
            return

        if self.auto_substate == new_substate:
            return

        previous = self.auto_substate.name if self.auto_substate else 'None'
        current = new_substate.name if new_substate else 'None'

        self.get_logger().info(
            f'AUTO_EXECUTING sub-state change: {previous} -> {current}'
        )
        self.auto_substate = new_substate
        self._publish_status()

    # ------------------------------------------
    # State Machine Tick
    # ------------------------------------------

    def state_machine_tick(self) -> None:
        if self.emergency_stop_latched:
            if self.state != MovementState.ERROR:
                self.set_state(MovementState.ERROR)
            return
        if self.state == MovementState.BOOTING:
            self._handle_booting()
        elif self.state == MovementState.IDLE:
            self._handle_idle()
        elif self.state == MovementState.MOVE_PRE_SCAN_POSITION:
            self._handle_move_pre_scan_position()
        elif self.state == MovementState.PRE_SCAN_POSITION:
            self._handle_pre_scan_position()
        elif self.state == MovementState.AUTO_EXECUTING:
            self._handle_auto_executing()
        elif self.state == MovementState.MANUAL_EXECUTING:
            self._handle_manual_executing()
        elif self.state == MovementState.PAUSED:
            self._handle_paused()
        elif self.state == MovementState.ERROR:
            self._handle_error()

    # ------------------------------------------
    # Top-Level State Handlers
    # ------------------------------------------

    def _handle_booting(self) -> None:
        if self.boot_completed:
            self.set_state(MovementState.IDLE)
            return

        try:
            self.scan_waypoint_file = self._resolve_scan_waypoint_file()
            self.all_scan_waypoints = self.load_scan_waypoints_from_csv(
                self.scan_waypoint_file
            )
            self.apply_resolution_to_waypoints()

            manual_csv = self._ensure_manual_position_file()
            self.manual_positions = self._load_manual_positions(manual_csv)

            self.boot_completed = True
            self.get_logger().info(
                f"Boot check passed. Loaded {len(self.scan_waypoints)} scan waypoints "
                f"from {self.scan_waypoint_file}. "
                f"Loaded {len(self.manual_positions)} manual positions."
            )
            self.publish_progress()
            self.set_state(MovementState.IDLE)

        except Exception as exc:
            self.get_logger().error(f"Boot check failed: {exc}")
            self.set_state(MovementState.ERROR)

    def _handle_idle(self) -> None:
        if self.requested_command == RequestedCommand.MOVE_PRE_SCAN_POSITION:
            self.requested_command = RequestedCommand.NONE
            self.reset_scan_progress()
            with self.motion_status_lock:
                self.motion_status = MotionStatus.IN_PROGRESS
            self.watchdog_thread = None
            self.set_state(MovementState.MOVE_PRE_SCAN_POSITION)

        elif self.requested_command == RequestedCommand.PAUSE:
            self.requested_command = RequestedCommand.NONE
            self.pause_motion()

    def _handle_move_pre_scan_position(self) -> None:
        if self.requested_command == RequestedCommand.PAUSE:
            self.requested_command = RequestedCommand.NONE
            self.pause_motion()
            return

        status = self.get_motion_status()

        # First entry — send command and start watchdog
        if status == MotionStatus.IN_PROGRESS and self.watchdog_thread is None:
            if not self.joints_ready():
                self.get_logger().warn("Joints not ready.")
                self.set_state(MovementState.ERROR)
                return

            target = self.get_final_joint_target(
                self.home_position,
                use_nearest=False,
            )

            if target is None:
                self.get_logger().error("Failed to calculate home target.")
                self.set_state(MovementState.ERROR)
                return

            success = self.publish_joint_target(
                target,
                move_time_sec=self.home_move_time_sec,
                use_nearest=False,
            )

            if not success:
                self.get_logger().error("Failed to publish joint target.")
                self.set_state(MovementState.ERROR)
                return

            self._start_watchdog(
                target_positions=target,
                tolerance=self.joint_reach_tolerance,
                timeout_sec=self.home_timeout_sec,
            )
            return

        # Subsequent ticks — check watchdog result
        if status == MotionStatus.REACHED:
            self.get_logger().info("Pre-scan position reached.")
            self.watchdog_thread = None
            self.set_state(MovementState.PRE_SCAN_POSITION)

        elif status == MotionStatus.TIMEOUT:
            self.get_logger().error("Timed out reaching pre-scan position.")
            self.watchdog_thread = None
            self.set_state(MovementState.ERROR)

        elif status == MotionStatus.FAULT:
            self.get_logger().error("Joint fault detected during pre-scan move.")
            self.watchdog_thread = None
            self.set_state(MovementState.ERROR)

        elif status == MotionStatus.INTERRUPTED:
            self.get_logger().warn("Pre-scan move interrupted.")
            self.watchdog_thread = None

    def _handle_pre_scan_position(self) -> None:
        if self.requested_command == RequestedCommand.PAUSE:
            self.requested_command = RequestedCommand.NONE
            self.pause_motion()
            return

        if self.requested_command == RequestedCommand.CONTINUE:
            self.requested_command = RequestedCommand.NONE
            with self.motion_status_lock:
                self.motion_status = MotionStatus.IN_PROGRESS
            self.watchdog_thread = None
            self.set_state(MovementState.AUTO_EXECUTING)
            self.set_auto_substate(AutoExecutionSubState.MOVING_TO_WAYPOINT)

    def _handle_auto_executing(self) -> None:
        if self.requested_command == RequestedCommand.PAUSE:
            self.requested_command = RequestedCommand.NONE
            self.pause_motion()
            return
        if self.auto_substate == AutoExecutionSubState.MOVING_TO_WAYPOINT:
            self._handle_moving_to_waypoint()
        elif self.auto_substate == AutoExecutionSubState.WAITING_FOR_SCAN:
            self._handle_waiting_for_scan()
        elif self.auto_substate == AutoExecutionSubState.FINISHED:
            self._handle_finished()

    def _handle_manual_executing(self) -> None:
        if self.pending_manual_target is None:
            self.get_logger().error("MANUAL_EXECUTING with no target. Returning to IDLE.")
            self.set_state(MovementState.IDLE)
            return

        if self.manual_sub_state == ManualMovementState.IDLE:
            target = self.manual_positions[self.pending_manual_target]
            status = self.get_motion_status()

            if status == MotionStatus.IN_PROGRESS and self.watchdog_thread is None:
                if not self.joints_ready():
                    self.get_logger().warn("Joints not ready for manual move.")
                    self.set_state(MovementState.ERROR)
                    return

                move_time = self.scan_move_time_sec
                success = self.publish_joint_target(target, move_time_sec=move_time)

                if not success:
                    self.get_logger().error("Failed to publish manual joint target.")
                    self.set_state(MovementState.ERROR)
                    return

                self._start_watchdog(
                    target_positions=target,
                    tolerance=self.joint_reach_tolerance,
                    timeout_sec=move_time + self.watchdog_extra_time_sec,
                )

                self.manual_sub_state = ManualMovementState.MOVING_TO_WAYPOINT
                self._publish_status()
                return

        elif self.manual_sub_state == ManualMovementState.MOVING_TO_WAYPOINT:
            status = self.get_motion_status()

            if status == MotionStatus.REACHED:
                self.get_logger().info(
                    f"Manual position '{self.pending_manual_target}' reached. Manual movement complete."
                )

                self.watchdog_thread = None
                self.pending_manual_target = None
                self.manual_sub_state = ManualMovementState.IDLE

                with self.motion_status_lock:
                    self.motion_status = MotionStatus.IN_PROGRESS

                self.set_state(MovementState.IDLE)
                return

            elif status == MotionStatus.TIMEOUT:
                self.get_logger().error(
                    f"Timed out reaching manual position '{self.pending_manual_target}'."
                )
                self.watchdog_thread = None
                self.set_state(MovementState.ERROR)
                return

            elif status == MotionStatus.FAULT:
                self.get_logger().error(
                    f"Joint fault during manual move to '{self.pending_manual_target}'."
                )
                self.watchdog_thread = None
                self.set_state(MovementState.ERROR)
                return

            elif status == MotionStatus.INTERRUPTED:
                self.get_logger().warn("Manual move interrupted.")
                self.watchdog_thread = None
                self.pending_manual_target = None
                self.manual_sub_state = ManualMovementState.IDLE

                with self.motion_status_lock:
                    self.motion_status = MotionStatus.IN_PROGRESS

                self.set_state(MovementState.IDLE)
                return

        

    def _handle_paused(self) -> None:
        if self.requested_command == RequestedCommand.MOVE_PRE_SCAN_POSITION:
            self.requested_command = RequestedCommand.NONE

            self.get_logger().info(
                "Move to home requested while paused. Resetting scan progress and returning to pre-scan position."
            )

            self.reset_scan_progress()
            self.paused_resume_state = None
            self.paused_resume_auto_substate = None

            with self.motion_status_lock:
                self.motion_status = MotionStatus.IN_PROGRESS

            self.watchdog_thread = None
            self.set_state(MovementState.MOVE_PRE_SCAN_POSITION)
            return

        if self.requested_command == RequestedCommand.CONTINUE:
            self.requested_command = RequestedCommand.NONE
            self.resume_from_pause()
            return

        if self.requested_command == RequestedCommand.IDLE:
            self.requested_command = RequestedCommand.NONE
            self.set_state(MovementState.IDLE)

    def _handle_error(self) -> None:
        pass

    # ------------------------------------------
    # Auto Sub-State Handlers
    # ------------------------------------------

    def _handle_moving_to_waypoint(self) -> None:
        if self.current_waypoint_index >= len(self.scan_waypoints):
            self.get_logger().info("All scan waypoints completed.")
            self.set_auto_substate(AutoExecutionSubState.FINISHED)
            return

        target = self.scan_waypoints[self.current_waypoint_index]
        status = self.get_motion_status()

        # First entry for this waypoint — send command and start watchdog.
        if status == MotionStatus.IN_PROGRESS and self.watchdog_thread is None:
            if not self.joints_ready():
                self.get_logger().warn("Joints not ready.")
                self.set_state(MovementState.ERROR)
                return

            waypoint_number = self.current_waypoint_index + 1
            total_waypoints = len(self.scan_waypoints)
            self.get_logger().info(
                f"Moving to scan waypoint {waypoint_number}/{total_waypoints}: {target}"
            )

            safe_target = self.make_nearest_joint_target(target)
            if safe_target is None:
                self.get_logger().error("Failed to calculate nearest scan waypoint target.")
                self.set_state(MovementState.ERROR)
                return

            success = self.publish_joint_target(
                safe_target,
                move_time_sec=self.scan_move_time_sec,
            )
            if not success:
                self.get_logger().error("Failed to publish scan waypoint target.")
                self.set_state(MovementState.ERROR)
                return

            self._start_watchdog(
                target_positions=safe_target,
                tolerance=self.joint_reach_tolerance,
                timeout_sec=self.scan_move_time_sec + self.watchdog_extra_time_sec,
            )
            return

        # Subsequent ticks — check watchdog result.
        if status == MotionStatus.REACHED:
            self.get_logger().info(
                f"Reached scan waypoint {self.current_waypoint_index + 1}/{len(self.scan_waypoints)}."
            )
            self.watchdog_thread = None
            self.publish_progress(completed_waypoints=self.current_waypoint_index + 1)
            self.set_auto_substate(AutoExecutionSubState.WAITING_FOR_SCAN)

        elif status == MotionStatus.TIMEOUT:
            self.get_logger().error(
                f"Timed out reaching scan waypoint {self.current_waypoint_index + 1}."
            )
            self.watchdog_thread = None
            self.set_state(MovementState.ERROR)

        elif status == MotionStatus.FAULT:
            self.get_logger().error(
                f"Joint fault detected while moving to scan waypoint {self.current_waypoint_index + 1}."
            )
            self.watchdog_thread = None
            self.set_state(MovementState.ERROR)

        elif status == MotionStatus.INTERRUPTED:
            self.get_logger().warn("Scan waypoint move interrupted.")
            self.watchdog_thread = None

    def _handle_waiting_for_scan(self) -> None:
        if self.requested_command == RequestedCommand.CONTINUE:
            self.requested_command = RequestedCommand.NONE
            self.current_waypoint_index += 1

            if self.current_waypoint_index >= len(self.scan_waypoints):
                self.get_logger().info("Final scan waypoint completed. Auto execution finished.")
                self.publish_progress(completed_waypoints=len(self.scan_waypoints))
                self.set_auto_substate(AutoExecutionSubState.FINISHED)
            else:
                with self.motion_status_lock:
                    self.motion_status = MotionStatus.IN_PROGRESS
                self.watchdog_thread = None
                self.set_auto_substate(AutoExecutionSubState.MOVING_TO_WAYPOINT)

    def _handle_finished(self) -> None:
        self.publish_progress(completed_waypoints=len(self.scan_waypoints))

        if self.requested_command == RequestedCommand.IDLE:
            self.requested_command = RequestedCommand.NONE
            self.current_waypoint_index = 0
            self.pending_manual_target = None
            self.watchdog_thread = None

            with self.motion_status_lock:
                self.motion_status = MotionStatus.IN_PROGRESS

            self.set_state(MovementState.IDLE)

    # ------------------------------------------
    # Main Control Command Handler
    # ------------------------------------------
    def _movement_command_callback(self, msg: String) -> None:
        command = msg.data.strip()
        self.get_logger().info(f'Received movement command: {command}')

        if command == 'EMERGENCY_STOP':
            self.trigger_emergency_stop()
            return

        if self.emergency_stop_latched:
            self.get_logger().error(
                f'Ignoring movement command "{command}" because emergency stop is latched. '
                'Restart launch required.'
            )
            return

        if command == 'MOVE_PRE_SCAN_POSITION':
            self.requested_command = RequestedCommand.MOVE_PRE_SCAN_POSITION

        elif command == 'CONTINUE':
            if (
                self.state == MovementState.AUTO_EXECUTING
                and self.auto_substate == AutoExecutionSubState.MOVING_TO_WAYPOINT
            ):
                self.get_logger().warn(
                    'Ignoring CONTINUE because movement is already in progress.'
                )
                return

            self.requested_command = RequestedCommand.CONTINUE

        elif command == 'PAUSE':
            self.requested_command = RequestedCommand.PAUSE

        elif command == 'IDLE':
            self.requested_command = RequestedCommand.IDLE

        elif command == 'ERROR':
            self.set_state(MovementState.ERROR)

        elif command.startswith('MANUAL_MOVE:'):
            name = command.split(':', 1)[1].strip().lower()
            if name not in self.manual_positions:
                self.get_logger().error(
                    f"Manual position '{name}' not found. "
                    f"Available: {list(self.manual_positions.keys())}"
                )
                return
            self.pending_manual_target = name
            with self.motion_status_lock:
                self.motion_status = MotionStatus.IN_PROGRESS
            self.watchdog_thread = None
            self.set_state(MovementState.MANUAL_EXECUTING)

        else:
            self.get_logger().warn(f'Unknown movement command: {command}')


    # ------------------------------------------
    # Speed Control Handler
    # ------------------------------------------

    def _movement_speed_callback(self, msg: String) -> None:
        text = msg.data.strip()

        # Supports both old format: "medium"
        # and new GUI JSON format: {"size":"large","resolution":"high","speed":"medium"}
        try:
            settings = json.loads(text)
            speed = str(settings.get("speed", self.scan_speed)).strip().lower()
            resolution = str(settings.get("resolution", self.scan_resolution)).strip().lower()
        except json.JSONDecodeError:
            speed = text.lower()
            resolution = self.scan_resolution

        if speed not in self.speed_to_move_time:
            self.get_logger().warn(
                f"Unknown movement speed '{speed}'. Expected low, medium, or high."
            )
        else:
            self.scan_speed = speed
            self.scan_move_time_sec = self.speed_to_move_time[speed]

        if resolution not in self.resolution_to_fraction:
            self.get_logger().warn(
                f"Unknown scan resolution '{resolution}'. Expected low, medium, or high."
            )
        else:
            if resolution != self.scan_resolution:
                self.scan_resolution = resolution
                self.apply_resolution_to_waypoints()

        self.get_logger().info(
            f"Scan settings updated: speed={self.scan_speed}, "
            f"resolution={self.scan_resolution}, "
            f"move_time={self.scan_move_time_sec:.1f}s, "
            f"waypoints={len(self.scan_waypoints)}"
        )

    # ------------------------------------------
    # Pause / resume / progress helpers
    # ------------------------------------------
    def reset_scan_progress(self) -> None:
        self.current_waypoint_index = 0
        self.publish_progress(completed_waypoints=0)

    def publish_progress(self, completed_waypoints: Optional[int] = None) -> None:
        if not self.scan_waypoints:
            percentage = 0.0
        else:
            if completed_waypoints is None:
                completed_waypoints = self.current_waypoint_index
            completed_waypoints = max(0, min(completed_waypoints, len(self.scan_waypoints)))
            percentage = (completed_waypoints / len(self.scan_waypoints)) * 100.0

        msg = Float32()
        msg.data = float(percentage)
        self.progress_pub.publish(msg)

    def pause_motion(self) -> None:
        self.paused_resume_state = self.state
        self.paused_resume_auto_substate = self.auto_substate
        self.watchdog_thread = None

        with self.motion_status_lock:
            self.motion_status = MotionStatus.INTERRUPTED

        self.publish_hold_position()
        self.set_state(MovementState.PAUSED)
        self.get_logger().warn(
            f"Movement paused. Saved waypoint objective "
            f"{self.current_waypoint_index + 1}/{len(self.scan_waypoints) if self.scan_waypoints else 0}."
        )

    def resume_from_pause(self) -> None:
        if self.paused_resume_state == MovementState.AUTO_EXECUTING:
            resume_substate = (
                self.paused_resume_auto_substate
                or AutoExecutionSubState.MOVING_TO_WAYPOINT
            )
            with self.motion_status_lock:
                self.motion_status = MotionStatus.IN_PROGRESS
            self.watchdog_thread = None
            self.set_state(MovementState.AUTO_EXECUTING)
            self.set_auto_substate(resume_substate)
            return

        if self.paused_resume_state == MovementState.MOVE_PRE_SCAN_POSITION:
            with self.motion_status_lock:
                self.motion_status = MotionStatus.IN_PROGRESS
            self.watchdog_thread = None
            self.set_state(MovementState.MOVE_PRE_SCAN_POSITION)
            return

        if self.paused_resume_state == MovementState.PRE_SCAN_POSITION:
            self.set_state(MovementState.PRE_SCAN_POSITION)
            return

        self.get_logger().warn("No saved movement objective to resume from pause.")
        self.set_state(MovementState.IDLE)

    def publish_hold_position(self) -> bool:
        current = self.get_ordered_current_joint_positions()
        if current is None:
            self.get_logger().warn("Could not publish hold position because joints are not ready.")
            return False

        traj = JointTrajectory()
        traj.joint_names = list(self.joint_names)

        point = JointTrajectoryPoint()
        point.positions = list(current)
        point.time_from_start.sec = 1

        traj.points = [point]
        self.trajectory_pub.publish(traj)
        self.get_logger().info(f"Published hold-position target: {current}")
        return True

    def trigger_emergency_stop(self) -> None:
        if self.emergency_stop_latched:
            return

        self.get_logger().error(
            'EMERGENCY STOP received. Halting robot immediately. Restart launch required.'
        )

        self.emergency_stop_latched = True
        self.requested_command = RequestedCommand.NONE
        self.paused_resume_state = None
        self.paused_resume_auto_substate = None
        self.watchdog_thread = None

        with self.motion_status_lock:
            self.motion_status = MotionStatus.INTERRUPTED

        self.publish_hold_position()
        self.set_state(MovementState.ERROR)

    # ------------------------------------------
    # Publishing Status to Main Control
    # ------------------------------------------
    def _publish_status(self) -> None:
        msg = String()

        if self.state == MovementState.MANUAL_EXECUTING:
            msg.data = f'MANUAL_EXECUTING:{self.manual_sub_state.name}'
        elif self.auto_substate is None:
            msg.data = self.state.name
        else:
            msg.data = f'{self.state.name}:{self.auto_substate.name}'

        self.status_pub.publish(msg)

    def _publish_heartbeat(self) -> None:
        self.heartbeat_pub.publish(Empty())


    def _publish_startup_status(self) -> None:
        if self.startup_status_publish_count >= self.startup_status_publish_limit:
            self.startup_status_timer.cancel()
            return

        self._publish_status()
        self.startup_status_publish_count += 1


    # ------------------------------------------
    # Watchdog
    # ------------------------------------------

    def _start_watchdog(
        self,
        target_positions: List[float],
        tolerance: float,
        timeout_sec: float,
    ) -> None:
        with self.motion_status_lock:
            self.motion_status = MotionStatus.IN_PROGRESS

        def _watchdog() -> None:
            start_time = time.monotonic()

            while time.monotonic() - start_time < timeout_sec:
                # Check for external interruption
                if self.state not in (
                    MovementState.MOVE_PRE_SCAN_POSITION,
                    MovementState.AUTO_EXECUTING,
                    MovementState.MANUAL_EXECUTING,
                ):
                    with self.motion_status_lock:
                        self.motion_status = MotionStatus.INTERRUPTED
                    return

                current = self.get_ordered_current_joint_positions()
                if current is None:
                    time.sleep(0.05)
                    continue

                # Check for joint fault — joints not moving at all after 2 seconds
                elapsed = time.monotonic() - start_time
                if elapsed > 2.0:
                    errors_from_start = [
                        abs(c - t) for c, t in zip(current, target_positions)
                    ]
                    if all(err > tolerance * 10 for err in errors_from_start):
                        with self.motion_status_lock:
                            self.motion_status = MotionStatus.FAULT
                        return

                # Check if target reached
                errors = [abs(c - t) for c, t in zip(current, target_positions)]
                if all(err <= tolerance for err in errors):
                    with self.motion_status_lock:
                        self.motion_status = MotionStatus.REACHED
                    return

                time.sleep(0.05)

            with self.motion_status_lock:
                self.motion_status = MotionStatus.TIMEOUT

        self.watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
        self.watchdog_thread.start()

    def get_motion_status(self) -> MotionStatus:
        with self.motion_status_lock:
            return self.motion_status
        

    # ------------------------------------------
    # Joint state, ensures joints have been recieved and updates current joints
    # ------------------------------------------
    def _joint_callback(self, msg: JointState) -> None:
        self.current_joints.update(dict(zip(msg.name, msg.position)))

    def joints_ready(self) -> bool:
        return all(name in self.current_joints for name in self.joint_names)

    def get_ordered_current_joint_positions(self) -> Optional[List[float]]:
        if not self.joints_ready():
            return None
        try:
            return [self.current_joints[name] for name in self.joint_names]
        except KeyError as exc:
            self.get_logger().error(f"Missing joint state for {exc}")
            return None

    # ------------------------------------------
    # Safety
    # ------------------------------------------
    def is_waypoint_safe(self, positions: List[float]) -> bool:
        if len(positions) != len(self.joint_names):
            self.get_logger().error(
                f"Expected {len(self.joint_names)} joints, got {len(positions)}"
            )
            return False
        for joint_name, joint_value in zip(self.joint_names, positions):
            lower, upper = self.safe_joint_limits[joint_name]
            if not (lower <= joint_value <= upper):
                self.get_logger().warn(
                    f"Unsafe joint target rejected: "
                    f"{joint_name}={joint_value:.3f} outside [{lower:.3f}, {upper:.3f}]"
                )
                return False
        return True
    
    def calculate_ur3e_fk(self, q: List[float]) -> Optional[np.ndarray]:
        """
        Basic UR3e forward kinematics using approximate UR3e DH parameters.

        Returns:
            4x4 transformation matrix for the end-effector/tool0 pose.
        """
        if len(q) != 6:
            self.get_logger().error("FK failed: expected 6 joint values.")
            return None

        # Approximate UR3e DH parameters.
        # These are suitable for safety sanity checks, not high-precision calibration.
        d = [0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921]
        a = [0.0, -0.24355, -0.2132, 0.0, 0.0, 0.0]
        alpha = [math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0]

        transform = np.eye(4)

        for i in range(6):
            theta = q[i]
            ct = math.cos(theta)
            st = math.sin(theta)
            ca = math.cos(alpha[i])
            sa = math.sin(alpha[i])

            link_transform = np.array([
                [ct, -st * ca,  st * sa, a[i] * ct],
                [st,  ct * ca, -ct * sa, a[i] * st],
                [0.0,      sa,      ca,      d[i]],
                [0.0,     0.0,     0.0,     1.0],
            ])

            transform = transform @ link_transform

        return transform


    def is_end_effector_pose_safe(self, positions: List[float]) -> bool:
        """
        Checks whether the final end-effector/tool0 position is inside the allowed
        workspace and above the minimum table clearance height.
        """
        if not self.enable_fk_workspace_check:
            return True

        transform = self.calculate_ur3e_fk(positions)
        if transform is None:
            return False

        x = float(transform[0, 3])
        y = float(transform[1, 3])
        z = float(transform[2, 3])

        if not (self.ee_min_x <= x <= self.ee_max_x):
            self.get_logger().error(
                f"Unsafe end-effector X rejected: x={x:.3f} outside "
                f"[{self.ee_min_x:.3f}, {self.ee_max_x:.3f}]"
            )
            return False

        if not (self.ee_min_y <= y <= self.ee_max_y):
            self.get_logger().error(
                f"Unsafe end-effector Y rejected: y={y:.3f} outside "
                f"[{self.ee_min_y:.3f}, {self.ee_max_y:.3f}]"
            )
            return False

        if not (self.ee_min_z <= z <= self.ee_max_z):
            self.get_logger().error(
                f"Unsafe end-effector Z rejected: z={z:.3f} outside "
                f"[{self.ee_min_z:.3f}, {self.ee_max_z:.3f}]"
            )
            return False

        self.get_logger().info(
            f"End-effector FK check passed: x={x:.3f}, y={y:.3f}, z={z:.3f}"
        )

        return True
    
    # ------------------------------------------
    # Take shortest route to goal
    # ------------------------------------------

    def make_nearest_joint_target(self, target_positions: List[float]) -> Optional[List[float]]:
        """
        Convert each target joint angle to the closest equivalent angle
        relative to the robot's current joint state.

        This prevents continuous UR joints from doing unnecessary full rotations.
        """
        current_positions = self.get_ordered_current_joint_positions()
        if current_positions is None:
            self.get_logger().warn("Cannot unwrap target because current joints are not ready.")
            return None

        safe_target = []

        for joint_name, target, current in zip(self.joint_names, target_positions, current_positions):
            nearest = current + math.atan2(
                math.sin(target - current),
                math.cos(target - current)
            )
            safe_target.append(nearest)

            if abs(nearest - target) > 0.01:
                self.get_logger().info(
                    f"Adjusted {joint_name}: raw={target:.3f}, current={current:.3f}, nearest={nearest:.3f}"
                )

        return safe_target
    
    def get_final_joint_target(
        self,
        positions: List[float],
        use_nearest: bool = True,
    ) -> Optional[List[float]]:
        if use_nearest:
            return self.make_nearest_joint_target(positions)
        return list(positions)
   
                                                                                
    # ------------------------------------------
    # Trajectory publishing
    # ------------------------------------------
    def publish_joint_target(
        self,
        positions: List[float],
        move_time_sec: float = 5.0,
        use_nearest: bool = True,
    ) -> bool:

        # Use shortest-angle wrapping only when desired
        if use_nearest:
            final_positions = self.make_nearest_joint_target(positions)

            if final_positions is None:
                self.get_logger().error(
                    "Could not calculate nearest joint target."
                )
                return False
        else:
            final_positions = list(positions)

        # Joint safety check
        if not self.is_waypoint_safe(final_positions):
            self.get_logger().error("Rejected unsafe joint target.")
            return False

        # End-effector workspace/table height safety check
        if not self.is_end_effector_pose_safe(final_positions):
            self.get_logger().error(
                "Rejected unsafe joint target because end-effector pose is outside the safe workspace."
            )
            return False

        traj = JointTrajectory()
        traj.joint_names = list(self.joint_names)

        point = JointTrajectoryPoint()
        point.positions = list(final_positions)

        seconds = int(move_time_sec)
        nanoseconds = int((move_time_sec - seconds) * 1_000_000_000)

        point.time_from_start.sec = seconds
        point.time_from_start.nanosec = nanoseconds

        traj.points = [point]

        self.trajectory_pub.publish(traj)

        self.get_logger().info(
            f"Published joint target: {final_positions} "
            f"over {move_time_sec:.2f}s "
            f"(use_nearest={use_nearest})"
        )

        return True

    
def main(args=None) -> None:
    rclpy.init(args=args)
    node = MovementNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
