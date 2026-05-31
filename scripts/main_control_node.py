#!/usr/bin/env python3
# 41069 Robotics Studio 2
# Authors: Jayden Hazell (24953364), Lachlan Partridge (24964557), Tomas Klimes (24773804)
# Top-level state machine that orchestrates movement and scan nodes, monitors heartbeats, and handles emergency stops.
"""
Main control node for the UR3e scanning system.

Supervises the movement and scan subsystems using a hierarchical state machine.
Forwards GUI commands, monitors heartbeat topics from all required nodes, and
triggers an emergency stop if any node becomes unresponsive.
"""

from __future__ import annotations

import yaml
from enum import Enum, auto
from pathlib import Path

import rclpy
import json
from rclpy.time import Time
from rclpy.node import Node
from std_msgs.msg import Bool, String, Empty
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from dataclasses import dataclass

BOOT_NODES_TIMEOUT_SEC: float = 5.0


class MainState(Enum):
    BOOTING = auto()
    IDLE = auto()
    MOVE_PRE_SCAN_POSITION = auto()
    PRE_SCAN_POSITION = auto()
    AUTO_EXECUTION = auto()
    MANUAL_EXECUTION = auto()
    ERROR = auto()
    EMERGENCY_STOP_SHUTDOWN = auto()
    
class AutoScanSubState(Enum):
    CONFIGURING_AUTO_SCAN = auto()
    MOVING_TO_WAYPOINT = auto()
    CAPTURING_SCAN = auto()
    PAUSED = auto()
    RECONSTRUCTING_SCAN = auto()
    SCAN_FAILED = auto()
    SCAN_COMPLETE = auto()

class PendingCommandFromGUI(Enum):
    NONE = auto()
    MOVE_PRE_SCAN_POSITION = auto()
    START = auto()
    PAUSE = auto()

class MovementMainState(Enum):
    BOOTING = auto()
    IDLE = auto()
    MOVE_PRE_SCAN_POSITION = auto()
    PRE_SCAN_POSITION = auto()
    AUTO_EXECUTING = auto()
    MANUAL_EXECUTING = auto()
    PAUSED = auto()
    ERROR = auto()
    UNKNOWN = auto()

class MovementAutoSubState(Enum):
    NONE = auto()
    MOVING_TO_WAYPOINT = auto()
    WAITING_FOR_SCAN = auto()
    FINISHED = auto()

class ManualMovementSubState(Enum):
    NONE = auto()
    IDLE = auto()
    MOVING_TO_WAYPOINT = auto()
    FINISHED = auto()

class ScanMainState(Enum):
    BOOTING           = auto()
    IDLE              = auto()
    SCANNING          = auto()
    FINISHED_SCANNING = auto()
    RECONSTRUCTING    = auto()
    ERROR             = auto()
    UNKNOWN           = auto()

@dataclass
class MovementStatus:
    main: MovementMainState = MovementMainState.UNKNOWN
    auto_sub: MovementAutoSubState = MovementAutoSubState.NONE
    manual_sub: ManualMovementSubState = ManualMovementSubState.NONE
    raw: str = ""  # raw string from movement, for debugging

@dataclass
class ScanStatus:
    main: ScanMainState = ScanMainState.UNKNOWN
    raw: str = ""  # raw string from scan, for debugging

class MainControlNode(Node):
    def __init__(self) -> None:
        super().__init__('main_control_node')

        # ------------------------------------------
        # Parameters
        # ------------------------------------------

        self.declare_parameter('use_fake_hardware', False)
        self.use_fake_hardware = bool(self.get_parameter('use_fake_hardware').value
        )

        # ------------------------------------------
        # Variables
        # ------------------------------------------

        self.state: MainState = MainState.BOOTING  # Current top-level state
        self.auto_substate: AutoScanSubState | None = None  # Auto mode sub-state
        self.boot_start_time = self.get_clock().now()
        self.heartbeat_timeout_sec: float = 3.0  # Timeout before watchdog trips
        self.pending_command: PendingCommandFromGUI = PendingCommandFromGUI.NONE  # What the user requested
        
        self.movement_status = MovementStatus()
        self.scan_status = ScanStatus()

        self.movement_command_sent: bool = False
        self.scan_command_sent: bool = False
        self.movement_seen_active: bool = False
        self.pending_manual_target: str | None = None
        self.scan_reset_sent: bool = False
        self._reconstruct_sent: bool = False
        self._seen_scan_reconstructing: bool = False
        self.manual_command_sent: bool = False
        self._manual_pre_scan_sent: bool = False

        self.error_message_sent: bool = False

        self.emergency_stop_latched: bool = False

        # Nodes whose missing heartbeat caused the boot timeout ERROR.
        # If all of them subsequently send a heartbeat, the system recovers to BOOTING.
        self._boot_error_nodes: set[str] = set()

        # Drop-off zone coordinates (loaded from workspace.yaml)
        _ws_cfg_path = Path(__file__).resolve().parent.parent / 'config' / 'workspace.yaml'
        _ws_cfg: dict = {}
        if _ws_cfg_path.exists():
            _ws_cfg = yaml.safe_load(_ws_cfg_path.read_text()) or {}
        _doz = _ws_cfg.get('drop_off_zone', {})
        self._drop_off_x: float = float(_doz.get('x', 0.0))
        self._drop_off_y: float = float(_doz.get('y', 0.0))
        self._drop_off_z: float = float(_doz.get('z', 0.0))

        self.required_heartbeats: dict[str, bool] = {
            'gui': True,  # GUI must always be alive
            'movement': True,  # Movement node must always be alive
            'scan': True,
        }

        self.last_heartbeat_time: dict[str, Time | None] = {
            'gui': None,
            'movement': None,
            'scan': None,
        }

        # ------------------------------------------
        # Publishers
        # ------------------------------------------

        self.movement_command_pub = self.create_publisher(
            String,
            '/control/movement_command',
            10,
        )
        self.scan_command_pub = self.create_publisher(
            String,
            '/control/scan_command',
            10,
        )
        self.system_log_pub = self.create_publisher(
            String,
            '/system/log',
            10,
        )
        self.system_status_pub = self.create_publisher(
            String,
            '/control/status',
            10,
        )
        self.movement_speed_pub = self.create_publisher(
            String,
            '/control/movement_speed',
            10,
        )
        self.drop_off_marker_pub = self.create_publisher(
            Marker,
            '/visualization/drop_off_zone',
            10,
        )

        # ------------------------------------------
        # Subscribers
        # ------------------------------------------

        #Heartbeat subscriptions
        self.gui_heartbeat_sub = self.create_subscription(
            Empty,
            '/heartbeat/gui',
            self._gui_heartbeat_callback,
            10,
        )
        self.movement_heartbeat_sub = self.create_subscription(
            Empty,
            '/heartbeat/movement',
            self._movement_heartbeat_callback,
            10,
        )
        self.scan_heartbeat_sub = self.create_subscription(
            Empty,
            '/heartbeat/scan',
            self._scan_heartbeat_callback,
            10,
        )

        #GUI Subscriptions
        self.gui_move_to_pre_scan_position_sub = self.create_subscription(
            Bool,
            '/gui/move_to_pre_scan_position',
            self._gui_move_to_pre_scan_position_callback,
            10,
        )
        self.gui_start_scan_sub = self.create_subscription(
            Bool,
            '/gui/start_scan',
            self._gui_start_scan_callback,
            10,
        )
        self.gui_pause_scan_sub = self.create_subscription(
            Bool,
            '/gui/pause_scan',
            self._pause_scan_callback,
            10,
        )
        self.gui_emergency_stop_sub = self.create_subscription(
            Bool,
            '/gui/emergency_stop',
            self._emergency_stop_callback,
            10,
        )

        self.gui_settings_sub = self.create_subscription(
            String,
            "/gui/settings",
            self._gui_settings_callback,
            10,
        )
        self.gui_viewpoint_sub = self.create_subscription(
            String,
            '/gui/viewpoint',
            self._viewpoint_callback,
            10,
        )

        #Movement Subscriptions
        self.movement_status_sub = self.create_subscription(
            String,
            '/movement/status',
            self._movement_status_callback,
            10,
        )

        #Scan Subscriptions
        self.scan_status_sub = self.create_subscription(
            String,
            '/scan/status',
            self._scan_status_callback,
            10,
        )

        # ------------------------------------------
        # Timers
        # ------------------------------------------
        self.state_machine_timer = self.create_timer(
            0.1,
            self.state_machine_tick,
        )
        self.drop_off_marker_timer = self.create_timer(
            1.0,
            self._publish_drop_off_marker,
        )

        # ------------------------------------------
        # Logging
        # ------------------------------------------
        self._log('INFO', f'main_control started. Initial state: {self.state.name}')
        self._publish_status()

    # ------------------------------------------
    # GUI Log + Status Helpers
    # ------------------------------------------

    def _log(self, level: str, message: str) -> None:
        """Write to the terminal logger and forward the message to the GUI log box."""
        level_upper = level.upper()
        formatted = f'[ERROR] {message}' if level_upper == 'ERROR' else message

        if level_upper == 'ERROR':
            self.get_logger().error(message)
        elif level_upper == 'WARN':
            self.get_logger().warn(message)
        else:
            self.get_logger().info(message)

        msg = String()
        msg.data = formatted
        self.system_log_pub.publish(msg)

    def _publish_status(self) -> None:
        """Publish the current main state (and sub-state if in AUTO_EXECUTION) to /system/status."""
        if self.state == MainState.AUTO_EXECUTION and self.auto_substate is not None:
            status = f'AUTO: {self.auto_substate.name}'
        else:
            status = self.state.name

        msg = String()
        msg.data = status
        self.system_status_pub.publish(msg)

    # ------------------------------------------
    # State Management
    # ------------------------------------------

    def set_state(self, new_state: MainState) -> None:
        if self.state == new_state:
            return

        self._log('INFO', f'State: {self.state.name} -> {new_state.name}')

        self.scan_command_sent = False
        self.movement_command_sent = False
        self.scan_reset_sent = False
        self._reconstruct_sent = False
        self._seen_scan_reconstructing = False
        self.manual_command_sent = False
        self._manual_pre_scan_sent = False

        self.state = new_state

        if new_state != MainState.AUTO_EXECUTION:
            self.auto_substate = None

        self._publish_status()

    def set_auto_substate(self, new_substate: AutoScanSubState | None) -> None:
        if self.state != MainState.AUTO_EXECUTION:
            self._log('WARN', 'set_auto_substate called while not in AUTO_EXECUTION — ignored.')
            return

        if self.auto_substate == new_substate:
            return

        previous = self.auto_substate.name if self.auto_substate else 'None'
        current = new_substate.name if new_substate else 'None'

        self._log('INFO', f'AUTO sub-state: {previous} -> {current}')
        self.auto_substate = new_substate
        self.scan_command_sent = False
        self.movement_command_sent = False
        self._publish_status()

    # ------------------------------------------
    # State Machine Tick
    # ------------------------------------------

    def state_machine_tick(self) -> None:
        if self.movement_status.main == MovementMainState.ERROR or self.scan_status.main == ScanMainState.ERROR:
            if not self.error_message_sent:
                self.error_message_sent = True
                self._trigger_emergency_stop('One or more nodes in ERROR.')
            return
        
        self._check_heartbeat_watchdog()

        if self.state == MainState.BOOTING:
            self._handle_booting()
        elif self.state == MainState.IDLE:
            self._handle_idle()
        elif self.state == MainState.MOVE_PRE_SCAN_POSITION:
            self._handle_move_pre_scan_position()
        elif self.state == MainState.PRE_SCAN_POSITION:
            self._handle_pre_scan_position()
        elif self.state == MainState.AUTO_EXECUTION:
            self._handle_auto_execution()
        elif self.state == MainState.MANUAL_EXECUTION:
            self._handle_manual_execution()
        elif self.state == MainState.ERROR:
            self._handle_error()
        elif self.state == MainState.EMERGENCY_STOP_SHUTDOWN:
            self._handle_emergency_stop_shutdown()

    # ------------------------------------------
    # Top-Level State Handlers
    # ------------------------------------------

    def _handle_booting(self) -> None:
        if not self._all_required_heartbeats_received():
            elapsed = (self.get_clock().now() - self.boot_start_time).nanoseconds / 1e9
            if elapsed > BOOT_NODES_TIMEOUT_SEC:
                missing = [
                    name for name, required in self.required_heartbeats.items()
                    if required and self.last_heartbeat_time[name] is None
                ]
                self._boot_error_nodes = set(missing)
                self._log(
                    'ERROR',
                    f'Boot timeout: no heartbeat from {missing} after '
                    f'{BOOT_NODES_TIMEOUT_SEC:.0f}s. Check that all required nodes started.',
                )
                self.set_state(MainState.ERROR)
            return

        if (self.movement_status.main == MovementMainState.IDLE and self.scan_status.main == ScanMainState.IDLE):
            self._log('INFO', 'All required nodes IDLE — boot complete.')
            self.set_state(MainState.IDLE)

    def _handle_idle(self) -> None:
        if self.pending_command == PendingCommandFromGUI.MOVE_PRE_SCAN_POSITION:
            self.pending_command = PendingCommandFromGUI.NONE
            self.set_state(MainState.MOVE_PRE_SCAN_POSITION)
        if self.pending_command == PendingCommandFromGUI.START:
            self.set_state(MainState.MOVE_PRE_SCAN_POSITION)
            
    def _handle_move_pre_scan_position(self) -> None:
        if self.pending_command == PendingCommandFromGUI.PAUSE:
            self.pending_command = PendingCommandFromGUI.NONE
            self._publish_movement_command('IDLE')
            self.set_state(MainState.IDLE)
            return

        if not self.movement_command_sent:
            self._publish_movement_command('MOVE_PRE_SCAN_POSITION')
            self.movement_command_sent = True

        if (self.movement_status.main == MovementMainState.PRE_SCAN_POSITION):
            self.set_state(MainState.PRE_SCAN_POSITION)
        
    def _handle_pre_scan_position(self) -> None:
        if not self.scan_command_sent:
            self._publish_scan_command('RESET')
            self.scan_command_sent = True

        if self.pending_command == PendingCommandFromGUI.START:
            self.pending_command = PendingCommandFromGUI.NONE
            self.set_state(MainState.AUTO_EXECUTION)
            # Skip CONFIGURING_AUTO_SCAN and begin moving immediately; configuration is handled via settings.
            self.set_auto_substate(AutoScanSubState.MOVING_TO_WAYPOINT)

    def _handle_auto_execution(self) -> None:
        if self.auto_substate == AutoScanSubState.CONFIGURING_AUTO_SCAN:
            self._handle_configuring_auto_scan()
        elif self.auto_substate == AutoScanSubState.MOVING_TO_WAYPOINT:
            self._handle_moving_to_waypoint()
        elif self.auto_substate == AutoScanSubState.CAPTURING_SCAN:
            self._handle_capturing_scan()
        elif self.auto_substate == AutoScanSubState.PAUSED:
            self._handle_scan_paused()
        elif self.auto_substate == AutoScanSubState.RECONSTRUCTING_SCAN:
            self._handle_reconstructing_scan()
        elif self.auto_substate == AutoScanSubState.SCAN_FAILED:
            self._handle_scan_failed()
        elif self.auto_substate == AutoScanSubState.SCAN_COMPLETE:
            self._handle_scan_complete()

    def _handle_manual_execution(self) -> None:
        if self.pending_manual_target is None:
            self._log('ERROR', 'MANUAL_EXECUTION with no target. Returning to IDLE.')
            self.set_state(MainState.IDLE)
            return

        if self.pending_command == PendingCommandFromGUI.PAUSE:
            self.pending_command = PendingCommandFromGUI.NONE
            self._publish_movement_command('IDLE')
            self.pending_manual_target = None
            self.set_state(MainState.IDLE)
            return

        # Step 1: Reset scan node so it is ready for the next scan sequence
        if not self.scan_reset_sent:
            self._publish_scan_command('RESET')
            self.scan_reset_sent = True
            return

        # Step 2: Wait for scan IDLE, then move to pre-scan position first.
        # This ensures the robot is always in a known safe configuration
        # before travelling to any manual viewpoint, preventing unexpected
        # joint-space paths that could cause collisions.
        # Skip the command if already at pre-scan position — sending it
        # would leave a stale MOVE_PRE_SCAN_POSITION in movement_node's
        # requested_command queue, which causes the robot to return there
        # automatically once the manual move finishes.
        if not self._manual_pre_scan_sent:
            if self.scan_status.main != ScanMainState.IDLE:
                return
            if self.movement_status.main != MovementMainState.PRE_SCAN_POSITION:
                self._publish_movement_command('MOVE_PRE_SCAN_POSITION')
            self._manual_pre_scan_sent = True
            return

        # Step 3/4: Before sending the manual move, require pre-scan position first.
        if not self.manual_command_sent:
            if self.movement_status.main != MovementMainState.PRE_SCAN_POSITION:
                return

            self._publish_movement_command(f'MANUAL_MOVE:{self.pending_manual_target}')
            self.manual_command_sent = True
            return

        # Step 5: After manual move has been sent, wait for it to finish.
        if self.manual_command_sent and self.movement_status.main == MovementMainState.IDLE:
            self._log('INFO', 'Manual movement complete. Returning main control to IDLE.')
            self.pending_manual_target = None
            self.scan_reset_sent = False
            self.manual_command_sent = False
            self.movement_command_sent = False
            self._manual_pre_scan_sent = False
            self.set_state(MainState.IDLE)
            return

    def _handle_error(self) -> None:
        # If this error was caused by a boot heartbeat timeout, check whether
        # all of the missing nodes have since sent a heartbeat.  If so, recover
        # by restarting the boot sequence.
        if self._boot_error_nodes:
            recovered = all(
                self.last_heartbeat_time[name] is not None
                for name in self._boot_error_nodes
            )
            if recovered:
                self._log(
                    'INFO',
                    f'Heartbeat(s) received from {sorted(self._boot_error_nodes)} — '
                    f'recovering from boot timeout error. Restarting boot sequence.',
                )
                self._boot_error_nodes.clear()
                self.boot_start_time = self.get_clock().now()
                self.error_message_sent = False
                self.set_state(MainState.BOOTING)

    def _handle_emergency_stop_shutdown(self) -> None:
        if not self.emergency_stop_latched:
            self._log(
                'ERROR',
                'EMERGENCY STOP active. System halted — restart required.',
            )
            self.emergency_stop_latched = True

        # Remain in halted state and do nothing else.
        return
    
    def _trigger_emergency_stop(self, reason: str) -> None:
        if self.emergency_stop_latched:
            return

        self.emergency_stop_latched = True
        self.pending_command = PendingCommandFromGUI.NONE

        self._log('ERROR', f'EMERGENCY STOP triggered: {reason}')

        # Tell both subsystems to stop immediately
        self._publish_movement_command('EMERGENCY_STOP')
        self._publish_scan_command('EMERGENCY_STOP')

        self.set_state(MainState.EMERGENCY_STOP_SHUTDOWN)

    # ------------------------------------------
    # Auto Sub-State Handlers
    # ------------------------------------------

    def _handle_configuring_auto_scan(self) -> None:
        pass

    def _handle_moving_to_waypoint(self) -> None:
        if not self.movement_command_sent:
            self.movement_seen_active = False  # Reset: don't accept stale WAITING_FOR_SCAN
            self._publish_movement_command('CONTINUE')
            self.movement_command_sent = True
            return

        if self.movement_status.auto_sub == MovementAutoSubState.WAITING_FOR_SCAN:
            if not self.movement_seen_active:
                return  # Stale status from previous waypoint — wait for fresh ACTIVE
            self.set_auto_substate(AutoScanSubState.CAPTURING_SCAN)
            return

        if self.movement_status.auto_sub == MovementAutoSubState.FINISHED:
            self.set_auto_substate(AutoScanSubState.RECONSTRUCTING_SCAN)
            return

        # Movement is actively travelling — record so we accept the next WAITING_FOR_SCAN
        self.movement_seen_active = True

    def _handle_capturing_scan(self) -> None:
        if not self.scan_command_sent:
            self._publish_scan_command('SCAN')
            self.scan_command_sent = True
        
        if self.scan_status.main == ScanMainState.FINISHED_SCANNING:
            self.set_auto_substate(AutoScanSubState.MOVING_TO_WAYPOINT)

    def _handle_scan_paused(self) -> None:
        if self.pending_command == PendingCommandFromGUI.MOVE_PRE_SCAN_POSITION:
            self.pending_command = PendingCommandFromGUI.NONE
            self._publish_movement_command('MOVE_PRE_SCAN_POSITION')
            self.set_state(MainState.MOVE_PRE_SCAN_POSITION)
            return
        
        if not self.movement_command_sent:
            self._publish_movement_command('PAUSE')
            self.movement_command_sent = True

        if self.pending_command == PendingCommandFromGUI.START:
            self.pending_command = PendingCommandFromGUI.NONE
            self.set_auto_substate(AutoScanSubState.MOVING_TO_WAYPOINT)

    def _handle_reconstructing_scan(self) -> None:
        # Step 1: send RESET to scan_node
        if not self.scan_reset_sent:
            self._publish_scan_command('RESET')
            self.scan_reset_sent = True
            return

        # Step 2: wait for scan_node to acknowledge reset (go IDLE)
        if not self._reconstruct_sent and self.scan_status.main != ScanMainState.IDLE:
            return

        # Step 3: fire reconstruction — scan_node transitions to RECONSTRUCTING
        if not self._reconstruct_sent:
            self._publish_scan_command('RECONSTRUCT')
            self._reconstruct_sent = True
            self._publish_movement_command('IDLE')
            return

        # Step 4: wait for scan_node to report RECONSTRUCTING at least once,
        # then wait for it to return to IDLE (reconstruction + meshing complete)
        if self.scan_status.main == ScanMainState.RECONSTRUCTING:
            self._seen_scan_reconstructing = True
            return
        if not self._seen_scan_reconstructing:
            return  # haven't seen RECONSTRUCTING yet — wait for scan_node to pick it up
        if self.scan_status.main != ScanMainState.IDLE:
            return

        self._log('INFO', 'Reconstruction and meshing complete.')
        self.pending_command = PendingCommandFromGUI.NONE
        self.set_auto_substate(AutoScanSubState.SCAN_COMPLETE)



    def _handle_scan_failed(self) -> None:
        pass

    def _handle_scan_complete(self) -> None:
        # Step 1: return robot to pre-scan position before going idle
        if not self.movement_command_sent:
            self._log('INFO', 'Scan complete — returning to pre-scan position.')
            self._publish_movement_command('MOVE_PRE_SCAN_POSITION')
            self.movement_command_sent = True
            return

        if self.movement_status.main != MovementMainState.PRE_SCAN_POSITION:
            return

        self._log('INFO', 'Pre-scan position reached. System is IDLE and ready for the next command.')

        self.pending_command = PendingCommandFromGUI.NONE
        self.movement_command_sent = False
        self.scan_command_sent = False
        self.scan_reset_sent = False
        self._reconstruct_sent = False
        self.manual_command_sent = False
        self.pending_manual_target = None

        self.set_state(MainState.IDLE)

    # ------------------------------------------
    # Scan Publisher and Status Reciever
    # ------------------------------------------
    def _publish_movement_command(self, command: str) -> None:
        msg = String()
        msg.data = command
        self.movement_command_pub.publish(msg)
        self._log('INFO', f'movement_node <- {command}')

    def _movement_status_callback(self, msg: String) -> None:
        text = msg.data
        parts = text.split(':')
        main_raw = parts[0]
        sub_raw = parts[1] if len(parts) > 1 else None

        main_map = {s.name: s for s in MovementMainState if s != MovementMainState.UNKNOWN}
        main_state = main_map.get(main_raw, MovementMainState.UNKNOWN)
        if main_state == MovementMainState.UNKNOWN:
            self._log('WARN', f'movement_node: unknown state "{main_raw}"')

        sub_map = {s.name: s for s in MovementAutoSubState if s != MovementAutoSubState.NONE}
        if main_state == MovementMainState.AUTO_EXECUTING and sub_raw is not None:
            sub_state = sub_map.get(sub_raw, MovementAutoSubState.NONE)
            if sub_state == MovementAutoSubState.NONE:
                self._log('WARN', f'movement_node: unknown AUTO sub-state "{sub_raw}"')
        else:
            sub_state = MovementAutoSubState.NONE

        manual_sub_map = {s.name: s for s in ManualMovementSubState if s != ManualMovementSubState.NONE}
        if main_state == MovementMainState.MANUAL_EXECUTING and sub_raw is not None:
            manual_sub_state = manual_sub_map.get(sub_raw, ManualMovementSubState.NONE)
        else:
            manual_sub_state = ManualMovementSubState.NONE

        prev_main = self.movement_status.main
        prev_sub  = self.movement_status.auto_sub
        prev_manual_sub = self.movement_status.manual_sub

        self.movement_status = MovementStatus(
            main=main_state,
            auto_sub=sub_state,
            manual_sub=manual_sub_state,
            raw=text,
        )

        if main_state != prev_main or sub_state != prev_sub or manual_sub_state != prev_manual_sub:
            if main_state == MovementMainState.MANUAL_EXECUTING and manual_sub_state != ManualMovementSubState.NONE:
                sub_str = f':{manual_sub_state.name}'
            else:
                sub_str = f':{sub_state.name}' if sub_state != MovementAutoSubState.NONE else ''
            self._log('INFO', f'movement_node: {main_state.name}{sub_str}')

    # ------------------------------------------
    # Scan Publisher and Status Reciever
    # ------------------------------------------
    def _publish_scan_command(self, command: str) -> None:
        msg = String()
        msg.data = command
        self.scan_command_pub.publish(msg)
        self._log('INFO', f'scan_node <- {command}')

    def _scan_status_callback(self, msg: String) -> None:
        status_map = {s.name: s for s in ScanMainState if s != ScanMainState.UNKNOWN}
        main_state = status_map.get(msg.data, ScanMainState.UNKNOWN)

        if main_state == ScanMainState.UNKNOWN:
            self._log('WARN', f'scan_node: unknown state "{msg.data}"')

        prev_main = self.scan_status.main
        self.scan_status = ScanStatus(main=main_state, raw=msg.data)

        # Set flag here (not just in the tick) so that a fast RECONSTRUCTING→IDLE
        # transition (< 100ms) is never missed between tick invocations.
        if main_state == ScanMainState.RECONSTRUCTING:
            self._seen_scan_reconstructing = True

        if main_state != prev_main:
            self._log('INFO', f'scan_node: {main_state.name}')

    # ------------------------------------------
    # GUI Callbacks
    # ------------------------------------------
    def _gui_move_to_pre_scan_position_callback(self, msg: Bool) -> None:
        if not msg.data:
            return

        if self.emergency_stop_latched:
            self._log(
                'WARN',
                'Move to pre-scan position ignored: emergency stop is latched. Restart required.'
            )
            return

        can_move_home = (
            self.state == MainState.IDLE
            or self.state == MainState.PRE_SCAN_POSITION
            or (
                self.state == MainState.AUTO_EXECUTION
                and self.auto_substate == AutoScanSubState.PAUSED
            )
        )

        if can_move_home:
            self._log('INFO', 'GUI: move to pre-scan position requested.')
            self.pending_command = PendingCommandFromGUI.MOVE_PRE_SCAN_POSITION
        else:
            self._log(
                'WARN',
                f'Move to pre-scan position requested but not in a valid state to move. '
                f'Ignoring. Current state = {self.state.name}, '
                f'auto_substate = {self.auto_substate.name if self.auto_substate else "None"}'
            )
                

    def _gui_start_scan_callback(self, msg: Bool) -> None:
        if msg.data:
            if self.state in [MainState.IDLE, MainState.PRE_SCAN_POSITION, MainState.MOVE_PRE_SCAN_POSITION, MainState.AUTO_EXECUTION]:
                self._log('INFO', 'GUI: start scan requested.')
                self.pending_command = PendingCommandFromGUI.START
            else:
                self.get_logger().warn(
                    f'Start scan requested but not in a valid state to start. '
                    f'Ignoring. Current state = {self.state.name}'
                )

    def _pause_scan_callback(self, msg: Bool) -> None:
        if not msg.data:
            return
        if self.state == MainState.AUTO_EXECUTION:
            self._log('INFO', 'GUI: pause requested.')
            self.set_auto_substate(AutoScanSubState.PAUSED)
        elif self.state == MainState.MANUAL_EXECUTION:
            self._log('INFO', 'GUI: pause requested during manual execution — cancelling move.')
            self.pending_command = PendingCommandFromGUI.PAUSE
        elif self.state == MainState.MOVE_PRE_SCAN_POSITION:
            self._log('INFO', 'GUI: pause requested during move to pre-scan position — cancelling move.')
            self.pending_command = PendingCommandFromGUI.PAUSE
        else:
            self.get_logger().warn(
                f'Pause scan requested but not currently scanning. '
                f'Ignoring. Current state = {self.state.name}'
            )

    def _emergency_stop_callback(self, msg: Bool) -> None:
        if not msg.data:
            return

        self._trigger_emergency_stop('GUI emergency stop button pressed.')


    def _gui_settings_callback(self, msg: String) -> None:
        try:
            settings = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn(f"Invalid GUI settings JSON: {msg.data}")
            return

        speed = str(settings.get("speed", "")).strip().lower()

        if speed not in ("low", "medium", "high"):
            self.get_logger().warn(f"Invalid GUI speed setting: {speed}")
            return

        settings_msg = String()
        settings_msg.data = json.dumps(settings)
        self.movement_speed_pub.publish(settings_msg)

        self.get_logger().info(f"Forwarded scan settings: {settings_msg.data}")

    def _viewpoint_callback(self, msg: String) -> None:
        name = msg.data.strip().lower()
        if not name:
            return
        if self.emergency_stop_latched:
            self._log('WARN', 'Viewpoint move ignored: emergency stop latched.')
            return

        can_trigger = (
            self.state == MainState.IDLE
            or self.state == MainState.PRE_SCAN_POSITION
            or (
                self.state == MainState.AUTO_EXECUTION
                and self.auto_substate == AutoScanSubState.PAUSED
            )
        )

        if not can_trigger:
            self._log(
                'WARN',
                f'Viewpoint move to "{name}" ignored: not in a valid state '
                f'(state={self.state.name}).',
            )
            return

        if self.state != MainState.IDLE:
            self._publish_movement_command('IDLE')

        self._log('INFO', f'GUI: manual viewpoint move to "{name}" requested.')
        self.pending_manual_target = name
        self.set_state(MainState.MANUAL_EXECUTION)

    # ------------------------------------------
    # Heartbeat + Status Callbacks
    # ------------------------------------------
    def _gui_heartbeat_callback(self, _msg: Empty) -> None:
        self.last_heartbeat_time['gui'] = self.get_clock().now()

    def _movement_heartbeat_callback(self, _msg: Empty) -> None:
        self.last_heartbeat_time['movement'] = self.get_clock().now()

    def _scan_heartbeat_callback(self, _msg: Empty) -> None:
        self.last_heartbeat_time['scan'] = self.get_clock().now()

    def _all_required_heartbeats_received(self) -> bool:
        for node_name, required in self.required_heartbeats.items():
            if required and self.last_heartbeat_time[node_name] is None:
                return False
        return True

    def _heartbeat_timed_out(self, node_name: str) -> bool:
        last_time = self.last_heartbeat_time[node_name]
        if last_time is None:
            return False

        elapsed = (self.get_clock().now() - last_time).nanoseconds / 1e9
        return elapsed > self.heartbeat_timeout_sec

    def _check_heartbeat_watchdog(self) -> None:
        if not self._all_required_heartbeats_received():
            return

        for node_name, required in self.required_heartbeats.items():
            if not required:
                continue

            if self._heartbeat_timed_out(node_name):
                self._trigger_emergency_stop(
                    f'Heartbeat lost: {node_name} (>{self.heartbeat_timeout_sec:.0f}s).'
                )
                return


    # ------------------------------------------
    # RViz Visualisation
    # ------------------------------------------

    def _publish_drop_off_marker(self) -> None:
        """Publish a 15 cm × 15 cm × 15 cm green wireframe box at the drop-off zone."""
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'drop_off_zone'
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.005  # line width in metres
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime.sec = 2
        marker.lifetime.nanosec = 0

        cx = self._drop_off_x
        cy = self._drop_off_y
        cz = self._drop_off_z + 0.075  # centre; base sits at _drop_off_z

        h = 0.075  # half-size: 7.5 cm -> 15 cm total

        def pt(x: float, y: float, z: float) -> Point:
            p = Point()
            p.x = x
            p.y = y
            p.z = z
            return p

        corners = [
            pt(cx - h, cy - h, cz - h),  # 0 bottom
            pt(cx + h, cy - h, cz - h),  # 1
            pt(cx + h, cy + h, cz - h),  # 2
            pt(cx - h, cy + h, cz - h),  # 3
            pt(cx - h, cy - h, cz + h),  # 4 top
            pt(cx + h, cy - h, cz + h),  # 5
            pt(cx + h, cy + h, cz + h),  # 6
            pt(cx - h, cy + h, cz + h),  # 7
        ]

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
        ]

        for a, b in edges:
            marker.points.append(corners[a])
            marker.points.append(corners[b])

        self.drop_off_marker_pub.publish(marker)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MainControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()