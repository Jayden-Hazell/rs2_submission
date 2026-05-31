#!/usr/bin/env python3
# 41069 Robotics Studio 2
# Authors: Jayden Hazell (24953364), Lachlan Partridge (24964557), Tomas Klimes (24773804)
# Interactive ROS2 node for capturing hand-eye calibration samples via keyboard input.
"""
02_collect_samples.py

Interactive ROS2 node for collecting hand-eye calibration samples.

For each robot pose you move to manually (via freedrive or teach pendant):
  SPACE  -- capture sample (only accepted if board is detected with >= 6 corners)
  D      -- discard the most recent sample
  Q      -- save all samples to calib_samples.json and quit

Requires (all already running before starting this script):
  - UR robot driver (publishes TF: base_link -> tool0)
  - RealSense driver (publishes /camera/d435i/color/image_raw and
                      TF: d435i_link -> d435i_color_optical_frame)

Usage:
    cd /home/jayden/ros2_ws/src/rs2
    source /opt/ros/humble/setup.bash
    source install/setup.bash
    python3 scripts/calibration/02_collect_samples.py

Output:
    scripts/calibration/calib_samples.json

Pose diversity guidance (aim for 20 samples):
  - Board must be visible and ~30-70 cm from the camera in every sample.
  - Vary wrist_1, wrist_2, wrist_3 significantly between samples.
  - Include at least 3 clearly different tool orientations
    (e.g. camera tilted 0°, 30°, 60° from its "straight-on" view of the board).
  - Include samples from different pan positions (camera left/center/right of board).
  - Do NOT only rotate about a single axis — coplanar poses produce poor results.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener, TransformException  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Parameters (must match 01_generate_board.py exactly)
# ---------------------------------------------------------------------------

BOARD_PARAMS_FILE = Path(__file__).parent / 'board_params.yaml'
OUTPUT_FILE       = Path(__file__).parent / 'calib_samples.json'

ROBOT_BASE_FRAME    = 'base_link'
ROBOT_EE_FRAME      = 'tool0'
CAMERA_LINK_FRAME   = 'd435i_link'
CAMERA_OPTICAL_FRAME = 'd435i_color_optical_frame'

COLOR_TOPIC    = '/camera/d435i/color/image_raw'
CAMINFO_TOPIC  = '/camera/d435i/color/camera_info'

MIN_CORNERS         = 6   # reject sample if fewer ChArUco corners detected
WARN_CORNERS        = 12  # show warning colour below this count (but still accept)
TF_TIMEOUT_SEC      = 1
STARTUP_TIMEOUT_SEC = 15.0

# ---------------------------------------------------------------------------
# OpenCV version detection (see 01_generate_board.py for rationale)
# ---------------------------------------------------------------------------
_NEW_API = hasattr(cv2.aruco, 'ArucoDetector')


def _quaternion_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


def _make_aruco_objects(params: dict):
    """Create aruco dictionary and board from loaded params."""
    dict_id = int(params['aruco_dict_id'])
    sq_x = int(params['squares_x'])
    sq_y = int(params['squares_y'])
    sq_len = float(params['square_length_m'])
    mk_len = float(params['marker_length_m'])

    if _NEW_API:
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        board = cv2.aruco.CharucoBoard(
            (sq_x, sq_y), sq_len, mk_len, dictionary
        )
        detector = cv2.aruco.ArucoDetector(dictionary)
    else:
        dictionary = cv2.aruco.Dictionary_get(dict_id)
        board = cv2.aruco.CharucoBoard_create(
            squaresX=sq_x, squaresY=sq_y,
            squareLength=sq_len, markerLength=mk_len,
            dictionary=dictionary,
        )
        detector = None  # use cv2.aruco.detectMarkers directly in 4.5.x

    return dictionary, board, detector


def _detect_charuco(
    gray: np.ndarray,
    dictionary,
    board,
    detector,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict:
    """
    Run ArUco + ChArUco detection on a grayscale image.

    Returns a dict with keys:
        valid       bool    True if pose was estimated
        n_corners   int     number of ChArUco corners detected
        rvec        ndarray (3,1) rotation vector (board-in-camera), or None
        tvec        ndarray (3,1) translation vector (board-in-camera), or None
        R_target2cam ndarray (3,3) rotation matrix (board-in-camera), or None
        t_target2cam ndarray (3,1) translation vector (board-in-camera), or None
        charuco_corners  ndarray detected corners for drawing, or None
        charuco_ids      ndarray detected corner IDs for drawing, or None
    """
    result = {
        'valid': False, 'n_corners': 0,
        'rvec': None, 'tvec': None,
        'R_target2cam': None, 't_target2cam': None,
        'charuco_corners': None, 'charuco_ids': None,
    }

    # --- Step 1: detect ArUco markers ---
    if _NEW_API and detector is not None:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)

    if ids is None or len(ids) == 0:
        return result

    # --- Step 2: refine to ChArUco sub-pixel corners ---
    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, board,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
    )

    if not retval or charuco_corners is None or len(charuco_corners) < MIN_CORNERS:
        result['n_corners'] = int(retval) if retval else 0
        return result

    result['n_corners'] = int(retval)
    result['charuco_corners'] = charuco_corners
    result['charuco_ids'] = charuco_ids

    # --- Step 3: estimate board pose in camera frame ---
    pose_ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board,
        camera_matrix, dist_coeffs,
        None, None,
    )

    if not pose_ok:
        return result

    result['valid']        = True
    result['rvec']         = rvec
    result['tvec']         = tvec
    result['R_target2cam'] = cv2.Rodrigues(rvec)[0]
    result['t_target2cam'] = tvec.reshape(3, 1)

    return result


def _draw_overlay(
    frame: np.ndarray,
    detection: dict,
    n_samples: int,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    status_msg: str = '',
) -> np.ndarray:
    """Draw detection result and status text onto a copy of frame."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Draw detected ChArUco corners
    if detection['charuco_corners'] is not None and detection['charuco_ids'] is not None:
        n = detection['n_corners']
        colour = (0, 200, 0) if n >= WARN_CORNERS else (0, 165, 255)
        cv2.aruco.drawDetectedCornersCharuco(
            vis, detection['charuco_corners'], detection['charuco_ids'], colour
        )

    # Draw board axes if pose is valid
    if detection['valid'] and detection['rvec'] is not None:
        cv2.drawFrameAxes(
            vis, camera_matrix, dist_coeffs,
            detection['rvec'], detection['tvec'], 0.05
        )

    # Status panel (semi-transparent dark bar at bottom)
    bar_h = 90
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, vis, 0.35, 0, vis)

    # Detection status line
    if detection['valid']:
        n = detection['n_corners']
        quality = 'GOOD' if n >= WARN_CORNERS else 'MARGINAL'
        col = (0, 220, 0) if n >= WARN_CORNERS else (0, 165, 255)
        det_text = f"Board DETECTED  {n} corners  [{quality}]"
    elif detection['n_corners'] > 0:
        det_text = f"Board PARTIAL  {detection['n_corners']} corners  (need >={MIN_CORNERS} for pose)"
        col = (0, 120, 255)
    else:
        det_text = "Board NOT DETECTED"
        col = (0, 0, 220)

    cv2.putText(vis, det_text, (15, h - bar_h + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)

    # Sample count line
    count_text = f"Samples: {n_samples}/20  |  SPACE=capture  D=discard  Q=save+quit"
    cv2.putText(vis, count_text, (15, h - bar_h + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 1, cv2.LINE_AA)

    # Optional status message (brief feedback after capture/discard)
    if status_msg:
        cv2.putText(vis, status_msg, (15, h - bar_h + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 100), 1, cv2.LINE_AA)

    return vis


class CalibCollector(Node):
    def __init__(self, board_params: dict, dictionary, board, detector):
        super().__init__('calib_collector')

        self.board_params = board_params
        self.dictionary   = dictionary
        self.board        = board
        self.detector     = detector

        self.bridge       = CvBridge()
        self.tf_buffer    = Buffer(cache_time=rclpy.duration.Duration(seconds=10))
        self.tf_listener  = TransformListener(self.tf_buffer, self)

        # State
        self.latest_frame:      np.ndarray | None = None
        self.camera_matrix:     np.ndarray | None = None
        self.dist_coeffs:       np.ndarray | None = None
        self.tf_link_to_optical: dict | None = None   # stored once at startup
        self.samples:           list[dict]  = []
        self.status_msg:        str         = ''
        self.status_msg_time:   float       = 0.0

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, COLOR_TOPIC, self._image_callback, 2
        )
        self.info_sub = self.create_subscription(
            CameraInfo, CAMINFO_TOPIC, self._camera_info_callback, 1
        )

        self.get_logger().info('CalibCollector started.')

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def _image_callback(self, msg: Image) -> None:
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        if self.camera_matrix is not None:
            return  # already stored
        k = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        d = np.array(msg.d, dtype=np.float64)
        self.camera_matrix = k
        self.dist_coeffs   = d
        self.get_logger().info(
            f'Camera intrinsics received: fx={k[0,0]:.1f}  fy={k[1,1]:.1f}  '
            f'cx={k[0,2]:.1f}  cy={k[1,2]:.1f}'
        )

    # ------------------------------------------------------------------
    # Startup TF acquisition
    # ------------------------------------------------------------------

    def try_acquire_link_to_optical_tf(self) -> bool:
        """
        Look up the static TF d435i_link -> d435i_color_optical_frame.
        Returns True on success. This is called once from the main loop.
        """
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                CAMERA_LINK_FRAME, CAMERA_OPTICAL_FRAME,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(nanoseconds=100_000_000),
            )
        except TransformException:
            return False

        t = tf_stamped.transform.translation
        q = tf_stamped.transform.rotation
        self.tf_link_to_optical = {
            'translation': {'x': t.x, 'y': t.y, 'z': t.z},
            'rotation_xyzw': {'x': q.x, 'y': q.y, 'z': q.z, 'w': q.w},
        }
        self.get_logger().info(
            f'TF {CAMERA_LINK_FRAME} -> {CAMERA_OPTICAL_FRAME} acquired: '
            f't=({t.x:.4f}, {t.y:.4f}, {t.z:.4f})'
        )
        return True

    # ------------------------------------------------------------------
    # Sample collection
    # ------------------------------------------------------------------

    def capture_sample(self, detection: dict) -> bool:
        """
        Record one (R_gripper2base, t_gripper2base, R_target2cam, t_target2cam) pair.
        Returns True on success.
        """
        if not detection['valid']:
            self._set_status('NOT CAPTURED — board not detected')
            return False

        # Robot EE pose from TF
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                ROBOT_BASE_FRAME, ROBOT_EE_FRAME,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=TF_TIMEOUT_SEC),
            )
        except TransformException as e:
            self.get_logger().warn(f'TF {ROBOT_BASE_FRAME}->{ROBOT_EE_FRAME} failed: {e}')
            self._set_status('NOT CAPTURED — robot TF unavailable')
            return False

        t = tf_stamped.transform.translation
        q = tf_stamped.transform.rotation
        R_g2b = _quaternion_to_matrix(q.x, q.y, q.z, q.w)
        t_g2b = np.array([t.x, t.y, t.z])

        sample = {
            'id':             len(self.samples),
            'n_charuco_corners': detection['n_corners'],
            'R_gripper2base': R_g2b.tolist(),
            't_gripper2base': t_g2b.tolist(),       # flat [x, y, z] in metres
            'R_target2cam':   detection['R_target2cam'].tolist(),
            't_target2cam':   detection['t_target2cam'].flatten().tolist(),
        }
        self.samples.append(sample)

        msg = (f'CAPTURED #{len(self.samples)}  '
               f'({detection["n_corners"]} corners)  '
               f'EE: ({t.x:.3f}, {t.y:.3f}, {t.z:.3f}) m')
        self._set_status(msg)
        print(f'  {msg}')
        return True

    def discard_last(self) -> None:
        if not self.samples:
            self._set_status('Nothing to discard')
            print('  Nothing to discard.')
            return
        removed = self.samples.pop()
        self._set_status(f'Discarded sample #{removed["id"]}')
        print(f'  Discarded sample #{removed["id"]}.')

    def _set_status(self, msg: str) -> None:
        self.status_msg      = msg
        self.status_msg_time = time.monotonic()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_samples(self) -> None:
        if not self.samples:
            print('No samples collected — nothing saved.')
            return

        if self.camera_matrix is None or self.dist_coeffs is None:
            print('No camera info received — cannot save.')
            return

        data = {
            'board_params':        self.board_params,
            'camera_info': {
                'frame_id':    CAMERA_OPTICAL_FRAME,
                'K':           self.camera_matrix.flatten().tolist(),
                'D':           self.dist_coeffs.tolist(),
            },
            'tf_d435i_link_to_optical': self.tf_link_to_optical,
            'samples':             self.samples,
        }
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(data, f, indent=2)

        print()
        print(f'Saved {len(self.samples)} samples to:  {OUTPUT_FILE}')
        print()
        if len(self.samples) < 8:
            print('WARNING: Fewer than 8 samples. Calibration may be unreliable.')
            print('         Collect at least 15-20 samples for a robust result.')
        elif len(self.samples) < 15:
            print('NOTE: Consider collecting more samples (15-20 recommended).')
        else:
            print(f'Sample count is good ({len(self.samples)} samples).')


def main() -> None:
    # --- Pre-flight: check cv2.aruco ---
    if not hasattr(cv2, 'aruco'):
        print('ERROR: cv2.aruco not available.')
        print('Run: pip3 install --break-system-packages opencv-contrib-python==4.5.5.64')
        sys.exit(1)

    # --- Load board params ---
    if not BOARD_PARAMS_FILE.exists():
        print(f'ERROR: board_params.yaml not found at {BOARD_PARAMS_FILE}')
        print('Run 01_generate_board.py first.')
        sys.exit(1)

    with open(BOARD_PARAMS_FILE) as f:
        board_params = yaml.safe_load(f)

    dictionary, board, detector = _make_aruco_objects(board_params)

    # --- Start ROS2 ---
    rclpy.init()
    node = CalibCollector(board_params, dictionary, board, detector)

    print()
    print('=== ChArUco Hand-Eye Calibration — Sample Collection ===')
    print(f'Waiting up to {STARTUP_TIMEOUT_SEC:.0f}s for camera and TF...')

    # --- Startup: wait for camera_info and static TF ---
    deadline = time.monotonic() + STARTUP_TIMEOUT_SEC
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.camera_matrix is not None and node.tf_link_to_optical is None:
            node.try_acquire_link_to_optical_tf()
        if node.camera_matrix is not None and node.tf_link_to_optical is not None:
            break

    if node.camera_matrix is None:
        print(f'ERROR: No camera info on {CAMINFO_TOPIC} after {STARTUP_TIMEOUT_SEC:.0f}s.')
        print('Is the RealSense driver running?')
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    if node.tf_link_to_optical is None:
        print(f'ERROR: TF {CAMERA_LINK_FRAME} -> {CAMERA_OPTICAL_FRAME} not available.')
        print('Is the RealSense driver running?')
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    print()
    print('Ready.')
    print('  1. On the teach pendant, enable FREEDRIVE.')
    print('  2. Move the robot so the ChArUco board is visible in the window below.')
    print('  3. Disable FREEDRIVE (robot locks in position).')
    print('  4. Press SPACE to capture when the board is detected (green corners).')
    print('  5. Move to a new pose and repeat (aim for 20 diverse samples).')
    print()
    print('  Controls:  SPACE = capture   D = discard last   Q = save & quit')
    print()

    status_display_sec = 2.0
    cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Calibration', 1280, 720)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            frame = node.latest_frame
            cam_matrix = node.camera_matrix
            dist_coeffs = node.dist_coeffs

            if frame is None or cam_matrix is None or dist_coeffs is None:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, 'Waiting for camera...', (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
                cv2.imshow('Calibration', placeholder)
                key = cv2.waitKey(50) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detection = _detect_charuco(
                gray, dictionary, board, detector, cam_matrix, dist_coeffs
            )

            # Clear status message after display period
            age = time.monotonic() - node.status_msg_time
            status = node.status_msg if age < status_display_sec else ''

            vis = _draw_overlay(frame, detection, len(node.samples), cam_matrix, dist_coeffs, status)
            cv2.imshow('Calibration', vis)

            key = cv2.waitKey(30) & 0xFF

            if key == ord(' '):
                node.capture_sample(detection)

            elif key in (ord('d'), ord('D')):
                node.discard_last()

            elif key in (ord('q'), ord('Q'), 27):
                break

    except KeyboardInterrupt:
        pass

    finally:
        cv2.destroyAllWindows()
        node.save_samples()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
