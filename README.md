# ARISS — Autonomous Robotic Inspection and Sampling System

> **UTS 41069 Robotics Studio 2 — Capstone Project**

ARISS is a table-mounted 3D scanning system that autonomously navigates a UR3e robotic arm along a pre-recorded waypoint path, captures aligned RGB-D images at each stop, and reconstructs the scanned object into a point cloud and surface mesh — all from a single operator GUI.

---

## Contents

1. [Project Overview](#1-project-overview)
2. [System Requirements](#2-system-requirements)
3. [Software Installation](#3-software-installation)
4. [Robot Connection Setup](#4-robot-connection-setup)
5. [Running ARISS](#5-running-ariss)
6. [Using the GUI](#6-using-the-gui)
7. [Offline Reconstruction](#7-offline-reconstruction)
8. [Safety](#8-safety)
9. [Known Limitations](#9-known-limitations)
10. [FAQ / Troubleshooting](#10-faq--troubleshooting)

---

## 1. Project Overview

ARISS consists of four cooperating ROS 2 nodes orchestrated by a central state machine:

| Node | Role |
|---|---|
| `main_control_node` | Top-level supervisor; routes commands and monitors node heartbeats |
| `movement_node` | Executes joint-trajectory waypoints on the UR3e |
| `scan_node` | Captures and saves aligned RGB-D frames from the RealSense D435i |
| `gui_node` | PySide6 operator interface with status, camera feeds, and 3D model viewer |

**Typical scan workflow:**

1. Launch ARISS and wait for all nodes to reach `IDLE`.
2. Press **Move to Home** in the GUI — the arm moves to the pre-scan position.
3. Press **Start** — the arm traverses all waypoints, pausing at each for an RGB-D capture.
4. When the scan completes the GUI status reads `AUTO: SCAN_COMPLETE`.
5. A background process automatically merges the captures into a point cloud and `.obj` mesh, stored under `reconstructed_scans/`.
6. Open the **Model** tab in the GUI to browse and download the finished mesh.

### Data Storage

| Directory | Contents |
|---|---|
| `raw_scans/<session>/` | Per-capture RGB, depth, camera info, and pose JSON files |
| `reconstructed_scans/<session>/` | `reconstruction.ply`, `<session>.obj`, and provenance JSON |
| `joints/autoScan.csv` | Joint-space waypoints for the automatic scan path |
| `joints/manualPosition.csv` | Named manual viewpoint positions |
| `config/workspace.yaml` | Workspace bounding box and drop-off zone coordinates |

---

## 2. System Requirements

### Hardware

| Component | Specification |
|---|---|
| Robot | Universal Robots UR3e (6-DOF collaborative arm) |
| Camera | Intel RealSense D435i (RGB-D, mounted on UR3e tool flange) |
| Computer | Ubuntu 22.04 LTS, x86-64, with a connected display |
| Network | Ethernet connection between the workstation and UR3e controller |

### Software

| Dependency | Notes |
|---|---|
| Ubuntu 22.04 LTS | Required OS |
| ROS 2 Humble Hawksbill | Core middleware |
| Python 3.10 | Ships with Ubuntu 22.04 |
| MoveIt 2 | `moveit` meta-package for ROS 2 Humble |
| UR Robot Driver | `ur_robot_driver` for ROS 2 Humble |
| RealSense ROS 2 Wrapper | `realsense2_camera` |
| pymoveit2 | External — must be cloned separately (see §3.5) |

---

## 3. Software Installation

### 3.1 ROS 2 Humble

Follow the official Debian package installation guide:
<https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html>

Install the full desktop variant:

```bash
sudo apt install ros-humble-desktop
```

Source ROS 2 in every new terminal, or add the line to `~/.bashrc`:

```bash
source /opt/ros/humble/setup.bash
```

### 3.2 UR Robot Driver

```bash
sudo apt install ros-humble-ur
```

### 3.3 Intel RealSense ROS 2 Wrapper

```bash
sudo apt install ros-humble-realsense2-camera
```

### 3.4 MoveIt 2

```bash
sudo apt install ros-humble-moveit
```

### 3.5 Workspace Setup

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Clone this repository
git clone <repo-url> rs2

# Clone pymoveit2 (required by movement_node)
git clone https://github.com/AndrejOrsula/pymoveit2.git
```

### 3.6 Python Dependencies

```bash
pip install -r ~/ros2_ws/src/rs2/requirements.txt
```

### 3.7 ROS 2 System Dependencies

```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

### 3.8 Build the Workspace

```bash
cd ~/ros2_ws
colcon build --symlink-install --merge-install
source install/setup.bash
```

Add to `~/.bashrc` to avoid re-sourcing every terminal:

```bash
source ~/ros2_ws/install/setup.bash
```

---

## 4. Robot Connection Setup

### 4.1 Network Configuration

Connect the UR3e controller box to the workstation with an Ethernet cable (direct connection or via a switch). Configure the workstation network interface in the same subnet as the robot.

Default addresses used by ARISS:

| Device | IP Address |
|---|---|
| UR3e controller | `192.168.0.194` |
| Workstation (reverse connection) | `192.168.0.100` |

Verify connectivity before launching:

```bash
ping 192.168.0.194
```

### 4.2 External Control URCap

ARISS requires the **External Control** URCap installed on the UR3e teach pendant.

1. On the teach pendant open **URCaps → +** and install `externalcontrol-x.x.x.urcap` from USB or the UR website.
2. Create or load a robot program containing a single **External Control** node.
3. Set the **Host IP** in the URCap settings to your workstation's IP (`192.168.0.100`).
4. Leave the **Custom Port** at `50002` (default).

Before starting ARISS, press **Play** on the teach pendant to start the External Control program and place the robot in **Remote Control** mode.

### 4.3 Hand-Eye Calibration

The camera-to-robot transform (`tool0` → `d435i_link`) is published as a static TF inside `master.launch.py`. The calibration values in the launch file were obtained using `easy_handeye2`.

If the camera is physically remounted or replaced, re-run the calibration scripts in the `calibration/` directory and update the `--qx`, `--qy`, `--qz`, `--qw`, `--x`, `--y`, `--z` arguments of the `handeye_tf` node in `master.launch.py`.

---

## 5. Running ARISS

### 5.1 Simulation Mode (no hardware required)

Simulation mode runs the full state machine and GUI without any physical hardware. The scan node skips camera and TF boot checks and simulates a 1-second capture delay. No real scan data is written.

```bash
ros2 launch rs2 master.launch.py use_fake_hardware:=true
```

### 5.2 Real Hardware Mode

Ensure the UR3e is powered on, the External Control program is playing on the teach pendant, and the RealSense D435i is connected via USB 3.

```bash
ros2 launch rs2 master.launch.py use_fake_hardware:=false
```

Override default IP addresses if your network is configured differently:

```bash
ros2 launch rs2 master.launch.py use_fake_hardware:=false \
  robot_ip:=192.168.0.194 \
  reverse_ip:=192.168.0.100
```

### 5.3 Launch Arguments Reference

| Argument | Default | Description |
|---|---|---|
| `use_fake_hardware` | `false` | `true` enables simulation without physical hardware |
| `robot_ip` | `192.168.0.194` | IP address of the UR3e controller |
| `reverse_ip` | `192.168.0.100` | Workstation IP for the UR driver reverse connection |
| `ur_type` | `ur3e` | UR robot model (do not change unless swapping robots) |
| `kinematics_params_file` | `config/my_robot_calibration.yaml` | Kinematic calibration file |

### 5.4 Node Startup Sequence

Nodes start in a staggered sequence to allow the lower-level stack to initialise first:

| Delay from launch | Component |
|---|---|
| +0 s | UR robot driver |
| +0 s | RealSense camera and hand-eye TF (real hardware only) |
| +4 s | MoveIt motion-planning stack |
| +2 s | `main_control_node` |
| +3 s | `gui_node` |
| +6 s | `movement_node` and `scan_node` |

Allow 15–20 seconds for all nodes to fully boot. The GUI status label will show `IDLE` when the system is ready to operate.

---

## 6. Using the GUI

The GUI opens automatically after launch. It is divided into a **left control panel** and a **right visualisation panel**.

### 6.1 Left Panel — Normal Mode

| Control | Function |
|---|---|
| **Status label** | Current system state (e.g., `IDLE`, `AUTO: MOVING_TO_WAYPOINT`) |
| **Start** | Begin the automatic scan sequence |
| **Pause** | Pause motion between waypoints; resume with Start |
| **Move to Home** | Move the arm to the pre-scan home position |
| **Emergency Stop** | Immediately halt all motion (latched — restart required) |
| **Object Size** | `large` / `small` — scan footprint selection (reserved for future use) |
| **Resolution** | `high` / `medium` / `low` — uses 100% / 75% / 50% of scan waypoints |
| **Speed** | `high` / `medium` / `low` — move time per waypoint: 1.5 s / 3.0 s / 6.0 s |
| **Coverage** | Average depth-pixel validity across all captures in the session (%) |
| **File Name** | Optional name for the scan output folder (ASCII, max 64 characters) |
| **Error Log** | Scrollable log of system warnings and errors received from all nodes |

### 6.2 Left Panel — Advanced Mode

Click the **Advanced Mode** tab to access manual robot positioning.

| Button | Description |
|---|---|
| **Top** | Move to the overhead viewpoint position |
| **Front** | Move to the front viewpoint position |
| **Back** | Move to the rear viewpoint position |
| **Left** | Move to the left-side viewpoint position |
| **Right** | Move to the right-side viewpoint position |
| **Emergency Stop** | Same as in Normal Mode |
| **Rescan Current Section** | Re-capture the current waypoint (reserved) |
| **Sample Object** | Trigger object sampling (reserved) |

Manual viewpoint moves are only accepted when the system state is `IDLE`, `PRE_SCAN_POSITION`, or `PAUSED`. Attempting a manual move in any other state will log a warning.

### 6.3 Right Panel — Visualisation Tabs

| Tab | Description |
|---|---|
| **Simulation** | Live digital twin of the UR3e rendered in OpenGL, updated from `/joint_states` |
| **Camera** | Three sub-tabs: **Camera Feed** (live RGB), **Depth Overlay** (RGB + JET colourmap blend at 45% opacity), **Point Cloud** (live 3D scatter from `/camera/d435i/depth/color/points`, downsampled to ≤ 8000 points) |
| **Model** | Select a completed scan from the dropdown, view its `.obj` mesh in the OpenGL viewer, and download it to `~/Downloads/` with the **Download** button |

### 6.4 Progress Bar

The progress bar at the top of the right panel shows the percentage of scan waypoints completed in the current run (published by `movement_node` on `/movement/progress`).

---

## 7. Offline Reconstruction

Reconstruction runs automatically in the background when a scan completes. To re-run reconstruction manually on a saved session:

```bash
cd ~/ros2_ws/src/rs2
python3 helpers/reconstruct.py \
  --session raw_scans/<session_name> \
  --no-visualise \
  --mesh
```

Output files written to `reconstructed_scans/<session_name>/`:

| File | Description |
|---|---|
| `reconstruction.ply` | Merged, filtered, and normal-estimated point cloud |
| `<session_name>.obj` | Surface mesh from Poisson reconstruction |
| `reconstruction.json` | Provenance record (parameters, capture count, timestamp) |

### 7.1 Common Options

| Flag | Default | Description |
|---|---|---|
| `--voxel-size` | `0.005` | Voxel downsampling size in metres; smaller = finer, slower |
| `--icp` | off | Refine each capture against the accumulated cloud via point-to-plane ICP |
| `--icp-threshold` | `0.01` | ICP max correspondence distance in metres |
| `--mesh` | off | Run `mesh_from_cloud.py` automatically after point cloud is saved |
| `--no-visualise` | off | Skip the Open3D interactive viewer |
| `--depth-trunc` | `2.0` | Discard depth values beyond this distance in metres |
| `--bbox-min X Y Z` | from `config/workspace.yaml` | Crop minimum corner in `base_link` frame |
| `--bbox-max X Y Z` | from `config/workspace.yaml` | Crop maximum corner in `base_link` frame |
| `--debug-bbox` | off | Preview full cloud + workspace bounding box wireframe without saving output |

### 7.2 Reconstruction Modes

| Mode | When to use |
|---|---|
| **Mode A** (default) | Uses `pose_composed` from each `capture.json`; requires no additional arguments |
| **Mode B** (`--recompose`) | Recomposes pose from the per-capture pose chain plus the session calibration; useful when the live TF snapshot is more accurate than the composed transform |
| **Legacy** (`--legacy-pose`) | For pre-`capture.json` sessions that have a `pose.yaml` per capture |

### 7.3 Tuning Reconstruction Quality

All filter constants (per-capture SOR, global ROR, DBSCAN cluster filtering, normal estimation parameters) are documented in `RECONSTRUCTION_TUNING_GUIDE.txt` and in the `TUNABLE CONSTANTS` block at the top of `helpers/reconstruct.py`.

---

## 8. Safety

### 8.1 Emergency Stop

The **Emergency Stop** button is present on both the Normal Mode and Advanced Mode panels. Pressing it:

1. Immediately publishes a stop command to all nodes.
2. The arm publishes a hold-position trajectory to freeze in place.
3. The emergency-stop state is **permanently latched** — the system cannot resume operation without a full restart of the launch file.

**Do not use Emergency Stop to end a normal scan.** Use **Pause** for routine interruptions. Reserve Emergency Stop for genuine safety events.

### 8.2 Heartbeat Watchdog

Each node (GUI, movement, scan) publishes an `std_msgs/Empty` heartbeat at 2 Hz. `main_control_node` triggers an emergency stop if any heartbeat is absent for more than **3 seconds**. This protects against process crashes, frozen nodes, and ROS communication failures.

### 8.3 Joint Limit Enforcement

`movement_node` validates every joint target against software limits (±2π per joint, i.e., full UR joint range) before publishing to the trajectory controller. Any waypoint outside these limits is rejected and the node transitions to `ERROR`. Hardware limits are also enforced by the UR3e controller and the UR driver.

### 8.4 Physical Workspace

- Keep all body parts, tools, and objects clear of the robot's reach envelope during automated or manual movement.
- Do not attempt to physically redirect the arm while it is in motion — press Emergency Stop or use the teach pendant's physical E-stop.
- Configure UR3e collaborative safety settings (force/torque thresholds, safety planes) on the teach pendant to match your site requirements before operating.

---

## 9. Known Limitations

- **Simulation captures no data.** In `use_fake_hardware:=true` mode the scan node simulates a 1-second delay and transitions to `FINISHED_SCANNING` without writing any files to disk.
- **Emergency stop cannot be reset without restarting.** There is no software-only recovery path once the emergency stop is latched.
- **No live collision avoidance during scans.** Scan waypoints are pre-recorded and replayed without real-time collision checking. The workspace must be clear before starting a scan.
- **Reconstruction quality is calibration-dependent.** Inaccurate hand-eye calibration causes point clouds from different viewpoints to misalign, producing blurry reconstructions.
- **Manual viewpoints are fixed.** The positions in `joints/manualPosition.csv` are pre-recorded and may not be safe for all table configurations or object placements.
- **Rescan Current Section and Sample Object are not implemented.** The GUI buttons exist but the back-end logic is not wired up.
- **GUI requires a display.** `gui_node.py` creates a Qt window and imports `python-xlib`; headless operation requires a virtual framebuffer (e.g., `Xvfb`).
- **Open3D visualiser may fail on Wayland.** Use `--no-visualise` with offline reconstruction on Wayland desktops; open `.ply` files in MeshLab or CloudCompare instead.

---

## 10. FAQ / Troubleshooting

**Q: The GUI shows `BOOTING` and never transitions to `IDLE`.**
Check that all three nodes (movement, scan, GUI) started successfully. `main_control_node` waits up to 5 seconds for each node's first heartbeat before declaring a boot failure. Run `ros2 node list` to confirm the nodes are active, and check the terminal for `Boot failed` or `heartbeat timeout` messages.

---

**Q: `movement_node` transitions to `ERROR` immediately on startup.**
The node could not find `joints/autoScan.csv`. Verify the file exists at `<package_root>/joints/autoScan.csv`. Check terminal output for the `Boot check failed` message and the list of searched paths.

---

**Q: The scan node prints `Boot failed: camera not publishing`.**
The RealSense D435i is not publishing on `/camera/d435i/color/image_raw`. Confirm the camera is connected via USB 3 and `realsense2_camera` started without errors:

```bash
ros2 topic hz /camera/d435i/color/image_raw
```

---

**Q: The scan node prints `Boot failed: TF ... not available`.**
The transform from `base_link` to `d435i_color_optical_frame` has not appeared yet. This usually means the hand-eye TF publisher failed (real-hardware-only condition), or the UR driver has not finished publishing robot state. Wait a few seconds and re-launch, or check terminal output for TF-related errors.

---

**Q: Unable to connect to the robot.**
Confirm the `robot_ip` launch argument matches the UR3e controller's configured IP. Ensure the teach pendant has the External Control program running and the robot is in **Remote Control** mode. Test with `ping 192.168.0.194`.

---

**Q: The reconstructed point cloud looks misaligned or blurry.**
The most common cause is hand-eye calibration error. Re-run the calibration workflow in `calibration/` and update `master.launch.py`. As a quick test, try `--icp` with offline reconstruction to apply ICP refinement between individual captures.

---

**Q: Open3D viewer crashes or fails to open after reconstruction.**
Pass `--no-visualise` to suppress the viewer. The output `.ply` file is always written first and can be opened in MeshLab, CloudCompare, or Blender.

---

**Q: The GUI's Model tab shows "Reconstructed scans folder not found".**
The `reconstructed_scans/` directory does not exist yet. Complete at least one scan cycle (which creates it automatically), or create it manually:

```bash
mkdir -p ~/ros2_ws/src/rs2/reconstructed_scans
```

---

**Q: RViz opens but shows no robot model or shows the wrong model.**
Confirm that the build completed without errors (`colcon build --symlink-install --merge-install`) and that the install overlay is sourced (`source ~/ros2_ws/install/setup.bash`). RViz loads the URDF from the `rs2` package; a stale build cache can cause it to display an outdated model.
