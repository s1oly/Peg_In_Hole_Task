# Peg-In-Hole Task

Autonomous robotic peg-in-hole insertion using tactile feedback from the GelSight Mini sensor. The system detects contact misalignment in real time and applies corrective rotations to a UR5e robotic arm until the insertion completes.

---

## Overview

This project uses the GelSight Mini, a vision-based tactile sensor, to guide a UR5e robot arm through a peg-in-hole insertion task. When the peg contacts the edge of the hole, the sensor's embedded marker array deforms in a pattern that encodes the nature of the misalignment. Image processing on the GelSight Mini stream produces a vector field highlighting the distortion on the sensor surface.

Using **Helmholtz-Hodge Decomposition**, the vector field is split into three components:

- **Potential (curl-free) component** — captures divergence, indicating the magnitude of the normal contact force
- **Solenoidal (divergence-free) component** — captures rotation, indicating torsional misalignment
- **Harmonic component** — the residual field; its mean direction identifies the contact force vector

The contact force vector from the harmonic field is combined with the peg's lever arm vector to compute an axis of rotation. The robot rotates about this axis by an angle proportional to the combined error magnitude, steering the peg clear of the obstruction until insertion completes.

---

## Tech Stack

| Component | Technology |
|---|---|
| Robot middleware | ROS (Robot Operating System) — Melodic / Noetic |
| Robot interface | `easyUR` — UR5e arm control library |
| Tactile sensor | GelSight Mini (vision-based) |
| Marker tracking | Coherent Point Drift with LLE regularization (CPD-LLE) |
| Vector field decomposition | Helmholtz-Hodge Decomposition via 2D FFT (`scipy.fftpack`) |
| Image processing | OpenCV, NumPy |
| Visualization | Matplotlib |
| Language | Python 3 |

---

## Requirements

### Hardware

- UR5e robotic arm with gripper
- GelSight Mini tactile sensor mounted on the gripper

### Software

- ROS Melodic or Noetic
- Python 3.6+

**Python packages:**
```
numpy
opencv-python
scipy
matplotlib
Pillow
rospy
ros_numpy
cv_bridge
easyUR
```

---

## Repository Structure

```
Peg_In_Hole_Task/
├── CMakeLists.txt                      # ROS catkin build configuration
├── package.xml                         # ROS package manifest
├── README.md
└── src/
    ├── stream_gelsight.py              # Step 1: capture and publish sensor frames
    ├── realtime_marker_processing.py   # Step 2: process tactile data, publish corrections
    ├── robot_control.py                # Step 3: apply corrections to the UR5e arm
    ├── marker_processing_utils.py      # Shared utilities (CPD-LLE, HHD, error metrics)
    ├── contact_case_1.bag              # Recorded ROS bag — contact scenario 1
    └── contact_case_2.bag              # Recorded ROS bag — contact scenario 2
```

---

## Running

The three scripts must be launched in separate terminals **in the order below**. Start `roscore` first.

**Terminal 0 — ROS core:**
```bash
roscore
```

---

**Terminal 1 — Stream GelSight sensor:**
```bash
cd src
python3 stream_gelsight.py
```

Captures frames from the GelSight Mini webcam and publishes them as ROS `sensor_msgs/Image` messages on `/gelsight_0` at 25 Hz. On startup, it scans `/sys/class/video4linux` to automatically locate the correct camera device.

---

**Terminal 2 — Real-time marker processing:**
```bash
cd src
python3 realtime_marker_processing.py
```

Subscribes to `/gelsight_0` and processes each frame:
1. Segments the marker array from the sensor image using color thresholding
2. Tracks marker displacement frame-to-frame using CPD-LLE registration
3. Applies Helmholtz-Hodge Decomposition to the displacement vector field
4. Determines the contact force direction from the harmonic component
5. Computes a ZYX Euler rotation correction and publishes it on `/target_euler_angles`

Correction is only published when the potential or solenoidal error exceeds a threshold, indicating meaningful contact.

---

**Terminal 3 — Robot control:**
```bash
cd src
python3 robot_control.py
```

Subscribes to `/target_euler_angles` and commands the UR5e arm each control cycle:
1. Reads the current end-effector pose from the robot
2. Transforms from end-effector frame → peg frame → corrected peg frame → new end-effector frame
3. Sends the new target pose to the arm via `easyUR`

When no correction is active and the arm is still above the target insertion depth, the robot advances the peg downward by 5 mm per cycle.

---

## ROS Topics

| Topic | Message Type | Publisher | Description |
|---|---|---|---|
| `/gelsight_0` | `sensor_msgs/Image` | `stream_gelsight.py` | Raw tactile sensor frames at 25 Hz |
| `/target_euler_angles` | `std_msgs/Float32MultiArray` | `realtime_marker_processing.py` | ZYX Euler correction angles (degrees) |
