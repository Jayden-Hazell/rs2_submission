#!/usr/bin/env python3
# 41069 Robotics Studio 2
# Authors: Jayden Hazell (24953364), Lachlan Partridge (24964557), Tomas Klimes (24773804)
# Generates the ChArUco calibration board PNG used for hand-eye calibration.
"""
01_generate_board.py

Generates a ChArUco calibration board PNG and companion params YAML.
Run this script once, then print the PNG at the stated physical size.

Board specification:
  Columns x Rows : 5 x 7 squares
  Square length  : 40 mm
  Marker length  : 30 mm
  Dictionary     : DICT_4X4_100
  Physical size  : 200 mm wide x 280 mm tall  (fits A4 portrait)

Usage:
    cd /home/jayden/ros2_ws/src/rs2
    python3 scripts/calibration/01_generate_board.py

Outputs (written next to this script):
    scripts/calibration/charuco_board.png   -- print this
    scripts/calibration/board_params.yaml   -- read by 02_collect_samples.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Board parameters — must match exactly between generation and detection.
# If you change any value here, delete charuco_board.png and re-run.
# ---------------------------------------------------------------------------

SQUARES_X     = 5       # number of squares along the horizontal axis
SQUARES_Y     = 7       # number of squares along the vertical axis
SQUARE_LENGTH = 0.040   # metres (40 mm)
MARKER_LENGTH = 0.030   # metres (30 mm)
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_100

# ---------------------------------------------------------------------------
# Image generation parameters
# Physical board: 200 mm wide x 280 mm tall
# At 300 DPI: 1 mm = 11.811 px → 40 mm square = ~473 px
# We generate slightly larger to include a white margin on all sides.
# ---------------------------------------------------------------------------
PIXELS_PER_SQUARE = 473
MARGIN_PX         = 120  # ~10 mm margin at 300 DPI

IMAGE_W = SQUARES_X * PIXELS_PER_SQUARE + 2 * MARGIN_PX  # ~2485 px
IMAGE_H = SQUARES_Y * PIXELS_PER_SQUARE + 2 * MARGIN_PX  # ~3431 px

# ---------------------------------------------------------------------------
# OpenCV version detection
# ArucoDetector was added in OpenCV 4.7+; its absence means we use the
# older (but still valid on Ubuntu 22.04 / OpenCV 4.5.x) API.
# ---------------------------------------------------------------------------
_NEW_API = hasattr(cv2.aruco, 'ArucoDetector')


def _check_aruco() -> None:
    if not hasattr(cv2, 'aruco'):
        print("ERROR: cv2.aruco is not available.")
        print("Run:  pip3 install --break-system-packages opencv-contrib-python==4.5.5.64")
        sys.exit(1)


def make_dictionary():
    if _NEW_API:
        return cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    return cv2.aruco.Dictionary_get(ARUCO_DICT_ID)


def make_board(dictionary):
    if _NEW_API:
        return cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary
        )
    return cv2.aruco.CharucoBoard_create(
        squaresX=SQUARES_X,
        squaresY=SQUARES_Y,
        squareLength=SQUARE_LENGTH,
        markerLength=MARKER_LENGTH,
        dictionary=dictionary,
    )


def generate_image(board) -> np.ndarray:
    if _NEW_API:
        return board.generateImage(
            outSize=(IMAGE_W, IMAGE_H), marginSize=MARGIN_PX, borderBits=1
        )
    return board.draw(
        outSize=(IMAGE_W, IMAGE_H), marginSize=MARGIN_PX, borderBits=1
    )


def save_params(out_dir: Path) -> None:
    params = {
        'squares_x':       SQUARES_X,
        'squares_y':       SQUARES_Y,
        'square_length_m': SQUARE_LENGTH,
        'marker_length_m': MARKER_LENGTH,
        'aruco_dict_id':   int(ARUCO_DICT_ID),
    }
    path = out_dir / 'board_params.yaml'
    with open(path, 'w') as f:
        yaml.safe_dump(params, f, default_flow_style=False)
    print(f"Board params written: {path}")


def main() -> None:
    _check_aruco()

    out_dir = Path(__file__).parent
    png_path = out_dir / 'charuco_board.png'
    yaml_path = out_dir / 'board_params.yaml'

    print(f"OpenCV {cv2.__version__}  (API: {'new 4.7+' if _NEW_API else 'legacy 4.5.x'})")

    dictionary = make_dictionary()
    board      = make_board(dictionary)
    image      = generate_image(board)

    cv2.imwrite(str(png_path), image)
    save_params(out_dir)

    phys_w_mm = SQUARES_X * SQUARE_LENGTH * 1000
    phys_h_mm = SQUARES_Y * SQUARE_LENGTH * 1000

    print()
    print("=" * 60)
    print("  BOARD GENERATED SUCCESSFULLY")
    print("=" * 60)
    print(f"  PNG file    : {png_path}")
    print(f"  Params file : {yaml_path}")
    print()
    print("  PRINTING INSTRUCTIONS:")
    print(f"  Physical size: {phys_w_mm:.0f} mm wide x {phys_h_mm:.0f} mm tall")
    print("  Paper size   : A4 portrait  (210 x 297 mm)")
    print("  Scale        : 100%  -- use 'Actual Size', NOT 'Fit to Page'")
    print("  Orientation  : Portrait")
    print()
    print("  After printing, verify with a ruler:")
    print(f"    Each square must measure exactly {SQUARE_LENGTH*1000:.0f} mm")
    print("    Total board: "
          f"{phys_w_mm:.0f} mm wide x {phys_h_mm:.0f} mm tall")
    print()
    print("  Mount the board FLAT and RIGID on a clipboard or 3 mm foamboard.")
    print("  A warped board will degrade calibration accuracy.")
    print("=" * 60)


if __name__ == '__main__':
    main()
