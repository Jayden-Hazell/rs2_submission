cmake_minimum_required(VERSION 3.5)
project(rs2)

find_package(ament_cmake REQUIRED)

# Add 'models' to this list
install(DIRECTORY 
  config 
  launch 
  rviz 
  srdf 
  urdf
  models
  joints
  raw_scans
  reconstructed_scans
  DESTINATION share/${PROJECT_NAME}
)
  
# 2. Install your Python node so it can be found by 'ros2 run'
install(PROGRAMS
  scripts/main_control_node.py
  scripts/movement_node.py
  scripts/gui_node.py
  scripts/scan_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()