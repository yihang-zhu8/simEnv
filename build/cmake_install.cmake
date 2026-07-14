# Install script for directory: /home/zhuyihang/simEnv/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/zhuyihang/simEnv/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  
      if (NOT EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
        file(MAKE_DIRECTORY "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}")
      endif()
      if (NOT EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/.catkin")
        file(WRITE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/.catkin" "")
      endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/zhuyihang/simEnv/install/_setup_util.py")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/zhuyihang/simEnv/install" TYPE PROGRAM FILES "/home/zhuyihang/simEnv/build/catkin_generated/installspace/_setup_util.py")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/zhuyihang/simEnv/install/env.sh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/zhuyihang/simEnv/install" TYPE PROGRAM FILES "/home/zhuyihang/simEnv/build/catkin_generated/installspace/env.sh")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/zhuyihang/simEnv/install/setup.bash;/home/zhuyihang/simEnv/install/local_setup.bash")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/zhuyihang/simEnv/install" TYPE FILE FILES
    "/home/zhuyihang/simEnv/build/catkin_generated/installspace/setup.bash"
    "/home/zhuyihang/simEnv/build/catkin_generated/installspace/local_setup.bash"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/zhuyihang/simEnv/install/setup.sh;/home/zhuyihang/simEnv/install/local_setup.sh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/zhuyihang/simEnv/install" TYPE FILE FILES
    "/home/zhuyihang/simEnv/build/catkin_generated/installspace/setup.sh"
    "/home/zhuyihang/simEnv/build/catkin_generated/installspace/local_setup.sh"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/zhuyihang/simEnv/install/setup.zsh;/home/zhuyihang/simEnv/install/local_setup.zsh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/zhuyihang/simEnv/install" TYPE FILE FILES
    "/home/zhuyihang/simEnv/build/catkin_generated/installspace/setup.zsh"
    "/home/zhuyihang/simEnv/build/catkin_generated/installspace/local_setup.zsh"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/zhuyihang/simEnv/install/setup.fish;/home/zhuyihang/simEnv/install/local_setup.fish")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/zhuyihang/simEnv/install" TYPE FILE FILES
    "/home/zhuyihang/simEnv/build/catkin_generated/installspace/setup.fish"
    "/home/zhuyihang/simEnv/build/catkin_generated/installspace/local_setup.fish"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/zhuyihang/simEnv/install/.rosinstall")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/zhuyihang/simEnv/install" TYPE FILE FILES "/home/zhuyihang/simEnv/build/catkin_generated/installspace/.rosinstall")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/zhuyihang/simEnv/build/gtest/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/building_generator_core/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/building_generator_interfaces/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/uav_simulator/Utils/quadrotor_msgs/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/unitree_guide/unitree_ros_to_real/unitree_legged_sdk/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/unitree_guide/unitree_ros/robots/a1_description/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/uav_simulator/Utils/cmake_utils/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/uav_simulator/map_generator/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/uav_simulator/Utils/pose_utils/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/building_obstacles/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/simenv_cbf_train/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/simenv_train/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/building_generator_classic/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/Mid360_imu_sim/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/uav_simulator/Utils/odom_visualization/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/uav_simulator/local_sensing/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/uav_simulator/mockamap/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/uav_simulator/so3_control/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/uav_simulator/Utils/multi_map_server/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/uav_simulator/Utils/uav_utils/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/uav_simulator/so3_quadrotor_simulator/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/unitree_guide/unitree_ros_to_real/unitree_legged_msgs/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/unitree_guide/unitree_ros/unitree_legged_control/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/unitree_guide/unitree_ros_to_real/unitree_legged_real/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/unitree_guide/unitree_guide/unitree_actuator_sdk/unitree_motor_ctrl/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/uav_simulator/Utils/rviz_plugins/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/unitree_guide/unitree_ros/unitree_controller/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/unitree_guide/unitree_ros/unitree_gazebo/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/unitree_guide/unitree_guide/unitree_guide/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/unitree_guide/unitree_guide/unitree_move_base/cmake_install.cmake")
  include("/home/zhuyihang/simEnv/build/uav_simulator/Utils/waypoint_generator/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/zhuyihang/simEnv/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
