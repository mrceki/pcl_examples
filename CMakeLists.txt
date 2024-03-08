cmake_minimum_required(VERSION 3.0.2)
project(pcl_interface)

add_compile_options(-std=c++17)

find_package(yaml-cpp REQUIRED)

find_package(catkin REQUIRED COMPONENTS
	roscpp
	pcl_ros
	pcl_conversions
	sensor_msgs
)

catkin_package(
	INCLUDE_DIRS include
	LIBRARIES pcl_interface
	CATKIN_DEPENDS roscpp pcl_ros pcl_conversions 
)

include_directories(
	include
	${catkin_INCLUDE_DIRS}
	${YAML_CPP_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
	src/pcl_interface.cpp
)

# target_link_libraries(${PROJECT_NAME} ${YAML_CPP_LIBRARIES})

add_executable(detection src/main.cpp)
# add_dependencies(detection ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(detection
	${PROJECT_NAME}
	${catkin_LIBRARIES}
	${YAML_CPP_LIBRARIES}
)

install(TARGETS detection
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
	DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
	FILES_MATCHING PATTERN "*.h"
)