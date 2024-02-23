cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(pcl_examples)

find_package(PCL 1.14 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(euclidian src/euclidian_extraction.cpp)
target_link_libraries(euclidian ${PCL_LIBRARIES})

add_executable(color_based src/color_based.cpp)
target_link_libraries(color_based ${PCL_LIBRARIES})

add_executable(plane src/plane.cpp)
target_link_libraries(plane ${PCL_LIBRARIES})

add_executable(3d_recognition src/3d_recognition.cpp)
target_link_libraries(3d_recognition ${PCL_LIBRARIES})

add_executable(region_growing src/region_growing.cpp)
target_link_libraries(region_growing ${PCL_LIBRARIES})

add_executable(moment_of_inertia src/moment_of_inertia.cpp)
target_link_libraries(moment_of_inertia ${PCL_LIBRARIES})