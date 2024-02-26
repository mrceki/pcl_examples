cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(pcl_examples)

find_package(PCL 1.14 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

function(demo name)
	add_executable(${name} src/${name}.cpp)
	#add_dependencies(${name} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
	target_link_libraries(${name} ${PCL_LIBRARIES}) #${catkin_LIBRARIES}
	#target_include_directories(${name} SYSTEM PUBLIC ${catkin_INCLUDE_DIRS})
	# set_target_properties(${name} PROPERTIES OUTPUT_NAME ${name} PREFIX "")
	# install(TARGETS ${name}
		# ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
		# LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
		# RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
	# )
endfunction()

demo (euclidian_extraction)
demo (color_based)
demo (plane)
demo (3d_recognition)
demo (region_growing)
demo (moment_of_inertia)
demo (cylinder_segmentation)
demo (pallet)
