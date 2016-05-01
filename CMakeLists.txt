cmake_minimum_required( VERSION 2.8 )

project( line_detection )

find_package( OpenCV REQUIRED )

include_directories( include )

add_library( detect SHARED src/detect.cpp )
add_executable( main src/main.cpp )

target_link_libraries( main detect ${OpenCV_LIBS} )
