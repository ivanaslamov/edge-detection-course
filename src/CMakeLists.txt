find_package( OpenCV REQUIRED )
 
include_directories( include )

add_library( detect SHARED cpp/detect.cpp )

add_executable( main cmd/main.cpp )

target_link_libraries( main detect ${OpenCV_LIBS} )
