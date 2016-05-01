find_package( OpenCV REQUIRED )
 
include_directories( include )

add_library( utils SHARED cpp/utils.cpp )

add_executable( main cmd/main.cpp )

target_link_libraries( main utils ${OpenCV_LIBS} )
