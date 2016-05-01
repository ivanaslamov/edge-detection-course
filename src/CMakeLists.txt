find_package( OpenCV REQUIRED )
 
include_directories( include )

add_library( edges SHARED cpp/edges.cpp )

add_executable( main cmd/main.cpp )

target_link_libraries( main edges ${OpenCV_LIBS} )
