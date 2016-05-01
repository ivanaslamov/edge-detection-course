cmake_minimum_required( VERSION 3.2.2 )

project( line_detection )

add_subdirectory (src) 
add_subdirectory (test) 

enable_testing ()
add_test (NAME EdgeDetectionTest COMMAND test_detection)
