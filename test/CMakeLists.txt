find_package( Boost COMPONENTS unit_test_framework REQUIRED )

find_package( OpenCV REQUIRED )

add_definitions (-DBOOST_TEST_DYN_LINK)
include_directories(../src/include ${Boost_INCLUDE_DIRS})

add_executable( test_detection test.cpp )
target_link_libraries( test_detection utils ${OpenCV_LIBS} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} )
