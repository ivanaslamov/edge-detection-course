#define BOOST_TEST_MODULE EdgeDetectionTest
#include <boost/test/unit_test.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utils.h"


BOOST_AUTO_TEST_CASE(FlipTest)
{
	cv::Mat src = cv::Mat::ones(100, 100, CV_8UC3);
	cv::Mat result = cv_flip(src);
	cv::Scalar src_sum = cv::sum(src);
	cv::Scalar result_sum = cv::sum(result);
	BOOST_CHECK(src_sum == result_sum);
}


BOOST_AUTO_TEST_CASE(FlipTestAll)
{
	cv::Mat src = cv::Mat::zeros(3, 1, CV_8UC3);
	src.data[0] = 1;
	src.data[1] = 1;
	src.data[2] = 1;
	src.data[3] = 2;
	src.data[4] = 2;
	src.data[5] = 2;
	src.data[6] = 3;
	src.data[7] = 3;
	src.data[8] = 3;
	cv::Mat dst = cv_flip(src);
    BOOST_CHECK(dst.data[0] == 3);
    BOOST_CHECK(dst.data[1] == 3);
    BOOST_CHECK(dst.data[2] == 3);
    BOOST_CHECK(dst.data[3] == 2);
    BOOST_CHECK(dst.data[4] == 2);
    BOOST_CHECK(dst.data[5] == 2);
    BOOST_CHECK(dst.data[6] == 1);
    BOOST_CHECK(dst.data[7] == 1);
    BOOST_CHECK(dst.data[8] == 1);
}
