#define BOOST_TEST_MODULE EdgeDetectionTest
#include <boost/test/unit_test.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utils.h"


BOOST_AUTO_TEST_CASE(FlipTest)
{
	cv::Mat src = cv::Mat::zeros(100, 100, CV_8UC1);
	cv::Mat result = cv_flip(src);
    BOOST_CHECK(cv::sum(src) == cv::sum(result));
}
