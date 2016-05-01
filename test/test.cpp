#define BOOST_TEST_MODULE EdgeDetectionTest
#include <boost/test/unit_test.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "edges.h"


BOOST_AUTO_TEST_CASE(EdgeTest)
{
	cv::Mat src = cv::Mat::zeros(100, 100, CV_8UC3);
	src.data[50*src.step + 50] = 255;
	src.data[51*src.step + 50] = 255;
	src.data[52*src.step + 50] = 255;

	cv::Mat result = cv_edges(src);
    BOOST_CHECK(cv::sum(result)[0] > 0);
}

BOOST_AUTO_TEST_CASE(SobelTest)
{
	cv::Mat src = cv::Mat::zeros(100, 100, CV_8UC3);
	
	src.data[50*src.step + 50] = 255;
	src.data[51*src.step + 50] = 255;
	src.data[52*src.step + 50] = 255;

    cv::Mat op = cv::Mat(3,3,CV_8UC1);
    op.data[0] = -1;
    op.data[1] = 0;
    op.data[2] = 1;
    
    op.data[3] = -2;
    op.data[4] = 0;
    op.data[5] = 2;

    op.data[6] = -1;
    op.data[7] = 0;
    op.data[8] = 1;

	cv::Mat result = cv_sobel(op, src);
	
	cv::Mat src_gray, sobel_result;
    cv::cvtColor(src, src_gray, CV_BGR2GRAY);
    cv::Sobel( src_gray, sobel_result, CV_32F, 1, 0, 3);
    convertScaleAbs( sobel_result, sobel_result );

    BOOST_CHECK_EQUAL(cv::sum(result)[0], cv::sum(sobel_result)[0]);
    BOOST_CHECK(cv::sum(sobel_result)[0] > 0);
}
