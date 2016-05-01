#include <stdio.h>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Mat cv_edges(cv::Mat src);

cv::Mat cv_sobel(cv::Mat op, cv::Mat src);
