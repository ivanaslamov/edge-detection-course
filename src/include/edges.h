#include <stdio.h>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Mat cv_edges(cv::Mat src);

cv::Mat cv_sobel(cv::Mat op, cv::Mat src);

cv::Mat cv_gaussian(int n);

cv::Mat cv_gaussian_second_derivative(int n);

cv::Mat cv_convolve(cv::Mat M, cv::Mat N);

cv::Mat cv_zero_crossings(cv::Mat image_f);

cv::Mat cv_anisotropic_blurring(cv::Mat constraint, cv::Mat temperature);
