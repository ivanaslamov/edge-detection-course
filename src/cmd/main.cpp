#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "edges.h"

using namespace cv;
using namespace std;


int main( int argc, char** argv )
{
    Mat img = imread(argv[1]);

    Mat img_gray;
    cvtColor(img, img_gray, CV_BGR2GRAY);
    Mat temperature;
    img_gray.convertTo(temperature, CV_32FC1);
    temperature = temperature / 255;

    Mat x_op = Mat(3,3,CV_32FC1);
    x_op.at<float>(0,0) = -1;
    x_op.at<float>(0,1) = 0;
    x_op.at<float>(0,2) = 1;
    
    x_op.at<float>(1,0) = -2;
    x_op.at<float>(1,1) = 0;
    x_op.at<float>(1,2) = 2;

    x_op.at<float>(2,0) = -1;
    x_op.at<float>(2,1) = 0;
    x_op.at<float>(2,2) = 1;

    Mat x_sobel_results = cv_sobel(x_op, temperature);

    Mat y_op = Mat(3,3,CV_32FC1);
    y_op.at<float>(0,0) = -1;
    y_op.at<float>(0,1) = -2;
    y_op.at<float>(0,2) = -1;
    
    y_op.at<float>(1,0) = 0;
    y_op.at<float>(1,1) = 0;
    y_op.at<float>(1,2) = 0;

    y_op.at<float>(2,0) = 1;
    y_op.at<float>(2,1) = 2;
    y_op.at<float>(2,2) = 1;

    Mat y_sobel_results = cv_sobel(y_op, temperature);

    cv::pow(x_sobel_results, 2, x_sobel_results);
    cv::pow(y_sobel_results, 2, y_sobel_results);

    Mat constraint;
    cv::sqrt(x_sobel_results + y_sobel_results, constraint);    
    
    threshold(constraint, constraint, 1, 1, cv::THRESH_TRUNC);

    cv::Mat edges;
    for(int i=0; i<450; i++)
    {
        temperature = cv_anisotropic_blurring(constraint, temperature);

        Mat x_sobel = cv_sobel(x_op, temperature);
        Mat y_sobel = cv_sobel(y_op, temperature);
        cv::pow(x_sobel, 2, x_sobel);
        cv::pow(y_sobel, 2, y_sobel);
        cv::sqrt(x_sobel + y_sobel, edges);    

        imshow("preview", temperature);
        waitKey(10);
    }

    imwrite("/home/ivan/Desktop/edges.png", edges * 255);
    imwrite("/home/ivan/Desktop/blur.png", temperature * 255);

    return 0;
}
