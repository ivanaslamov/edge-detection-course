#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "edges.h"

using namespace cv;
using namespace std;


int main( int argc, char** argv )
{
    Mat img = imread(argv[1]);
    Mat edges_results = cv_edges(img);

    Mat op = Mat(3,3,CV_8SC1);
    op.data[0] = -1;
    op.data[1] = 0;
    op.data[2] = 1;
    
    op.data[3] = -2;
    op.data[4] = 0;
    op.data[5] = 2;

    op.data[6] = -1;
    op.data[7] = 0;
    op.data[8] = 1;

    Mat sobel_results = cv_sobel(op, img);
    
    //imshow("results", edges_results);
    //waitKey();

	//imshow("results", sobel_results);
    //waitKey();
    
    cv::Mat src_gray, grad_x;
    cv::cvtColor(img, src_gray, CV_BGR2GRAY);
    //cv::Sobel( src_gray, grad_x, CV_32F, 1, 0, 3);
    //convertScaleAbs( grad_x, grad_x );

    //imshow("results", grad_x);
    //waitKey();

    Mat edges = cv::Mat::ones(img.rows, img.cols, CV_8UC1);

    for(int i=29; i>=9; i=i-2)
    {
        cv::Mat kernel_s = cv_gaussian_second_derivative(i);
        cv::Mat dd_image = cv_convolve(kernel_s, src_gray);
        cv::Mat crossings = cv_zero_crossings(dd_image);

        edges = cv_filter_edges(edges, crossings);

        imshow("results", edges);
        waitKey(300);
    }

    cv::imwrite("output.png", edges);

    return 0;
}
