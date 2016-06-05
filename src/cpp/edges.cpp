#include "edges.h"

using namespace std;

/**
    Computes horizontal edge magnitude for each pixel

    @param colored image
    @return grayscale image with detected edges
*/
cv::Mat cv_edges(cv::Mat src)
{
	cvtColor(src, src, CV_BGR2GRAY);

	cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);

    for(int y=0; y<src.rows; y++)
    {
    	for(int x=0; x<src.cols-1; x++)
	    {
	    	unsigned char current = src.data[y*src.step + x];
	    	unsigned char next = src.data[y*src.step + x + 1];
	    	
	    	dst.data[y*src.step + x] = abs(current-next);
	    }
    }

    return dst;
}

/**
    Computes horizontal edge magnitude for each pixel

    @param op - 3x3 sobel operator, colored image
    @return grayscale image with detected edges
*/
cv::Mat cv_sobel(cv::Mat op, cv::Mat src)
{
	cvtColor(src, src, CV_BGR2GRAY);

	cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);

    for(int y=0; y<src.rows-op.rows+1; y++)
    {
    	for(int x=0; x<src.cols-op.cols+1; x++)
	    {
	    	double sum = 0;

		    for(int i=0; i<op.rows; i++)
		    {
		    	for(int j=0; j<op.cols; j++)
			    {
			    	sum += ((signed char) op.data[i*op.step + j]) * src.data[(y + i)*src.step + (x + j)];
			    }
			}

	    	dst.data[y*src.step + x] = (unsigned char) std::min(abs(sum), 255.0);
	    }
    }

    // shift image to center
    double offset_x = op.cols / 2.0 - 0.5;
    double offset_y = op.rows / 2.0 - 0.5;

	cv::Mat trans = cv::Mat::zeros(2, 3, CV_64FC1);
	trans.at<double>(0, 0) = 1;
	trans.at<double>(0, 2) = offset_x;
	trans.at<double>(1, 1) = 1;
	trans.at<double>(1, 2) = offset_y;

	warpAffine(dst, dst, trans, dst.size());

    return dst;
}

/**
    Computes gaussian second derivative

    @param op - 3x3 sobel operator, colored image
    @return grayscale image with detected edges
*/
cv::Mat cv_gaussian(int n)
{
	double s = 0.3 * ( n / 2 - 1 ) + 0.8;

	cv::Mat kernel = cv::Mat::zeros(n, n, CV_64FC1);

	for(int x=0; x<n; x++)
	{
		for(int y=0; y<n; y++)
		{
            double sq_dist = (double) (x-n/2)*(x-n/2) + (y-n/2)*(y-n/2);

            kernel.at<double>(x, y) = exp( - sq_dist / 2 / s / s );
		}
	}

	return kernel;
}

/**
    Computes gaussian second derivative

    @param op - 3x3 sobel operator, colored image
    @return grayscale image with detected edges
*/
cv::Mat cv_gaussian_second_derivative(int n)
{
    double s = 0.3 * ( n / 2 - 1 ) + 0.8;

	cv::Mat kernel = cv::Mat::zeros(n, n, CV_64FC1);

    for(int x=0; x<n; x++)
    {
        for(int y=0; y<n; y++)
        {
            double sq_x = (double) (x-n/2)*(x-n/2);
            double sq_y = (double) (y-n/2)*(y-n/2);
            double sq_dist = sq_x + sq_y;

			kernel.at<double>(x, y) =   (sq_x - s*s) / s / s / s / s * exp( - sq_dist / 2 / s / s ) +
                                        (sq_y - s*s) / s / s / s / s * exp( - sq_dist / 2 / s / s );
		}
	}

	return kernel;
}

/**
    Computes gaussian second derivative

    @param op - 3x3 sobel operator, colored image
    @return grayscale image with detected edges
*/
cv::Mat cv_convolve(cv::Mat M, cv::Mat image)
{
    cv::Mat dst = cv::Mat::zeros(image.rows, image.cols, CV_64FC1);

	for(int x=0; x<image.cols-M.cols; x++) {
        for(int y=0; y<image.rows-M.rows; y++) {
            for (int m = 0; m < M.cols; m  ++) {
                for (int n = 0; n < M.rows; n++) {
                    dst.at<double>(y, x) += ((double) image.at<uchar>(y+n, x+m)) * M.at<double>(n, m);
                }
            }
        }
    }

    // shift image to center
    double offset_x = M.cols / 2.0 - 0.5;
    double offset_y = M.rows / 2.0 - 0.5;

    cv::Mat trans = cv::Mat::zeros(2, 3, CV_64FC1);
    trans.at<double>(0, 0) = 1;
    trans.at<double>(0, 2) = offset_x;
    trans.at<double>(1, 1) = 1;
    trans.at<double>(1, 2) = offset_y;

    warpAffine(dst, dst, trans, dst.size());

    dst = dst / cv::sum(M)[0];

    return dst;
}

cv::Mat cv_zero_crossings(cv::Mat image_f)
{
    double epsilon = -0.001;

    cv::Mat dst = cv::Mat::zeros(image_f.rows, image_f.cols, CV_8UC1);

    for(int x=1; x<image_f.cols-1; x++) {
        for(int y=1; y<image_f.rows-1; y++) {
            if (image_f.at<double>(y-1, x) * image_f.at<double>(y+1, x) < epsilon) {
                dst.at<uchar>(y, x) = 255;
            }

            if (image_f.at<double>(y, x-1) * image_f.at<double>(y, x+1) < epsilon) {
                dst.at<uchar>(y, x) = 255;
            }

            if (image_f.at<double>(y-1, x-1) * image_f.at<double>(y+1, x+1) < epsilon) {
                dst.at<uchar>(y, x) = 255;
            }

            if (image_f.at<double>(y+1, x-1) * image_f.at<double>(y-1, x+1) < epsilon) {
                dst.at<uchar>(y, x) = 255;
            }
        }
    }

    return dst;
}

cv::Mat cv_filter_edges(cv::Mat upper_scale_edges, cv::Mat lower_scale_edges)
{
    cv::Mat src = upper_scale_edges.clone();
    cv::Mat dst = lower_scale_edges.clone() / 2;

    cv::Mat intersection;
    cv::bitwise_and(upper_scale_edges, lower_scale_edges, intersection);

    for(int x=1; x< intersection.cols-1; x++) {
        for(int y = 1; y < intersection.rows - 1; y++) {
            if( intersection.at<uchar>(y, x) > 0)
            {
                floodFill(dst, cv::Point(x,y), cv::Scalar(255));
                floodFill(intersection, cv::Point(x,y), cv::Scalar(0));
            }
        }
    }

    for(int x=1; x< dst.cols-1; x++) {
        for(int y = 1; y < dst.rows - 1; y++) {
            if( dst.at<uchar>(y, x) < 255)
            {
                dst.at<uchar>(y, x) = 0;
            }
        }
    }

    return dst;
}
