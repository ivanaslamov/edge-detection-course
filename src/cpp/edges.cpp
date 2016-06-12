#include "edges.h"

using namespace std;
using namespace cv;

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
	cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, CV_32FC1);

    for(int y=0; y<src.rows-op.rows+1; y++)
    {
    	for(int x=0; x<src.cols-op.cols+1; x++)
	    {
	    	float sum = 0;

		    for(int i=0; i<op.rows; i++)
		    {
		    	for(int j=0; j<op.cols; j++)
			    {
			    	sum += op.at<float>(i, j) * src.at<float>(y + i,x + j);
			    }
			}

	    	dst.at<float>(y, x) = std::min(abs(sum), 255.0f);
	    }
    }

    // shift image to center
    float offset_x = op.cols / 2.0 - 0.5;
    float offset_y = op.rows / 2.0 - 0.5;

	cv::Mat trans = cv::Mat::zeros(2, 3, CV_32FC1);
	trans.at<float>(0, 0) = 1;
	trans.at<float>(0, 2) = offset_x;
	trans.at<float>(1, 1) = 1;
	trans.at<float>(1, 2) = offset_y;

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
    float s = 0.3 * ( n / 2 - 1 ) + 0.8;

	cv::Mat kernel = cv::Mat::zeros(n, n, CV_32FC1);

    for(int x=0; x<n; x++)
    {
        for(int y=0; y<n; y++)
        {
            float sq_x = (float) (x-n/2)*(x-n/2);
            float sq_y = (float) (y-n/2)*(y-n/2);
            float sq_dist = sq_x + sq_y;

			kernel.at<float>(x, y) =   (sq_x - s*s) / s / s / s / s * exp( - sq_dist / 2 / s / s ) +
                                        (sq_y - s*s) / s / s / s / s * exp( - sq_dist / 2 / s / s );
		}
	}

	return kernel;
}

/**

*/
cv::Mat cv_convolve(cv::Mat M, cv::Mat image)
{
    cv::Mat dst = cv::Mat::zeros(image.rows, image.cols, CV_32FC1);

	for(int x=0; x<image.cols-M.cols; x++) {
        for(int y=0; y<image.rows-M.rows; y++) {
            for (int m = 0; m < M.cols; m  ++) {
                for (int n = 0; n < M.rows; n++) {
                    dst.at<float>(y, x) += ((float) image.at<float>(y+n, x+m)) * M.at<float>(n, m);
                }
            }
        }
    }

    // shift image to center
    float offset_x = M.cols / 2.0 - 0.5;
    float offset_y = M.rows / 2.0 - 0.5;

    cv::Mat trans = cv::Mat::zeros(2, 3, CV_32FC1);
    trans.at<float>(0, 0) = 1;
    trans.at<float>(0, 2) = offset_x;
    trans.at<float>(1, 1) = 1;
    trans.at<float>(1, 2) = offset_y;

    warpAffine(dst, dst, trans, dst.size());

    dst = dst / cv::sum(M)[0];

    return dst;
}

cv::Mat cv_zero_crossings(cv::Mat image_f)
{
    float epsilon = -0.001;

    cv::Mat dst = cv::Mat::zeros(image_f.rows, image_f.cols, CV_32FC1);

    for(int x=1; x<image_f.cols-1; x++) {
        for(int y=1; y<image_f.rows-1; y++) {
            if (image_f.at<float>(y-1, x) * image_f.at<float>(y+1, x) < epsilon) {
                dst.at<float>(y, x) = 255;
            }

            if (image_f.at<float>(y, x-1) * image_f.at<float>(y, x+1) < epsilon) {
                dst.at<float>(y, x) = 255;
            }

            if (image_f.at<float>(y-1, x-1) * image_f.at<float>(y+1, x+1) < epsilon) {
                dst.at<float>(y, x) = 255;
            }

            if (image_f.at<float>(y+1, x-1) * image_f.at<float>(y-1, x+1) < epsilon) {
                dst.at<float>(y, x) = 255;
            }
        }
    }

    return dst;
}


/* compute anisotropic blurring */
cv::Mat cv_anisotropic_blurring(Mat constraint, Mat temperature)
{
    float k = 100;

    if(temperature.rows != constraint.rows || temperature.cols != constraint.cols)
    {
        exit(1);
    }

    cv::Mat output = temperature.clone();

    for(int x=0; x<temperature.cols; x++) {
        for (int y = 0; y < temperature.rows; y++) {
            // south
            if (x+1 < temperature.cols) {
                float c = (constraint.at<float>(y, x) + constraint.at<float>(y, x+1)) / 2;
//                float c = 1 / (1 + pow( ((constraint.at<float>(y, x) + constraint.at<float>(y, x+1)) / 2) / k, 2));

                float d_t = (temperature.at<float>(y, x+1)- temperature.at<float>(y, x)) / 2 / 4;
                output.at<float>(y, x) += d_t*(1-c);
            }

            // north
            if (x > 0) {
                float c = (constraint.at<float>(y, x) + constraint.at<float>(y, x-1)) / 2;
//                float c = 1 / (1 + pow( ((constraint.at<float>(y, x) + constraint.at<float>(y, x-1)) / 2) / k, 2));

                float d_t = (temperature.at<float>(y, x-1) - temperature.at<float>(y, x)) / 2 / 4;
                output.at<float>(y, x) += d_t*(1-c);
            }

            // east
            if (y+1 < temperature.rows) {
                float c = (constraint.at<float>(y, x) + constraint.at<float>(y+1, x)) / 2;
//                float c = 1 / (1 + pow( ((constraint.at<float>(y, x) + constraint.at<float>(y+1, x)) / 2) / k, 2));

                float d_t = (temperature.at<float>(y+1, x) - temperature.at<float>(y, x)) / 2 / 4;
                output.at<float>(y, x) += d_t*(1-c);
            }

            // west
            if (y > 0) {
                float c = (constraint.at<float>(y, x) + constraint.at<float>(y-1, x)) / 2;
//                float c = 1 / (1 + pow( ((constraint.at<float>(y, x) + constraint.at<float>(y-1, x)) / 2) / k, 2));

                float d_t = (temperature.at<float>(y-1, x) - temperature.at<float>(y, x)) / 2 / 4;
                output.at<float>(y, x) += d_t*(1-c);
            }
        }
    }
    return output;
}
