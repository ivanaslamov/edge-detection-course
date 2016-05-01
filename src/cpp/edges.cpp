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
	cv::Mat trans = (cv::Mat_<double>(2, 3) << 1, 0, offset_x, 0, 1, offset_y);
	
	warpAffine(dst, dst, trans, dst.size());

    return dst;
}
