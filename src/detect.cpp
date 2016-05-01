#include "detect.h"

using namespace std;

/**
    Computes horizontal edge magnitude for each pixel

    @param colored image
    @return grayscale image with detected edges
*/
cv::Mat detect(cv::Mat src)
{
	cvtColor(src, src, CV_BGR2GRAY);

	cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);

    for(int y=0; y<src.rows-1; y++)
    {
    	for(int x=0; x<src.cols; x++)
	    {
	    	unsigned char current = src.data[y*src.step + x];
	    	unsigned char next = src.data[(y+1)*src.step + x];
	    	
	    	dst.data[y*src.step + x] = abs(current-next);
	    }
    }

    return dst;
}
