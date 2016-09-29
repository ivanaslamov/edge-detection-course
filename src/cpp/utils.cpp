#include "utils.h"

using namespace std;


cv::Mat cv_flip(cv::Mat src)
{
	cv::Mat dst = cv::Mat(src.rows, src.cols, CV_8UC3);

	for(int y=0; y<dst.rows; y++) {
		for(int x=0; x<dst.cols; x++) {
			for(int c=0; c<dst.channels(); c++) {
				dst.data[(dst.rows - y - 1)*dst.step + x*dst.channels() + c] = src.data[y*dst.step + x*dst.channels() + c];
			}
		}
	}

	return dst;
}
