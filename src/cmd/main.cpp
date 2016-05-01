#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "detect.h"

using namespace cv;
using namespace std;


int main( int argc, char** argv )
{
    Mat img = imread(argv[1]);

    Mat results = detect(img);
    
    imshow("lines", results);
    waitKey();

    return 0;
}
