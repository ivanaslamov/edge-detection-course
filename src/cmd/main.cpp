#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "utils.h"

using namespace cv;
using namespace std;


int main( int argc, char** argv )
{
    Mat img = imread(argv[1]);
    Mat flipped = cv_flip(img);

    imshow("flipped", flipped);
    waitKey();
    
    return 0;
}
