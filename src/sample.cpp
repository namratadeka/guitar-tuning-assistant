#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <Math.h>

#define PI 3.14159265

using namespace std;
using namespace cv;

Mat doErosion(Mat src1, int x);
Mat doDilation(Mat src1, int x);
int countBoxes(Mat src1, int x);
Mat convolve(Mat src, double avgFilter[3][3]);

Mat src1;

Mat img3;
char window_name1[] = "Unprocessed Image";
char window_name2[] = "Processed Image";

int main( int argc, char** argv )
{
    cv::VideoCapture cap("inp3.mp4");
    if(!cap.isOpened()) {
        std::cout << "Unable to open the camera\n";
        std::exit(-1);
    }
    
    // Get the width/height and the FPS of the vide
    int width = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    double FPS = cap.get(CV_CAP_PROP_FPS);

    // Open a video file for writing (the MP4V codec works on OS X and Windows)
    cv::VideoWriter out("output.mov", CV_FOURCC('m','p', '4', 'v'), FPS, cv::Size(width, height));
    if(!out.isOpened()) {
        std::cout <<"Error! Unable to open video file for output." << std::endl;
        std::exit(-1);
    }

    cv::Mat image;

    int finAvg = 0;
    int finArr[360] = {0};
    while(true) {
        cap >> image;
        if(image.empty()) {
            break;
        }

        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        // cv::medianBlur(image, image, 3);
        // cv::Canny(image, image, 100, 300, 3);

        double sobelXFil[3][3] = {{-1,0,1},
                                  {-2,0,2},
                                  {-1,0,1}};

        double sobelYFil[3][3] = {{-1,-2,-1},
                                  {0,0,0},
                                  {1,2,1}};
                                  
        Mat sobelX = convolve(image,sobelXFil);
        Mat sobelY = convolve(image,sobelYFil);

        int angles[width][height];

        int avgRot, avgN, avgSum;
        avgSum = 0;
        avgN = 0;

        for(int i=0; i<width/2; i++){
            for(int j=0; j<height/2; j++){

                angles[i][j] = atan2(sobelY.at<uchar>(i,j),sobelX.at<uchar>(i,j))*180/PI;
                if(angles[i][j] > 5 || angles[i][j] < -5){
                    avgSum+= angles[i][j];
                    avgN++;
                }
            }

        }
        avgRot = avgSum/avgN;
        finArr[avgRot+179]++;
    }

    int finMax = 0;
    int finMaxI = 0;
    for(int i=0; i<360; i++){
        if(finArr[i]>finMax){
            finMax = finArr[i];
            finMaxI = i;
        }
    }
    Point2f src_center(width/2.0F, height/2.0F);
    Mat rot_mat = getRotationMatrix2D(src_center, 90, 1.0);
    while(true) {
        cap >> image;
        if(image.empty()) {
            break;
        }

        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        warpAffine(image, image, rot_mat, image.size());
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);

        // Save frame to video
        out << image;

        cv::imshow("Modified video", image);

        // Stop the camera if the user presses the "ESC" key
        if(cv::waitKey(1000.0/FPS) == 27) break;
    }
    return 0;
}

Mat convolve(Mat src, double avgFilter[3][3]){
    int i, j, k;
    Size s1;
    s1 = src.size();

    int kerSize = 3;

    Mat img3 = Mat::ones(s1.height, s1.width, CV_8UC1);
    for (i = 1; i<s1.height-1; i++){
        for (j = 1; j<s1.width-1; j++){
                k = 0;
                for(int a = 0; a<kerSize; a++){
                    for (int b = 0; b < kerSize; b++){
                        k += src.at<uchar>(i+(a-kerSize/2),j+(b-kerSize/2))*avgFilter[a][b];
                    }
                }
            if(k<0){
                k = 0;
            }
            else if(k>255){
                k = 255;
            }
            img3.at<uchar>(i, j) = k;
            }
        }
    return img3;
}