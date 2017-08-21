#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <Math.h>

#define PI 3.14159265

using namespace std;
using namespace cv;

void shift(Mat magI);
Mat doErosion(Mat src1, int x);
Mat doErosionLine(Mat src1, int x);
Mat doDilation(Mat src1, int x);
Mat doDilationLine(Mat src1, int x);
Mat FT(Mat img);

int countBoxes(Mat src1, int x);
Mat convolve(Mat src, double avgFilter[3][3]);

Mat src1;

Mat img3;
char window_name1[] = "Unprocessed Image";
char window_name2[] = "Processed Image";

int main( int argc, char** argv ){
    cv::VideoCapture cap("../data/input.mp4");
    if(!cap.isOpened()) {
        std::cout << "Unable to open the camera\n";
        std::exit(-1);
    }

    // Get the width/height and the FPS of the video
    int width = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    double FPS = cap.get(CV_CAP_PROP_FPS);

    cv::Mat image;

    int finAvg = 0;
    int finArr[360] = {0};
    int avgRadius = 0;
    int avgRn = 0;

    Point avgCenter(0,0);

    cap >> image;
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    GaussianBlur( image, image, Size(3, 3), 2, 2 );

    vector<Vec3f> circles;

    /// Apply the Hough Transform to find the circles
    HoughCircles( image, circles, CV_HOUGH_GRADIENT, 1, height/8, 200, 100, 0, 0 );
    /// Draw the circles detected
    for( size_t i = 0; i < circles.size(); i++ ){
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);

        avgRadius+=radius;
        avgCenter+=center;
        avgRn++;
        // circle center
        circle( image, center, 3, Scalar(0,255,0), -1, 8, 0 );
        // circle outline
        circle( image, center, radius, Scalar(0,0,255), 3, 8, 0 );
    }
    avgRadius/=avgRn;
    avgCenter/=avgRn;

    int r = 0.7*avgRadius;
    // Open a video file for writing (the MP4V codec works on OS X and Windows)
    cv::VideoWriter out("../data/output.mp4", CV_FOURCC('m','p', '4', 'v'), FPS, cv::Size(2*r, 2*r));
    if(!out.isOpened()) {
        std::cout <<"Error! Unable to open video file for output." << std::endl;
        std::exit(-1);
    }
    while(true) {
        cap >> image;
        if(image.empty()) {
            break;
        }
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

        // // Show your results
        // namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
        // imshow( "Hough Circle Transform Demo", image );
        cv::Mat disc = Mat::zeros(2*r, 2*r, CV_8UC1); 

        for(int i=-r; i<r; i++){
            for(int j=-r; j<r; j++){
                disc.at<uchar>(r+j,r+i) = image.at<uchar>(avgCenter.y+j,avgCenter.x+i);
            }
        }

        cv::cvtColor(disc, disc, cv::COLOR_GRAY2BGR);

        // Save frame to video
        out << disc;
        // cv::imshow("Modified video", disc);
        // Stop the camera if the user presses the "ESC" key
        if(cv::waitKey(1000.0/FPS) == 27) break;
    }

    cv::VideoCapture cap1("../data/output.mp4");
    if(!cap.isOpened()) {
        std::cout << "Unable to open the camera\n";
        std::exit(-1);
    }
    int frams = 0;
    while(true) {
        cap1 >> image;
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

        for(int i=0; i<width; i++){
            for(int j=0; j<height; j++){
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

    finArr[finMaxI] = 0;

    int finMaxI2 = finMaxI;
    for(int i=0; i<360; i++){
        if(finArr[i]>finMax){
            finMax = finArr[i];
            finMaxI = i;
        }
    }

    finArr[finMaxI] = 0;

    int finMaxI3 = finMaxI;
    for(int i=0; i<360; i++){
        if(finArr[i]>finMax){
            finMax = finArr[i];
            finMaxI = i;
        }
    }

    Point2f src_center(r, r);
    Mat rot_mat = getRotationMatrix2D(src_center, 180-(finMaxI+finMaxI2+finMaxI3)/3, 1.0);
    // Open a video file for writing (the MP4V codec works on OS X and Windows)
    cv::VideoWriter out2("../data/output2.mp4", CV_FOURCC('m','p', '4', 'v'), FPS, cv::Size(2*r, 2*r));
    if(!out.isOpened()) {
        std::cout <<"Error! Unable to open video file for output." << std::endl;
        std::exit(-1);
    }

    vector< vector< vector< int > > > fourierPixels;

    frams = 0;

    while(true) {
        cap1 >> image;
        if(image.empty()) {
            break;
        }
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

        // warpAffine(image, image, rot_mat, image.size());

        threshold( image, image, 50, 255,THRESH_BINARY );

        // image = doDilation(image, 10);
        image = doErosionLine(image, 1.8*r);
        image = doDilationLine(image, 1.8*r);

        for(int i=0; i<2*r; i++){
            for(int j=0; j<2*r; j++){
                fourierPixels.push_back(vector<vector<int> >());
                fourierPixels[i].push_back(vector<int>());
                fourierPixels[i][j].push_back(image.at<uchar>(i,j));
            }
        }
        frams++;
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
        // Save frame to video
        out2 << image;
        // cv::imshow("Modified video", image);
        // Stop the camera if the user presses the "ESC" key
        if(cv::waitKey(1000.0/FPS) == 27) break;
    }

    int arr[frams];
    int avgFre = 0;

    int freqAvg = 0;
    int freqN = 0;

    for(int k=0; k<2*r; k++){
        for(int j=0; j<2*r; j++){
            vector<int> v = fourierPixels.at(k).at(j);
            int* arr = &v[0];

            cv::Mat A = Mat(1, frams, CV_32FC1, &arr);
            cv::Mat f = FT(A);

            shift(f);
            Mat planes[] = { Mat::zeros(f.size(), CV_32F), Mat::zeros(f.size(), CV_32F) };

            planes[0] = f;
            planes[1] = f;

            int max = 0;
            int maxI = 0;
            for(int a=0; a<f.cols; a++){
                    int mag = sqrt(pow(planes[0].at<uchar>(a,0),2)+pow(planes[1].at<uchar>(a,0),2));
                    if(mag>max){
                        max = mag;
                        maxI = a;
                    } 
            }
            int val = maxI*(600/f.cols);
            if(val>10){
                freqAvg+=val;
                freqN++;
            }
        }
    }
    printf("%d\n", freqAvg/freqN);
    return 0;
}

void shift(Mat magI) {
  // crop if it has an odd number of rows or columns
  magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

  int cx = magI.cols / 2;
  int cy = magI.rows / 2;

  Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
  Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
  Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
  Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

  Mat tmp;                        // swap quadrants (Top-Left with Bottom-Right)
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);
  q1.copyTo(tmp);                  // swap quadrant (Top-Right with Bottom-Left)
  q2.copyTo(q1);
  tmp.copyTo(q2);
}

Mat FT(Mat img) {
  Size s1 = img.size();
  Mat fImage;
  img.convertTo(fImage, CV_32F);
  Mat fourierTransform;
  dft(fImage, fourierTransform, DFT_SCALE | DFT_COMPLEX_OUTPUT);
  return fourierTransform;
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

Mat doErosion(Mat src1, int x){
	int val=0;
    Mat imgBin = Mat::ones(src1.rows, src1.cols, CV_8UC1);
    for(int i=0; i<src1.rows; i++){
    	for (int j = 0; j < src1.cols; j++){
    		val=0;
    		for (int k = -x/2; k<=x/2; ++k){
    			for (int l = -x/2; l<=x/2; ++l){
    				int r = i+k;
    				int c = j+l;

    				if(r<0 || c<0 || r>=src1.rows || c>=src1.cols){
    					// val = 0;
    				}
    				else{
    					// val = src1.at<uchar>(r,c);
    					if(src1.at<uchar>(r,c)!=0){
    						val++;
    					}
    				}
    			}	
    		}
    		if(val == x*x){
    			imgBin.at<uchar>(i,j) = 255;
    		}
            else{
                imgBin.at<uchar>(i,j) = 0;
            }
        }
    }
    return imgBin;
}


Mat doErosionLine(Mat src1, int x){
    int val=0;
    Mat imgBin = Mat::ones(src1.rows, src1.cols, CV_8UC1);
    for(int i=0; i<src1.rows; i++){
        for (int j = 0; j < src1.cols; j++){
            val=0;
            for (int k = -x/2; k<=x/2; ++k){
                    int c = i+k;
                    int r = j;

                    if(r<0 || c<0 || r>=src1.rows || c>=src1.cols){
                        // val = 0;
                    }
                    else{
                        // val = src1.at<uchar>(r,c);
                        if(src1.at<uchar>(r,c)!=0){
                            val++;
                        }
                    }  
            }
            if(val == x){
                imgBin.at<uchar>(j,i) = 255;
            }
            else{
                imgBin.at<uchar>(j,i) = 0;
            }
        }
    }
    return imgBin;
}


Mat doDilationLine(Mat src1, int x){
    int val=0;
    Mat imgBin = Mat::ones(src1.rows, src1.cols, CV_8UC1);
    for(int i=0; i<src1.rows; i++){
        for (int j = 0; j < src1.cols; j++){
            val=0;
            for (int k = -x/2; k<=x/2; ++k){
                    int c = i+k;
                    int r = j;
                    if(r<0 || c<0 || r>=src1.rows || c>=src1.cols){
                        // val = 0;
                    }
                    else{
                        // val = src1.at<uchar>(r,c);
                        if(src1.at<uchar>(r,c)!=0){
                            val++;
                        }
                    }  
            }
            if(val>0){
                imgBin.at<uchar>(j,i) = 255;
            }
            else{
                imgBin.at<uchar>(j,i) = 0;
            }
        }
    }
    return imgBin;
}


Mat doDilation(Mat src1, int x){
	int val=0;
    Mat imgBin = Mat::ones(src1.rows, src1.cols, CV_8UC1);
    for(int i=0; i<src1.rows; i++){
    	for (int j = 0; j < src1.cols; j++){
    		val=0;
    		for (int k = -x/2; k<=x/2; ++k){
    			for (int l = -x/2; l<=x/2; ++l){
    				int r = i+k;
    				int c = j+l;

    				if(r<0 || c<0 || r>=src1.rows || c>=src1.cols){
    					// val = 0;
    				}
    				else{
    					// val = src1.at<uchar>(r,c);
    					if(src1.at<uchar>(r,c)!=0){
    						val++;
    					}
    				}
    			}	
    		}
    		if(val>0){
    			imgBin.at<uchar>(i,j) = 255;
    		}
            else{
                imgBin.at<uchar>(i,j) = 0;
            }
        }
    }
    return imgBin;
}

int countBoxes(Mat src1, int x){
    int count=0;
    int val=0;
    Mat imgBin = Mat::ones(src1.rows, src1.cols, CV_8UC1);
    for(int i=0; i<src1.rows; i++){
        for (int j = 0; j < src1.cols; j++){
            val=0;
            for (int k = -x/2; k<=x/2; ++k){
                for (int l = -x/2; l<=x/2; ++l){
                    int r = i+k;
                    int c = j+l;

                    if(r<0 || c<0 || r>=src1.rows || c>=src1.cols){
                        // val = 0;
                    }
                    else{
                        // val = src1.at<uchar>(r,c);
                        if(src1.at<uchar>(r,c)!=0){
                            val++;
                        }
                    }
                }   
            }
            if(val == x*x){
                count++;
                j+=x/2;
            }
        }
    }
    return count;
}
