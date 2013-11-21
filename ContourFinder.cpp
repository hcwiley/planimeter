#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"


#include <time.h>
#include <ctype.h>
#include <iostream>
//#include <stdlib.h>
#include <stdio.h>
//#include <cstdio>
//#include <cmath>


#define ESC_KEY 27
#define T_KEY 116
#define SAPCE_BAR 32
#define UP_KEY 63232
#define LEFT_KEY 63234
#define DOWN_KEY 63233
#define RIGHT_KEY 63235


using namespace std;
//using namespace cv;

// Forwards
//void Filter ( IplImage * img );
//void Find ( IplImage * src, IplImage * img );

//
// Global variables
int		width = 1;
int		height = 1;
//CvFont	_idFont;

cv::RNG rng(12345);

// various tracking parameters (in seconds)
const double MHI_DURATION = 1;
const double MAX_TIME_DELTA = 0.5;
const double MIN_TIME_DELTA = 0.05;

typedef cv::vector<cv::vector<cv::Point> > TContours;

TContours templateContour;
TContours liveContour;

void newTemplateImage(cv::Mat * templateMat, cv::Mat * grayMat){
  printf("capture new template image\n");
  *templateMat = grayMat->clone();
  /*
   * Find features
   */
  //std::vector<cv::Point2f> corners;
  //cv::Scalar color = cv::Scalar(255,0,0);
  //cv::goodFeaturesToTrack(*templateMat,corners, 500, 0.01, 10);
  //for (size_t idx = 0; idx < corners.size(); idx++) {
  //cv::circle(*templateMat, corners.at(idx), 3, color);
  //}

  /*
   * Do contours
   */
  TContours contours;
  std::vector<cv::Vec4i> hierarchy;

  cv::Mat canny_output;//  = cv::Mat::zeros( frame.size(), CV_8UC3 );

  Canny(*templateMat, canny_output, 100, 300, 3);

  //cv::findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
  cv::findContours( canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

  // Print number of found contours.
  //std::cout << "Found " << contours.size() << " contours." << std::endl;

  double area0 = contourArea(templateContour[0]);
  printf("contour: %f\n", area0);
  /// Draw contours
  int first = contours.size() + 1;
  for( int i = 0; i< contours.size(); i++ )
  {
    if( contourArea(contours[i]) > 10){
      if ( first > i){
        templateContour.clear();
        templateContour.push_back(contours[i]);
        first = i;
      }
      templateContour[0].insert(templateContour[0].end(), contours[i].begin(), contours[i].end());
    }
  }
  //cv::convexHull(templateContour[0], templateContour[0], false);
  cv::Scalar color = cv::Scalar( 0,255,0);
  drawContours( *templateMat, templateContour, 0, color, 2, 8, hierarchy, 0, cv::Point() );
}


int main( int argc, const char** argv )
{
  CvCapture* capture = 0;
  cv::Mat frame, frameCopy, image, grayMat, differenceMat;//edgesMat;
  cv::Mat templateMat, mhiMat;

  IplImage	*	img		= NULL;
  IplImage	*	gray	= NULL;
  //IplImage	*	edges	= NULL;
  IplImage	*	difference = NULL;
  IplImage	*	templateImg = NULL;
  //float			angle	= 0.0;
  cv::Scalar lowerThresh = cv::Scalar(0,0,0);
  cv::Scalar upperThresh = cv::Scalar(45,45,45);
  //cv::Moments momentsTemplate = 0;
  //cv::Moments momentsLive = 0;
  //
  cv::vector<cv::Point> foo;
  foo.push_back(cv::Point());
  templateContour.push_back(foo);
  liveContour.push_back(foo);

  //initSocket();
  //templateContour = cv::vector<cv::vector<cv::Point> >;

  bool run = true;

  // Initialize the Font used to draw into the source image
  //cvInitFont ( &_idFont, CV_FONT_VECTOR0, 0.5, 0.5, 0.0, 1 );



  //capture = cvCaptureFromCAM( 0 ); //0=default, -1=any camera, 1..99=your camera
  //if(!capture) cout << "No camera detected" << endl;

  //cvSetCaptureProperty ( capture, CV_CAP_PROP_FRAME_WIDTH, 640 );
  //cvSetCaptureProperty ( capture, CV_CAP_PROP_FRAME_HEIGHT, 480 );

  //width	= (int) cvGetCaptureProperty ( capture, CV_CAP_PROP_FRAME_WIDTH );
  //height	= (int) cvGetCaptureProperty ( capture, CV_CAP_PROP_FRAME_HEIGHT );
  width = 640;
  height = 480;

  cvNamedWindow( "live", 1 );
  //cvNamedWindow( "gray", 1 );
  //cvNamedWindow( "edges", 1 );
  cvNamedWindow( "difference", 1 );
  cvNamedWindow( "template", 1 );

  cv::resizeWindow("live", 640, 480);//50, 50);
  cv::resizeWindow("template", 640, 480);//700, 50);
  cv::resizeWindow("difference", 640, 480);// 700, 500);
  //cv::resizeWindow("gray", 640, 480);// 50, 500);

  cv::moveWindow("live", 50, 50);
  cv::moveWindow("template", 700, 50);
  cv::moveWindow("difference", 1200, 50);
  //cv::moveWindow("gray", 50, 500);

  gray = cvCreateImage ( cvSize ( width, height ), IPL_DEPTH_8U, 1 );
  if ( !gray ) {
    printf ( "failed to create gray image!!!\n" );
    exit(-1);
  }

  img = cvCreateImage ( cvSize ( width, height ), IPL_DEPTH_8U, 1 );
  //if ( !edges ) {
  //printf ( "Failed to create edges image!!!\n" );
  //exit(-1);
  //}

  templateImg = cvCreateImage ( cvSize ( width, height ), IPL_DEPTH_8U, 1 );
  if ( !templateImg ) {
    printf ( "Failed to create templateImg image!!!\n" );
    exit(-1);
  }

  difference = cvCreateImage ( cvSize ( width, height ), IPL_DEPTH_8U, 3 );
  if ( !difference ) {
    printf ( "failed to create difference image!!!\n" );
    exit(-1);
  }

  //
  // Set Region of Interest to the whole image
  cvResetImageROI ( gray );
  gray->origin = 1;


  if( true )
  {
    cout << "In capture ..." << endl;
    while(run)
    {

      double timestamp = (double)clock()/CLOCKS_PER_SEC; // get current time in seconds

      //img = cvQueryFrame( capture );
      //frame = img;
      frame = cv::imread("contours.jpg", CV_LOAD_IMAGE_COLOR);

      if( frame.empty() )
        break;
      //if( img->origin == IPL_ORIGIN_TL )
        //frame.copyTo( frameCopy );
      //else
        //flip( frame, frameCopy, 0 );


      //cv::Mat differenceMat(frame);

      cv::inRange(frame, lowerThresh, upperThresh, grayMat);


      //GaussianBlur(frame, frame, cv::Size(7,7), 1.5, 1.5);

      TContours contours;
      std::vector<cv::Vec4i> hierarchy;

      cv::Mat canny_output;//  = cv::Mat::zeros( frame.size(), CV_8UC3 );

      Canny(grayMat, canny_output, 10, 50, 3);
      //Canny(grayMat, canny_output, 1, 100, 3);

      cv::findContours( canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
      //cv::findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

      // Print number of found contours.
      //std::cout << "Found " << contours.size() << " contours." << std::endl;


      /// Draw contours
      int first = contours.size() + 1;
      for( int i = 0; i< contours.size(); i++ )
      {
        if( contourArea(contours[i]) > 10){
          if ( first > i){
            liveContour.clear();
            liveContour.push_back(contours[i]);
            first = i;
          }
          liveContour[0].insert(liveContour[0].end(), contours[i].begin(), contours[i].end());
        }
      }

      //cv::convexHull(liveContour[0], liveContour[0], false);
      cv::Scalar color = cv::Scalar(255,0,255);
      drawContours( frame, liveContour, 0, color, 2, 8, hierarchy, 0, cv::Point() );

      cvtColor (frame, grayMat, CV_BGR2GRAY );

      *gray = grayMat;
      if(templateMat.empty() ){
        newTemplateImage(&templateMat, &grayMat);
      }

      if(!templateContour.empty()){
        drawContours( frame, templateContour, 0, cv::Scalar(0,255,0), 2, 8, hierarchy, 0, cv::Point() );
        //double area0 = contourArea(liveContour);
        //printf("contour: %f\n", area0);
      }


      int key = cv::waitKey( 10 );
      if (key >= 0){
        switch ( key ) {
          case ESC_KEY:
            run = false;
            break;
          case T_KEY:
            newTemplateImage(&templateMat, &grayMat);
            break;
          case UP_KEY:
            upperThresh = cv::Scalar(upperThresh.val[0] + 1, upperThresh.val[1] + 1, upperThresh.val[2] + 1);
            printf("upperThresh: %f, %f, %f\n", upperThresh.val[0], upperThresh.val[1], upperThresh.val[2]);
            break;
          case DOWN_KEY:
            upperThresh = cv::Scalar(upperThresh.val[0] - 1, upperThresh.val[1] - 1, upperThresh.val[2] - 1);
            printf("upperThresh: %f, %f, %f\n", upperThresh.val[0], upperThresh.val[1], upperThresh.val[2]);
            break;
          case LEFT_KEY:
            lowerThresh = cv::Scalar(lowerThresh.val[0] + 1, lowerThresh.val[1] + 1, lowerThresh.val[2] + 1);
            printf("lowerThresh: %f, %f, %f\n", lowerThresh.val[0], lowerThresh.val[1], lowerThresh.val[2]);
            break;
          case RIGHT_KEY:
            lowerThresh = cv::Scalar(lowerThresh.val[0] - 1, lowerThresh.val[1] - 1, lowerThresh.val[2] - 1);
            printf("lowerThresh: %f, %f, %f\n", lowerThresh.val[0], lowerThresh.val[1], lowerThresh.val[2]);
            break;
          default:
            printf("%d key hit\n", key);
            break;
        }
      }


      cv::absdiff( grayMat, templateMat, differenceMat); // get difference between frames


      *difference = differenceMat;
      *img = frame;
      *templateImg = templateMat;


      cvShowImage( "live", img );
      //cvShowImage( "gray", gray );
      cvShowImage( "difference", difference );
      cvShowImage( "template", templateImg);


    }
  }

  //cvReleaseCapture( &capture );
  cvReleaseImage ( &img );
  //cvReleaseImage ( &edges);
  //cvReleaseImage ( &templateImg);
  //cvReleaseImage ( &difference);
  //cvReleaseImage ( &gray );
  cvDestroyWindow( "live" );
  cvDestroyWindow( "difference" );
  //cvDestroyWindow( "gray" );
  //cvDestroyWindow( "edges" );
  cvDestroyWindow( "template" );

  return 0;
}
