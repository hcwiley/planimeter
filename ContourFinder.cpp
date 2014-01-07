#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"


#include <time.h>
#include <ctype.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cstdio>
#include <cmath>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/visualization/pcl_visualizer.h>


#define ESC_KEY 27
#define T_KEY 116
#define SAPCE_BAR 32
#define UP_KEY 63232
#define LEFT_KEY 63234
#define DOWN_KEY 63233
#define RIGHT_KEY 63235
#define S_KEY 115


using namespace std;
using namespace cv;

// Forwards
//void Filter ( IplImage * img );
//void Find ( IplImage * src, IplImage * img );

//
// Global variables
int		width = 1;
int		height = 1;
CvFont	_idFont;
//
int min_contour_area = 1000;

cv::RNG rng(12345);

// various tracking parameters (in seconds)
const double MHI_DURATION = 1;
const double MAX_TIME_DELTA = 0.5;
const double MIN_TIME_DELTA = 0.05;

typedef cv::vector<cv::vector<cv::Point> > TContours;

TContours templateContour;
TContours templateContours;
TContours liveContour;
cv::vector<cv::Point> origins;
int SCALE_REF = 1;
cv::vector<int> areas;

//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
pcl::visualization::PCLVisualizer viewer ("Volume");

Mat frame, frameCopy, image, grayMat, differenceMat;//edgesMat;
Mat templateMat, mhiMat;

void newTemplateImage(cv::Mat * templateMat, cv::Mat * grayMat, cv::Mat * differenceMat){
  printf("capture new template image\n");
  *templateMat = grayMat->clone();
  cv::absdiff( *grayMat, *grayMat, *differenceMat); // get difference between frames
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

  cv::findContours( canny_output, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
  //cv::findContours( canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

  // Print number of found contours.
  //std::cout << "Found " << contours.size() << " contours." << std::endl;

  /// Draw contours
  int first = 0;
  int num_over_min_area = 0;
  origins.clear();
  areas.clear();
  templateContours.clear();
  viewer.removeAllPointClouds();
  for( int i = contours.size() - 1; i >= 0; i-- )
  {
    bool contains = false;
    if( contourArea(contours[i]) > min_contour_area){
      for( int j = 0; j< origins.size(); j++ ){
        //printf("origins: %d, %d\n", origins[j].x, origins[j].y);
        //printf("contours: %d, %d\n", contours[j][0].x, contours[j][0].y);
        int bounds = 150;
        int origX = origins[j].x;
        int origY = origins[j].y;
        int conX = contours[i][0].x;
        int conY = contours[i][0].y;
        int conArea = contourArea(contours[i]);
        if( origX + bounds >= conX && origX - bounds <= conX ){
          if( origY + bounds >= conY && origY - bounds <= conY ){
            if( areas[j] * 1.20 >= conArea && areas[j] * 0.80 <= conArea ){
              //printf("contains!!\n");
              contains = true;
            }
          }
        }
        //printf("-------------\n");
      }
      if(contains)
        continue;
      if ( first < i){
        templateContour.clear();
      templateContour.push_back(contours[i]);
        first = i;
      }
      origins.push_back(contours[i][0]);
      templateContours.push_back(contours[i]);
      double area0 = contourArea(contours[i]);
      areas.push_back(area0);
      templateContour[0].insert(templateContour[0].end(), contours[i].begin(), contours[i].end());
      num_over_min_area++;
      std::cout << "Creating Point Cloud..." <<std::endl;
      pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
      for(int p = 0; p < contours[i].size(); p++){
        //Insert info into point cloud structure
        pcl::PointXYZ point;
        point.x = contours[i][p].x;
        point.y = contours[i][p].y;
        point.z = 40;
        //point.rgb = *reinterpret_cast<float*>(&rgb);
        point_cloud_ptr->points.push_back (point);
      }
      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler (point_cloud_ptr, 255 * i, 0, 255);
      char name[20];
      sprintf(name, "volume%d", i);
      viewer.addPointCloud<pcl::PointXYZ> (point_cloud_ptr, handler, name);
      /*
      // Normal estimation*
      pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
      pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
      pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
      tree->setInputCloud (point_cloud_ptr);
      n.setInputCloud (point_cloud_ptr);
      n.setSearchMethod (tree);
      n.setKSearch (20);
      n.compute (*normals);
      //* normals should not contain the point normals + surface curvatures

      // Concatenate the XYZ and normal fields*
      pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
      pcl::concatenateFields (*point_cloud_ptr, *normals, *cloud_with_normals);
      //* cloud_with_normals = cloud + normals

      // Create search tree*
      pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
      tree2->setInputCloud (cloud_with_normals);

      // Initialize objects
      pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
      pcl::PolygonMesh triangles;

      // Set the maximum distance between connected points (maximum edge length)
      gp3.setSearchRadius (0.025);

      // Set typical values for the parameters
      gp3.setMu (2.5);
      gp3.setMaximumNearestNeighbors (100);
      gp3.setMaximumSurfaceAngle(M_PI/4); // 45 degrees
      gp3.setMinimumAngle(M_PI/18); // 10 degrees
      gp3.setMaximumAngle(2*M_PI/3); // 120 degrees
      gp3.setNormalConsistency(false);

      // Get result
      gp3.setInputCloud (cloud_with_normals);
      gp3.setSearchMethod (tree2);
      gp3.reconstruct (triangles);
      */
    }
  }
  //cv::convexHull(templateContour[0], templateContour[0], false);
  cv::Scalar color = cv::Scalar( 0, 0, 255);
  drawContours( *differenceMat, templateContour, 0, color, 2, 8, hierarchy, 0, cv::Point() );
  viewer.addCoordinateSystem(1.0,0);

  for( int j = 0; j< origins.size(); j++ ){
    char str[40];
    double scale = 1000*1000/(SCALE_REF); // 1946 is a magic number that needs to be derived programmatically eventually
    double area0 = areas[j];
    area0 *= scale; // to sq ft
    area0 /= 43560; // to acres
    sprintf(str, "area: %.3f acres" , area0);
    putText(*differenceMat, str, origins[j], cv::FONT_HERSHEY_PLAIN, 1, Scalar( 0,255,0), 1, 8, false);
  }
}

void liveWindowClickCallback(int event, int x, int y, int flags, void* userdata)
{
  if  ( event == EVENT_LBUTTONDOWN )
  {
    cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    for( int i = 0; i< origins.size(); i++ ){
      printf("origins: %d, %d\n", origins[i].x, origins[i].y);
      int origX = origins[i].x;
      int origY = origins[i].y;
      int bounds = 20;
      int isIn = -1;
      //printf("contours size: %d (%d)\n", templateContours.size(), i);
      if(templateContours.size() > i)
        isIn = pointPolygonTest(templateContours[i], Point(x,y), false);
      if(isIn == 1){
        printf("found the origin you want: %d!!\n", isIn);
        SCALE_REF = areas[i];
        newTemplateImage(&templateMat, &grayMat, &differenceMat);
      }
    }
  }
  else if  ( event == EVENT_RBUTTONDOWN )
  {
    cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
  }
  else if  ( event == EVENT_MBUTTONDOWN )
  {
    cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
  }
  else if ( event == EVENT_MOUSEMOVE )
  {
    //cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;

  }
}

int main( int argc, const char** argv )
{
  CvCapture* capture = 0;

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
  cvInitFont ( &_idFont, CV_FONT_VECTOR0, 0.5, 0.5, 0.0, 1 );



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
  //cvNamedWindow( "template", 1 );

  //viewer.setSize (800, 600);

  cv::resizeWindow("live", 640, 480);//50, 50);
  cv::resizeWindow("template", 640, 480);//700, 50);
  cv::resizeWindow("difference", 640, 480);// 700, 500);
  //cv::resizeWindow("gray", 640, 480);// 50, 500);

  cv::moveWindow("live", 50, 50);
  cv::moveWindow("template", 700, 50);
  cv::moveWindow("difference", 700, 50);
  //cv::moveWindow("gray", 50, 500);
  //
  cv::setMouseCallback("live", liveWindowClickCallback, NULL);

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
    frame = cv::imread("contours.jpg", CV_LOAD_IMAGE_COLOR);
    while(true)
    {

      double timestamp = (double)clock()/CLOCKS_PER_SEC; // get current time in seconds

      //img = cvQueryFrame( capture );
      //frame = img;

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

      //cv::findContours( canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
      //cv::findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );

      // Print number of found contours.
      //std::cout << "Found " << contours.size() << " contours." << std::endl;


      /// Draw contours
      //int first = contours.size() + 1;
      //for( int i = 0; i< contours.size(); i++ )
      //{
      //if( contourArea(contours[i]) > min_contour_area){
      //if ( first > i){
      //liveContour.clear();
      //liveContour.push_back(contours[i]);
      //first = i;
      //}
      //liveContour[0].insert(liveContour[0].end(), contours[i].begin(), contours[i].end());
      //cv::Scalar color = cv::Scalar(50*i,0,0);
      //drawContours( frame, contours, 0, color, 2, 8, hierarchy, 0, cv::Point() );
      //}
      //}

      //cv::convexHull(liveContour[0], liveContour[0], false);

      //cvtColor (frame, grayMat, CV_BGR2GRAY );
      grayMat= frame.clone();

      *gray = grayMat;
      if(templateMat.empty() ){
        newTemplateImage(&templateMat, &grayMat, &differenceMat);
      }

      //if(!templateContour.empty()){
      //drawContours( frame, templateContour, 0, cv::Scalar(0,255,0), 2, 8, hierarchy, 0, cv::Point() );
      //double area0 = contourArea(liveContour);
      //printf("contour: %f\n", area0);
      //}


      int key = cv::waitKey( 10 );
      if (key >= 0){
        switch ( key ) {
          case ESC_KEY:
            run = false;
            break;
          case T_KEY:
            //newTemplateImage(&templateMat, &grayMat);
            newTemplateImage(&templateMat, &grayMat, &differenceMat);
            break;
          case UP_KEY:
            min_contour_area += 10000;
            printf("min_contour_area: %d\n", min_contour_area);
            newTemplateImage(&templateMat, &grayMat, &differenceMat);
            break;
          case DOWN_KEY:
            min_contour_area -= 10000;
            printf("min_contour_area: %d\n", min_contour_area);
            newTemplateImage(&templateMat, &grayMat, &differenceMat);
            break;
          case LEFT_KEY:
            min_contour_area -= 100;
            printf("min_contour_area: %d\n", min_contour_area);
            newTemplateImage(&templateMat, &grayMat, &differenceMat);
            break;
          case RIGHT_KEY:
            min_contour_area += 100;
            printf("min_contour_area: %d\n", min_contour_area);
            newTemplateImage(&templateMat, &grayMat, &differenceMat);
            break;
          case S_KEY:
            // save the template image
            break;
          default:
            printf("%d key hit\n", key);
            break;
        }
      }




      if(!templateMat.empty() ){
        *difference = differenceMat;
        cvShowImage( "difference", difference );
      }
      *img = frame;
      *templateImg = templateMat;


      cvShowImage( "live", img );
      //viewer.spin();
      
      //cvShowImage( "gray", gray );
      //cvShowImage( "template", templateImg);


      if(!run)
        break;
    }
  }

  //cvReleaseCapture( &capture );
  //cvReleaseImage ( &img );
  //cvReleaseImage ( &edges);
  //cvReleaseImage ( &templateImg);
  //cvReleaseImage ( &difference);
  //cvReleaseImage ( &gray );
  cvDestroyWindow( "live" );
  cvDestroyWindow( "difference" );
  //cvDestroyWindow( "gray" );
  //cvDestroyWindow( "edges" );
  //cvDestroyWindow( "template" );

  return 0;
}
