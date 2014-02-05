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
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>


#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/concave_hull.h>


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

int volumeCount = 0;
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
  for( int i = contours.size() - 1; i >= 0; i-- ) {
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
    }
  }
  std::cout << "Creating Point Cloud..." <<std::endl;
  //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>), 
                                  cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>), 
                                  cloud_projected (new pcl::PointCloud<pcl::PointXYZ>);

  double scale = 1000/(SCALE_REF); // 1946 is a magic number that needs to be derived programmatically eventually
  int min_x, min_y, max_x, max_y;
  min_x = min_y =  1000000;
  max_x = max_y = -1000000;
  for( int i = templateContours.size() - 1; i >= 0; i-- ) {
    if( contourArea(templateContours[i]) < 2100 ) {
      continue;
    }
    for(int p = 0; p < templateContours[i].size(); p++){
      //Insert info into point cloud structure
      if( contourArea(templateContours[i]) > 21000 ) {
        pcl::PointXYZ point;
        point.x = templateContours[i][p].x;
        point.y = templateContours[i][p].y;
        point.z = 0;
        cloud->points.push_back (point);
        if(point.x > max_x)
          max_x = point.x;
        else if(point.x < min_x)
          min_x = point.x;
        if(point.y > max_y)
          max_y = point.y;
        else if(point.y < min_y)
          min_y = point.y;
      }
      else if( contourArea(templateContours[i]) > 8500 ) {
        pcl::PointXYZ point;
        point.x = templateContours[i][p].x;
        point.y = templateContours[i][p].y;
        point.z = 20;
        cloud->points.push_back (point);
        point.x = templateContours[i][p].x;
        point.y = templateContours[i][p].y;
        point.z = -20;
        cloud->points.push_back (point);
      }
      else {
        pcl::PointXYZ point;
        point.x = templateContours[i][p].x;
        point.y = templateContours[i][p].y;
        point.z = 40;
        cloud->points.push_back (point);
        point.x = templateContours[i][p].x;
        point.y = templateContours[i][p].y;
        point.z = -40;
        cloud->points.push_back (point);
      }
    }
  }
  printf("min (%d, %d)\nmax (%d, %d)\n", min_x,min_y,max_x,max_y);
  // loop through between min and max x and y
  for( int x = min_x; x < max_x; x+=5 ) {
    for( int y = min_y; y < max_y; y+=5 ) {
      // loop through all the contours
      for( int i = templateContours.size() - 1; i >= 0; i-- ) {
        // if the point is in the contour, lets do it
        if( pointPolygonTest(templateContours[i], cv::Point(x,y), false) > 0 ) {
          // if its in the main area
          if( contourArea(templateContours[i]) > 21000 ) {
            pcl::PointXYZ point;
            point.x = x;
            point.y = y;
            point.z = 0;
            cloud->points.push_back (point);
          }
          // if its in the middle sizes
          else if( contourArea(templateContours[i]) > 8500 ) {
            pcl::PointXYZ point;
            point.x = x;
            point.y = y;
            point.z = 20;
            cloud->points.push_back (point);
            point.x = x;
            point.y = y;
            point.z = -20;
            cloud->points.push_back (point);
          }
          // other wise its a small one
          else{
            pcl::PointXYZ point;
            point.x = x;
            point.y = y;
            point.z = 40;
            cloud->points.push_back (point);
            point.x = x;
            point.y = y;
            point.z = -40;
            cloud->points.push_back (point);
          }
        }
      }
    }
  }
  printf("done adding extra points\n");
  char name[20];
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> handler (cloud, 0, 0, 255);
  sprintf(name, "volume%d", volumeCount);
  viewer.addPointCloud<pcl::PointXYZ> (cloud, handler, name);
  
  // Normal estimation*
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud);
  n.setInputCloud (cloud);
  n.setSearchMethod (tree);
  n.setKSearch (21);
  n.compute (*normals);
  //* normals should not contain the point normals + surface curvatures
  
  // Concatenate the XYZ and normal fields*
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
  pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);
  //* cloud_with_normals = cloud + normals
  
  // Create search tree*
  pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
  tree2->setInputCloud (cloud_with_normals);
  
  // Initialize objects
  pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
  pcl::PolygonMesh triangles;
  
  // Set the maximum distance between connected points (maximum edge length)
  gp3.setSearchRadius (10);
  
  // Set typical values for the parameters
  gp3.setMu (3.5);
  gp3.setMaximumNearestNeighbors (21);
  gp3.setMaximumSurfaceAngle(M_PI/2); // 45 degrees
  gp3.setMinimumAngle(M_PI/36); // 10 degrees
  gp3.setMaximumAngle(M_PI/2); // 120 degrees
  gp3.setNormalConsistency(false);
  
  // Get result
  gp3.setInputCloud (cloud_with_normals);
  gp3.setSearchMethod (tree2);
  gp3.reconstruct (triangles);
  
  // Additional vertex information
  std::vector<int> parts = gp3.getPartIDs();
  std::vector<int> states = gp3.getPointStates();

  std::cerr << "Shape has: " << parts.size ()
            << " parts." << std::endl;

  sprintf(name, "shape-%d", volumeCount++);
  viewer.addPolygonMesh(triangles, name, 0);

  //cv::convexHull(templateContour[0], templateContour[0], false);
  cv::Scalar color = cv::Scalar( 0, 0, 255);
  drawContours( *differenceMat, templateContour, 0, color, 2, 8, hierarchy, 0, cv::Point() );
  viewer.addCoordinateSystem(1.0,0);

  scale = 1000*1000/(SCALE_REF); // 1946 is a magic number that needs to be derived programmatically eventually
  for( int j = 0; j< origins.size(); j++ ){
    char str[40];
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
  IplImage	*	difference = NULL;
  IplImage	*	templateImg = NULL;
  cv::Scalar lowerThresh = cv::Scalar(0,0,0);
  cv::Scalar upperThresh = cv::Scalar(45,45,45);
  cv::vector<cv::Point> foo;
  foo.push_back(cv::Point());
  templateContour.push_back(foo);
  liveContour.push_back(foo);

  bool run = true;

  // Initialize the Font used to draw into the source image
  cvInitFont ( &_idFont, CV_FONT_VECTOR0, 0.5, 0.5, 0.0, 1 );



  width = 640;
  height = 480;

  cvNamedWindow( "live", 1 );
  cvNamedWindow( "difference", 1 );

  cv::resizeWindow("live", 640, 480);//50, 50);
  cv::resizeWindow("template", 640, 480);//700, 50);
  cv::resizeWindow("difference", 640, 480);// 700, 500);

  cv::moveWindow("live", 450, 50);
  cv::moveWindow("template", 1400, 50);
  cv::moveWindow("difference", 1400, 50);

  // click handler binding
  cv::setMouseCallback("live", liveWindowClickCallback, NULL);

  gray = cvCreateImage ( cvSize ( width, height ), IPL_DEPTH_8U, 1 );
  if ( !gray ) {
    printf ( "failed to create gray image!!!\n" );
    exit(-1);
  }

  img = cvCreateImage ( cvSize ( width, height ), IPL_DEPTH_8U, 1 );

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

      if( frame.empty() )
        break;

      cv::inRange(frame, lowerThresh, upperThresh, grayMat);


      TContours contours;
      std::vector<cv::Vec4i> hierarchy;

      cv::Mat canny_output;//  = cv::Mat::zeros( frame.size(), CV_8UC3 );

      Canny(grayMat, canny_output, 10, 50, 3);
      grayMat= frame.clone();

      *gray = grayMat;
      if(templateMat.empty() ){
        newTemplateImage(&templateMat, &grayMat, &differenceMat);
      }


      int key = cv::waitKey( 10 );
      if (key >= 0){
        switch ( key ) {
          case ESC_KEY:
            run = false;
            break;
          case T_KEY:
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
