INCLUDE=-I/usr/local/include/opencv -I/usr/local/include/opencv2 -I/usr/local/include/pcl-1.7 -I/usr/local/include/vtk-5.10 -I/usr/local/include/eigen3
LIBS=-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_features2d -lpcl_common -lpcl_visualization -lpcl_io -lpcl_kdtree -lpcl_features -lpcl_surface -lpcl_search -lboost_system -lpcl_filters -lpcl_segmentation
CFLAGS=#-Wall
CC=c++

all:
		$(CC) $(CFLAGS) $(INCLUDE) $(LIBS) -o finder ContourFinder.cpp

clean:
		rm -rf *o finder
