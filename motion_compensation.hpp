/*
****************************************
* This file contains the code to 
* contruct the interpolated frame.
* Author : Shehyaaz Khan Nayazi
****************************************
*/
#ifndef MOTION_HPP
#define MOTION_HPP

#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

void bidirectionalMotionCompensation(const vector<vector<UMat>> &prevBlocks, const UMat &curr, const vector<vector<Point2f>> &prevBlocksMV, UMat &newFrame);

#endif