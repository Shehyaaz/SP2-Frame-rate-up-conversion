/*
****************************************
* This file contains the utility 
* functions used in the algorithm.
* Author : Shehyaaz Khan Nayazi
****************************************
*/

#ifndef UTIL_HPP
#define UTIL_HPP

#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <fstream>
#include "constants.hpp"
#include "opencv_methods.hpp"

using namespace cv;
using namespace std;

vector<UMat> readFrames(String videoFile);
vector<Point2f> phaseCorr(InputArray _src1, InputArray _src2, InputArray _window, double *response);
float calcSAD(UMat prevBlock, int rowpos, int colpos, UMat curr, float dx, float dy);
Point2f medianNeighbor(int rowpos, int colpos, vector<vector<Point2f>> &prevBlockMV);
UMat getPaddedROI(const UMat &input, int top_left_x, int top_left_y, int width, int height, Scalar paddingColor);
void writeToFile(ofstream &file, chrono::milliseconds duration);

#endif
