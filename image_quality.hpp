/*
****************************************
* This file contains the code to 
* analyse the frames produced by
* the Block Matching Correlation
* (BMC) algorithm. Two metrics are used
* here :
* 1. Peak Signal to Noise Ratio (PSNR)
* 2. Structural Similarity Index (SSIM)
* This code has been taken from :
* docs.opencv.org
****************************************
*/

#ifndef IMG_QUAL
#define IMG_QUAL

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include "constants.hpp"
#include "util.hpp"

using namespace cv;
using namespace std;

double getPSNR(const Mat &I1, const Mat &I2);
Scalar getMSSIM(const Mat &i1, const Mat &i2);
void writeValuesToFile(ofstream &resFile, int frameNo, double psnr, Scalar mssim);
void calcQuality();

#define ANALYSIS_FILE "analysis/image_quality.txt"

#endif
