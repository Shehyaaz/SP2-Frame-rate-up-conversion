/*
****************************************
* This file contains the declaration
* of the Block Matching Correlation
* (BMC) algorithm.
* Author : Shehyaaz Khan Nayazi
****************************************
*/

/*
* NOTE : a) x is in the direction of width, i.e., x will be the number of columns
*        b) y is in the direction of height, i.e., y will be the number of rows 
*/
#ifndef BMC_HPP

#define BMC_HPP

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include "constants.hpp"

using namespace cv;
using namespace std;

class BlockMatchingCorrelation
{
    // variable declarations
    vector<Mat> images;
    vector<Mat> interpolated;
    vector<vector<vector<Point2f>>> globalRegionMV;
    vector<vector<vector<Point2f>>> localRegionMV;
    vector<vector<Point2f>> prevBlockMV;
    vector<vector<Point2f>> currBlockMV;

public:
    // function declarations
    BlockMatchingCorrelation()
        : globalRegionMV(NUM_GR_Y, vector<vector<Point2f>>(NUM_GR_X, vector<Point2f>(2, Point2f(0, 0)))),
          localRegionMV(NUM_LR_Y, vector<vector<Point2f>>(NUM_LR_X, vector<Point2f>(2, Point2f(0, 0)))),
          prevBlockMV(NUM_BLOCKS_Y, vector<Point2f>(NUM_BLOCKS_X, Point2f(0, 0))),
          currBlockMV(NUM_BLOCKS_Y, vector<Point2f>(NUM_BLOCKS_X, Point2f(0, 0)))

    {
        // initialization of variables
    }

    vector<Mat> divideIntoGlobal(Mat inpImg);
    vector<Mat> divideIntoLocal(Mat inpImg);
    vector<vector<Mat>> divideIntoBlocks(Mat inpImg);
    void customisedPhaseCorr(Mat prev, Mat curr);
    void blockMatching(Mat prev, Mat curr);
    Mat BMC(Mat prev, Mat curr);
    void interpolate();
};

static const vector<vector<Point2f>> zeroes = vector<vector<Point2f>>(NUM_BLOCKS_Y, vector<Point2f>(NUM_BLOCKS_X, Point2f(0, 0)));
static const Size stdSize = Size(STANDARD_REGION_WIDTH, STANDARD_REGION_HEIGHT);

#define INTERPOLATED_FRAME_FOLDER "interpolated/frame"
#define EXEC_TIME_FILE "analysis/execution-time.txt"

#endif