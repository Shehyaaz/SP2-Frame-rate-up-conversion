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
    vector<UMat> frames;
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

    void divideIntoGlobal(const UMat &inpFrame, vector<UMat> &globalRegions);
    void divideIntoLocal(const UMat &inpFrame, vector<UMat> &localRegions);
    void divideIntoBlocks(const UMat &inpFrame, vector<vector<UMat>> &blockRegions);
    void customisedPhaseCorr(const UMat &prev, const UMat &curr);
    void blockMatching(const UMat &prev, const UMat &curr);
    void BMC(const UMat &prev, const UMat &curr, UMat &interpolatedFrame);
    void interpolate();
};

static const vector<vector<Point2f>> zeroes = vector<vector<Point2f>>(NUM_BLOCKS_Y, vector<Point2f>(NUM_BLOCKS_X, Point2f(0, 0)));
static const Size stdSize = Size(STANDARD_REGION_WIDTH, STANDARD_REGION_HEIGHT);

#define EXEC_TIME_FILE "execution-time.txt"

#endif
