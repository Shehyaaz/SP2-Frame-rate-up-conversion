/*
****************************************
* This file contains the implementation
* of the Block Matching Correlation
* (BMC) algorithm.
* Author : Shehyaaz Khan Nayazi
****************************************
*/

/*
* NOTE : a) x is in the direction of width, i.e., x will be the number of columns
*        b) y is in the direction of height, i.e., y will be the number of rows 
*/

#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include "constants.hpp"
#include "util.hpp"

using namespace cv;
using namespace std;

class BlockMatchingCorrelation
{
    // variable declarations
    vector<Mat> images;
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
    void BMC(Mat prev, Mat curr);
    void interpolate();
};

vector<Mat> BlockMatchingCorrelation::divideIntoGlobal(Mat inpImg)
{
    vector<Mat> globalRegions(NUM_GR_X * NUM_GR_Y);
    Size globalRegionSize(GR_WIDTH, GR_HEIGHT);
    for (int y = 0; y < inpImg.rows - GR_HEIGHT; y += globalRegionSize.height)
    {
        for (int x = 0; x < inpImg.cols - GR_WIDTH; x += inpImg.cols - GR_WIDTH)
        {
            Rect rect = Rect(x, y, globalRegionSize.width, globalRegionSize.height);
            globalRegions.push_back(Mat(inpImg, rect));
        }
    }
    Rect rect = Rect(inpImg.cols - GR_WIDTH, inpImg.rows - GR_HEIGHT, globalRegionSize.width, globalRegionSize.height);
    globalRegions.push_back(Mat(inpImg, rect));
    return globalRegions;
}

vector<Mat> BlockMatchingCorrelation::divideIntoLocal(Mat inpImg)
{
    vector<Mat> localRegions(NUM_LR_X * NUM_LR_Y);
    Size localRegionSize(LR_WIDTH, LR_HEIGHT);
    for (int y = 0; y < inpImg.rows - LR_HEIGHT; y += localRegionSize.height)
    {
        for (int x = 0; x < inpImg.cols - LR_WIDTH; x += localRegionSize.width)
        {
            Rect rect = Rect(x, y, localRegionSize.width, localRegionSize.height);
            localRegions.push_back(Mat(inpImg, rect));
        }
    }
    Rect rect = Rect(inpImg.cols - LR_WIDTH, inpImg.rows - LR_HEIGHT, localRegionSize.width, localRegionSize.height);
    localRegions.push_back(Mat(inpImg, rect));
    return localRegions;
}

vector<vector<Mat>> BlockMatchingCorrelation::divideIntoBlocks(Mat inpImg)
{
    vector<vector<Mat>> blockRegions(NUM_BLOCKS_Y, vector<Mat>(NUM_BLOCKS_X));
    Size blockRegionSize(BLOCK_SIZE, BLOCK_SIZE);
    for (int y = 0; y < inpImg.rows - BLOCK_SIZE; y += blockRegionSize.height)
    {
        for (int x = 0; x < inpImg.cols - BLOCK_SIZE; x += blockRegionSize.width)
        {
            Rect rect = Rect(x, y, blockRegionSize.width, blockRegionSize.height);
            blockRegions[y].push_back(Mat(inpImg, rect));
        }
    }
    Rect rect = Rect(inpImg.cols - BLOCK_SIZE, inpImg.rows - BLOCK_SIZE, blockRegionSize.width, blockRegionSize.height);
    blockRegions[NUM_BLOCKS_Y - 1].push_back(Mat(inpImg, rect));
    return blockRegions;
}

void BlockMatchingCorrelation::blockMatching(Mat prev, Mat curr)
{
    // finds the motion vector for a block
    vector<vector<Mat>> prevBlocks;
    vector<Point2f> motionVectorCandidates(7, Point2f(0, 0)); // stores the seven possible MVC
    int rowGR, colGR, rowLR, colLR;
    float SAD, minSAD;

    prevBlocks = divideIntoBlocks(prev);
    for (int i = 0; i < prevBlocks.size(); i++)
    {
        for (int j = 0; j < prevBlocks[i].size(); j++)
        {
            // obtain the motion vectors of the global region that this block lies in
            rowGR = i / ((int)(GR_WIDTH / BLOCK_SIZE));
            colGR = j / ((int)(GR_HEIGHT / BLOCK_SIZE));
            if (rowGR > NUM_GR_Y - 1)
                rowGR = NUM_GR_Y - 1;
            if (colGR > NUM_GR_X - 1)
                colGR = NUM_GR_X - 1;
            motionVectorCandidates[0] = globalRegionMV[rowGR][colGR][0];
            motionVectorCandidates[1] = globalRegionMV[rowGR][colGR][1];

            // obtain the motion vectors of the local region that this block lies in
            rowLR = i / ((int)(LR_WIDTH / BLOCK_SIZE));
            colLR = j / ((int)(LR_HEIGHT / BLOCK_SIZE));
            if (rowLR > NUM_LR_Y - 1)
                rowGR = NUM_LR_Y - 1;
            if (colLR > NUM_LR_X - 1)
                colGR = NUM_LR_X - 1;
            motionVectorCandidates[2] = localRegionMV[rowLR][colLR][0];
            motionVectorCandidates[3] = localRegionMV[rowLR][colLR][1];

            // obtain the motion vector of the immediate LEFT neighbor
            // also, add a small random value (noise) to the motion vector of the immediate LEFT neighbor
            if (i - 1 < 0)
            {
                motionVectorCandidates[4] = Point2f(0, 0);
                motionVectorCandidates[5] = Point2f((float)rand() / RAND_MAX, (float)rand() / RAND_MAX); // random number between 0 and 1
            }
            else
            {
                motionVectorCandidates[4] = currBlockMV[i - 1][j];
                motionVectorCandidates[5] = Point2f(currBlockMV[i - 1][j].x + (float)rand() / RAND_MAX, currBlockMV[i - 1][j].y + (float)rand() / RAND_MAX);
            }

            // find the median of neighboring candidates from the previous MVF, i.e, MVF(n-1)
            motionVectorCandidates[6] = medianNeighbor(i, j, prevBlockMV);

            // find minimum SAD and winning motion vector
            minSAD = (float)INT_MAX;
            for (auto point : motionVectorCandidates)
            {
                SAD = calcSAD(prevBlocks[i][j], i, j, curr, point.x, point.y);
                if (minSAD < SAD)
                {
                    minSAD = SAD;
                    currBlockMV[i][j].x = point.x;
                    currBlockMV[i][j].y = point.y;
                }
            }
        }
    }
    // the motion vectors of all blocks have been found
    prevBlockMV = currBlockMV;
    currBlockMV = zeroes;
}

void BlockMatchingCorrelation::customisedPhaseCorr(Mat prev, Mat curr)
{
    vector<Mat> prevRegions, currRegions;
    vector<Point2f> motionVectorCandidates;
    int num;
    // calculate PPC for each global region
    prevRegions = divideIntoGlobal(prev);
    currRegions = divideIntoGlobal(curr);
    num = 0;
    for (int i = 0; i < NUM_GR_Y; i++) // rows
    {
        for (int j = 0; j < NUM_GR_X; j++) //columns
        {
            motionVectorCandidates = phaseCorr(prevRegions[num], currRegions[num], NULL, NULL);
            globalRegionMV[i].push_back(motionVectorCandidates);
            /* This is the same as :
            globalRegionMV[i][j][0] = motionVectorCandidates[0];
            globalRegionMV[i][j][1] = motionVectorCandidates[1];
            */
            num++;
        }
    }

    // calculate PPC for each local region
    prevRegions = divideIntoLocal(prev);
    currRegions = divideIntoLocal(curr);
    num = 0;
    for (int i = 0; i < NUM_LR_Y; i++) // rows
    {
        for (int j = 0; j < NUM_LR_X; j++) //columns
        {
            motionVectorCandidates = phaseCorr(prevRegions[num], currRegions[num], NULL, NULL);
            localRegionMV[i].push_back(motionVectorCandidates);
            /* This is the same as :
            localRegionMV[i][j][0] = motionVectorCandidates[0];
            localRegionMV[i][j][1] = motionVectorCandidates[1];
            */
            num++;
        }
    }
}

void BlockMatchingCorrelation::BMC(Mat prev, Mat curr)
{
    /* recursive algorithm which determines the motion vector for each block */

    /*---------- Customised Phase Plane Correlation (CPPC) ---------*/
    customisedPhaseCorr(prev, curr);

    /*---------- Block Matching ----------*/
    blockMatching(prev, curr);

    // TODO : perform bidirectional motion compensation
}

void BlockMatchingCorrelation::interpolate()
{
    // TODO : call BMC with initial conditions
}

int main(int argc, char **argv)
{
    //vector<Mat> images = readImg(); // read images from video folder
    // TODO : CPPC function
    // TODO : BM function
    // TODO : call Bidirectional motion estimation function
    // TODO : compute results and accuracy
    return 0;
}

/*
cv::Point2d cv::phaseCorrelate(InputArray _src1, InputArray _src2, InputArray _window, double* response)
{
    Mat src1 = _src1.getMat();
    Mat src2 = _src2.getMat();
    Mat window = _window.getMat();

    CV_Assert( src1.type() == src2.type());
    CV_Assert( src1.type() == CV_32FC1 || src1.type() == CV_64FC1 );
    CV_Assert( src1.size == src2.size);

    if(!window.empty())
    {
        CV_Assert( src1.type() == window.type());
        CV_Assert( src1.size == window.size);
    }

    int M = getOptimalDFTSize(src1.rows);
    int N = getOptimalDFTSize(src1.cols);

    Mat padded1, padded2, paddedWin;

    if(M != src1.rows || N != src1.cols)
    {
        copyMakeBorder(src1, padded1, 0, M - src1.rows, 0, N - src1.cols, BORDER_CONSTANT, Scalar::all(0));
        copyMakeBorder(src2, padded2, 0, M - src2.rows, 0, N - src2.cols, BORDER_CONSTANT, Scalar::all(0));

        if(!window.empty())
        {
            copyMakeBorder(window, paddedWin, 0, M - window.rows, 0, N - window.cols, BORDER_CONSTANT, Scalar::all(0));
        }
    }
    else
    {
        padded1 = src1;
        padded2 = src2;
        paddedWin = window;
    }

    Mat FFT1, FFT2, P, Pm, C;

    // perform window multiplication if available
    if(!paddedWin.empty())
    {
        // apply window to both images before proceeding...
        multiply(paddedWin, padded1, padded1);
        multiply(paddedWin, padded2, padded2);
    }

    // execute phase correlation equation
    // Reference: http://en.wikipedia.org/wiki/Phase_correlation
    dft(padded1, FFT1, DFT_REAL_OUTPUT);
    dft(padded2, FFT2, DFT_REAL_OUTPUT);

    mulSpectrums(FFT1, FFT2, P, 0, true);

    magSpectrums(P, Pm);
    divSpectrums(P, Pm, C, 0, false); // FF* / |FF*| (phase correlation equation completed here...)

    idft(C, C); // gives us the nice peak shift location...

    fftShift(C); // shift the energy to the center of the frame.

    // locate the highest peak
    Point peakLoc;
    minMaxLoc(C, NULL, NULL, NULL, &peakLoc);

    // get the phase shift with sub-pixel accuracy, 5x5 window seems about right here...
    Point2d t;
    t = weightedCentroid(C, peakLoc, Size(5, 5), response);

    // max response is M*N (not exactly, might be slightly larger due to rounding errors)
    if(response)
        *response /= M*N;

    // adjust shift relative to image center...
    Point2d center((double)padded1.cols / 2.0, (double)padded1.rows / 2.0);

    return (center - t);
}

*/
