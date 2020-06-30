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

#include "bmc.hpp"
#include "constants.hpp"
#include "util.hpp"
#include "motion_compensation.hpp"

using namespace cv;
using namespace std;

vector<Mat> BlockMatchingCorrelation::divideIntoGlobal(Mat inpImg)
{
    vector<Mat> globalRegions(NUM_GR_Y * NUM_GR_X);
    Size globalRegionSize(GR_WIDTH, GR_HEIGHT);
    Rect rect;
    int k = 0; // index for iterating through globalRegions
    for (int y = 0; y <= inpImg.rows - GR_HEIGHT; y += inpImg.rows - GR_HEIGHT)
    {
        for (int x = 0; x <= inpImg.cols - GR_WIDTH; x += inpImg.cols - GR_WIDTH)
        {
            rect = Rect(x, y, globalRegionSize.width, globalRegionSize.height);
            globalRegions[k++] = Mat(inpImg, rect);
        }
    }

    // cout << globalRegions.size() << endl;
    return globalRegions;
}

vector<Mat> BlockMatchingCorrelation::divideIntoLocal(Mat inpImg)
{
    vector<Mat> localRegions(NUM_LR_Y * NUM_LR_X);
    Size localRegionSize(LR_WIDTH, LR_HEIGHT);
    Rect rect;
    int k = 0; // index for iterating through localRegions
    for (int y = 0; y < inpImg.rows - LR_HEIGHT; y += localRegionSize.height)
    {
        for (int x = 0; x < inpImg.cols - LR_WIDTH; x += localRegionSize.width)
        {
            rect = Rect(x, y, localRegionSize.width, localRegionSize.height);
            localRegions[k++] = Mat(inpImg, rect);
        }
        rect = Rect(inpImg.cols - LR_WIDTH, y, localRegionSize.width, localRegionSize.height);
        localRegions[k++] = Mat(inpImg, rect);
    }

    for (int x = 0; x < inpImg.cols - LR_WIDTH; x += localRegionSize.width)
    {
        rect = Rect(x, inpImg.rows - LR_HEIGHT, localRegionSize.width, localRegionSize.height);
        localRegions[k++] = Mat(inpImg, rect);
    }
    rect = Rect(inpImg.cols - LR_WIDTH, inpImg.rows - LR_HEIGHT, localRegionSize.width, localRegionSize.height);
    localRegions[k++] = Mat(inpImg, rect);

    // cout << localRegions.size() << endl;
    return localRegions;
}

vector<vector<Mat>> BlockMatchingCorrelation::divideIntoBlocks(Mat inpImg)
{
    vector<vector<Mat>> blockRegions(NUM_BLOCKS_Y, vector<Mat>(NUM_BLOCKS_X));
    Size blockRegionSize(BLOCK_SIZE, BLOCK_SIZE);
    Rect rect;
    int i = 0, j = 0; // for indexing blockRegions
    for (int y = 0; y < inpImg.rows; y += blockRegionSize.height)
    {
        j = 0;
        for (int x = 0; x < inpImg.cols - BLOCK_SIZE; x += blockRegionSize.width)
        {
            rect = Rect(x, y, blockRegionSize.width, blockRegionSize.height);
            blockRegions[i][j] = Mat(inpImg, rect);
            j++;
        }
        rect = Rect(inpImg.cols - BLOCK_SIZE, y, blockRegionSize.width, blockRegionSize.height);
        blockRegions[i][j] = Mat(inpImg, rect);
        i++;
    }

    // cout << blockRegions.size() << endl;
    return blockRegions;
}

void BlockMatchingCorrelation::blockMatching(Mat prev, Mat curr)
{
    // finds the motion vector for a block
    vector<vector<Mat>> prevBlocks;
    Mat prev32f, curr32f;
    vector<Point2f> motionVectorCandidates(7, Point2f(0, 0)); // stores the seven possible MVC
    int rowGR, colGR, rowLR, colLR;
    float SAD, minSAD;

    prevBlocks = divideIntoBlocks(prev);

    for (int i = 0; i < NUM_BLOCKS_Y; i++)
    {
        for (int j = 0; j < NUM_BLOCKS_X; j++)
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
                rowLR = NUM_LR_Y - 1;
            if (colLR > NUM_LR_X - 1)
                colLR = NUM_LR_X - 1;
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
                prevBlocks[i][j].convertTo(prev32f, CV_32FC1);
                curr.convertTo(curr32f, CV_32FC1);

                SAD = calcSAD(prev32f, i, j, curr32f, point.x, point.y);
                if (SAD < minSAD)
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
    Mat prev32f, curr32f;
    int num;
    // calculate PPC for each global region
    prevRegions = divideIntoGlobal(prev);
    currRegions = divideIntoGlobal(curr);
    num = 0;

    for (int i = 0; i < NUM_GR_Y; i++) // rows
    {
        for (int j = 0; j < NUM_GR_X; j++) //columns
        {
            prevRegions[num].convertTo(prev32f, CV_32FC1);
            currRegions[num].convertTo(curr32f, CV_32FC1);
            motionVectorCandidates = phaseCorr(prev32f, curr32f, noArray(), 0);
            globalRegionMV[i][j] = motionVectorCandidates;
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
            prevRegions[num].convertTo(prev32f, CV_32FC1);
            currRegions[num].convertTo(curr32f, CV_32FC1);
            motionVectorCandidates = phaseCorr(prev32f, curr32f, noArray(), 0);
            localRegionMV[i][j] = motionVectorCandidates;
            /* This is the same as :
            localRegionMV[i][j][0] = motionVectorCandidates[0];
            localRegionMV[i][j][1] = motionVectorCandidates[1];
            */
            num++;
        }
    }
}

Mat BlockMatchingCorrelation::BMC(Mat prev, Mat curr)
{
    /* this algorithm determines the motion vector for each block */

    /*---------- Customised Phase Plane Correlation (CPPC) ---------*/
    customisedPhaseCorr(prev, curr);
    cout << "before BM\t";
    /*---------- Block Matching ----------*/
    blockMatching(prev, curr);

    cout << "before interpolation\t";

    // perform bidirectional motion compensation and return the interpolated frame
    vector<vector<Mat>> prevBlocks = divideIntoBlocks(prev);
    Mat interpolatedFrame = bidirectionalMotionCompensation(prevBlocks, curr, prevBlockMV);

    cout << "after interpolation\n";
    return interpolatedFrame; // return interpolated frame
}

void BlockMatchingCorrelation::interpolate()
{
    images = readImg(); // vector of all images in grayscale
    ofstream execFile(EXEC_TIME_FILE, ios_base::app);
    Mat colorImg;

    for (int i = 0; i < images.size() - 1; i++)
    {
        auto start = chrono::high_resolution_clock::now();

        Mat frame = BMC(images[i], images[i + 1]);
        //cout << "interpolation done";
        interpolated.push_back(frame);

        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        writeToFile(execFile, duration);
    }
    execFile.close();

    // save all interpolated frames
    for (int i = 0; i < images.size(); i++)
    {
        cvtColor(images[i], colorImg, COLOR_GRAY2RGB);
        imwrite(INTERPOLATED_FRAME_FOLDER + to_string((i + 1)) + ".jpg", colorImg);
    }
}
