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

vector<UMat> BlockMatchingCorrelation::divideIntoGlobal(UMat inpFrame)
{
    vector<UMat> globalRegions(NUM_GR_Y * NUM_GR_X);
    Rect rect;
    int k = 0; // index for iterating through globalRegions
    for (int y = 0; y <= inpFrame.rows - GR_HEIGHT; y += inpFrame.rows - GR_HEIGHT)
    {
        for (int x = 0; x <= inpFrame.cols - GR_WIDTH; x += inpFrame.cols - GR_WIDTH)
        {
            rect = Rect(x, y, GR_WIDTH, GR_HEIGHT);
            globalRegions[k++] = UMat(inpFrame, rect);
        }
    }

    // cout << globalRegions.size() << endl;
    return globalRegions;
}

vector<UMat> BlockMatchingCorrelation::divideIntoLocal(UMat inpFrame)
{
    vector<UMat> localRegions(NUM_LR_Y * NUM_LR_X);
    Rect rect;
    int k = 0; // index for iterating through localRegions
    for (int y = 0; y < inpFrame.rows - LR_HEIGHT; y += LR_HEIGHT)
    {
        for (int x = 0; x < inpFrame.cols - LR_WIDTH; x += LR_WIDTH)
        {
            rect = Rect(x, y, LR_WIDTH, LR_HEIGHT);
            localRegions[k++] = UMat(inpFrame, rect);
        }
        rect = Rect(inpFrame.cols - LR_WIDTH, y, LR_WIDTH, LR_HEIGHT);
        localRegions[k++] = UMat(inpFrame, rect);
    }

    for (int x = 0; x < inpFrame.cols - LR_WIDTH; x += LR_WIDTH)
    {
        rect = Rect(x, inpFrame.rows - LR_HEIGHT, LR_WIDTH, LR_HEIGHT);
        localRegions[k++] = UMat(inpFrame, rect);
    }
    rect = Rect(inpFrame.cols - LR_WIDTH, inpFrame.rows - LR_HEIGHT, LR_WIDTH, LR_HEIGHT);
    localRegions[k++] = UMat(inpFrame, rect);

    // cout << localRegions.size() << endl;
    return localRegions;
}

vector<vector<UMat>> BlockMatchingCorrelation::divideIntoBlocks(UMat inpFrame)
{
    vector<vector<UMat>> blockRegions(NUM_BLOCKS_Y, vector<UMat>(NUM_BLOCKS_X));
    Rect rect;
    int i = 0, j = 0; // for indexing blockRegions
    for (int y = 0; y < inpFrame.rows - BLOCK_SIZE; y += BLOCK_SIZE)
    {
        j = 0;
        for (int x = 0; x < inpFrame.cols; x += BLOCK_SIZE)
        {
            rect = Rect(x, y, BLOCK_SIZE, BLOCK_SIZE);
            blockRegions[i][j] = UMat(inpFrame, rect);
            j++;
        }
        i++;
    }
    j = 0;
    i = NUM_BLOCKS_Y - 1;
    for (int x = 0; x < inpFrame.cols; x += BLOCK_SIZE)
    {
        rect = Rect(x, inpFrame.rows - BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
        blockRegions[i][j] = UMat(inpFrame, rect);
        j++;
    }

    // cout << blockRegions.size() << endl;
    return blockRegions;
}

void BlockMatchingCorrelation::blockMatching(UMat prev, UMat curr)
{
    // finds the motion vector for a block
    vector<vector<UMat>> prevBlocks;
    UMat prev32f, curr32f;
    vector<Point2f> motionVectorCandidates(7, Point2f(0, 0)); // stores the seven possible MVC
    int rowGR, colGR, rowLR, colLR;
    float SAD, minSAD;

    prevBlocks = divideIntoBlocks(prev);
    curr.convertTo(curr32f, CV_32FC1);

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
            prevBlocks[i][j].convertTo(prev32f, CV_32FC1);
            minSAD = (float)INT_MAX;
            for (auto point : motionVectorCandidates)
            {
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

void BlockMatchingCorrelation::customisedPhaseCorr(UMat prev, UMat curr)
{
    vector<UMat> prevRegions, currRegions;
    vector<Point2f> motionVectorCandidates;
    UMat prev32f, curr32f;
    int num;
    UMat diff;
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
            resize(prev32f, prev32f, stdSize);
            resize(curr32f, curr32f, stdSize);
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
            resize(prev32f, prev32f, stdSize);
            resize(curr32f, curr32f, stdSize);

            absdiff(prev32f, curr32f, diff);
            if (countNonZero(diff) == 0) // both regions are equal
            {
                localRegionMV[i][j][0] = Point2f(0, 0);
                localRegionMV[i][j][1] = Point2f(0, 0);
            }
            else
            {
                motionVectorCandidates = phaseCorr(prev32f, curr32f, noArray(), 0);
                localRegionMV[i][j] = motionVectorCandidates;
                /* This is the same as :
            localRegionMV[i][j][0] = motionVectorCandidates[0];
            localRegionMV[i][j][1] = motionVectorCandidates[1];
            */
            }
            num++;
        }
    }
}

UMat BlockMatchingCorrelation::BMC(UMat prev, UMat curr)
{
    /* this algorithm determines the motion vector for each block */
    UMat f1, f2, lumI1, lumI2;
    vector<UMat> lum1, lum2;

    cvtColor(prev, f1, COLOR_BGR2YCrCb);
    cvtColor(curr, f2, COLOR_BGR2YCrCb);
    split(f1, lum1);
    split(f2, lum2);
    lumI1 = lum1[0];
    lumI2 = lum2[0];

    /*---------- Customised Phase Plane Correlation (CPPC) ---------*/
    //auto start = chrono::high_resolution_clock::now();
    cout << "Beginning CPPC\t";
    customisedPhaseCorr(lumI1, lumI2);

    //auto stop = chrono::high_resolution_clock::now();
    //auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    //cout << "Time taken to perform CPPC for one frame :" << duration.count() << " ms\n";

    cout << "CPPC complete\t";
    /*---------- Block Matching ----------*/
    //start = chrono::high_resolution_clock::now();
    cout << "Beginning BM\t";
    blockMatching(lumI1, lumI2);

    //stop = chrono::high_resolution_clock::now();
    //duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    //cout << "Time taken to perform BM for one frame :" << duration.count() << " ms\n";

    cout << "BM complete\t";
    /*---------- Frame interpolation ----------*/
    // perform bidirectional motion compensation and return the interpolated frame
    vector<vector<UMat>> prevBlocks = divideIntoBlocks(prev);

    //start = chrono::high_resolution_clock::now();
    cout << "Frame interpolation\t";
    UMat interpolatedFrame = bidirectionalMotionCompensation(prevBlocks, curr, prevBlockMV);

    //stop = chrono::high_resolution_clock::now();
    //duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    //cout << "Time taken to interpolate one frame :" << duration.count() << " ms\n";

    cout << "Interpolation complete\n";
    return interpolatedFrame; // return interpolated frame
}

void BlockMatchingCorrelation::interpolate()
{
    //auto start = chrono::high_resolution_clock::now();

    frames = readFrames(INPUT_VIDEO); // vector of all frames in the video

    //auto stop = chrono::high_resolution_clock::now();
    //auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    //cout << "Time taken to read all video frames :" << duration.count() << " ms\n";

    ofstream execFile(EXEC_TIME_FILE, ios_base::app);

    for (int i = 0; i < frames.size() - 2; i += 2) //for (int i = 0; i < 1; i += 2)
    {
        auto start = chrono::high_resolution_clock::now();

        UMat frame = BMC(frames[i], frames[i + 2]); // considering alternate frames for now
        interpolated.push_back(frame);

        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        writeToFile(execFile, duration);
    }
    execFile.close();

    cout << "Frame interpolation complete, creating new video..." << endl;

    //start = chrono::high_resolution_clock::now();

    // create interpolated video
    VideoWriter interpolatedVideo(INTERPOLATED_VIDEO, VideoWriter::fourcc('X', 'V', 'I', 'D'), UPCONVERTED_FRAME_RATE, Size(FRAME_WIDTH, FRAME_HEIGHT));
    int j = 0;
    for (int i = 1; i < frames.size() - 1; i += 2)
    {
        frames[i] = interpolated[j];
        j++;
    }

    for (int i = 0; i < frames.size(); i++) //for (int i = 0; i < 1; i++)
    {
        interpolatedVideo.write(frames[i]);
        waitKey(1);
    }
    interpolatedVideo.release();

    //stop = chrono::high_resolution_clock::now();
    //duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    //cout << "Time taken to write one frame :" << duration.count() << " ms\n";

    cout << "...completed the new video" << endl;
}
