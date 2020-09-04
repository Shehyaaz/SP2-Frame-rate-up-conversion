/*
****************************************
* This file contains the code to 
* contruct the interpolated frame.
* Author : Shehyaaz Khan Nayazi
****************************************
*/

#include "motion_compensation.hpp"
#include "constants.hpp"
#include "util.hpp"

using namespace cv;
using namespace std;

UMat bidirectionalMotionCompensation(vector<vector<UMat>> &prevBlocks, UMat curr, vector<vector<Point2f>> &prevBlocksMV)
{
    // creates the interpolated frame using bidirectional motion compensation
    UMat interpolatedFrame, interpolatedRegion(Size(BLOCK_SIZE, BLOCK_SIZE), CV_8UC3), currRegion(Size(BLOCK_SIZE, BLOCK_SIZE), CV_8UC3);
    int x, y;   // for accessing pixels in the x and y directions respectively
    int dx, dy; // for accessing motion vector components

    interpolatedFrame = UMat::zeros(Size(FRAME_WIDTH * 2, FRAME_HEIGHT * 2), CV_8UC3); // create a frame with double the size initially

    for (int i = 0; i < prevBlocks.size(); i++)
    {
        for (int j = 0; j < prevBlocks[i].size(); j++)
        {
            // for each block in prevBlock, determine the block in the next frame
            // currRegion has the block corresponding to prevRegions[i][j]
            x = BLOCK_SIZE * j; // column
            y = BLOCK_SIZE * i; // row
            dx = (int)round(prevBlocksMV[i][j].x);
            dy = (int)round(prevBlocksMV[i][j].y);

            currRegion = getPaddedROI(curr, x + dx, y + dy, BLOCK_SIZE, BLOCK_SIZE, Scalar(0.0));

            // determining the interpolateRegion and interpolatedFrame
            addWeighted(prevBlocks[i][j], 0.5, currRegion, 0.5, 0.0, interpolatedRegion); // interpolatedRegion = 0.5*prevBlocks[i][j] + 0.5*currRegion
            interpolatedRegion.copyTo(interpolatedFrame(Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)));
        }
    }

    // reduce interpolatedFrame to the original size, and return
    //resize(interpolatedFrame, interpolatedFrame, Size(FRAME_WIDTH, FRAME_HEIGHT), 0, 0, INTER_AREA);
    return interpolatedFrame;
}
