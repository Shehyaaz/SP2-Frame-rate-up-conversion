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

void bidirectionalMotionCompensation(const vector<vector<UMat>> &prevBlocks, const UMat &curr, const vector<vector<Point2f>> &prevBlocksMV, UMat &newFrame)
{
    // creates the interpolated frame using bidirectional motion compensation
    UMat interpolatedRegion(Size(BLOCK_SIZE, BLOCK_SIZE), CV_8UC3), currRegion(Size(BLOCK_SIZE, BLOCK_SIZE), CV_8UC3);
    int x, y;   // for accessing pixels in the x and y directions respectively
    int dx, dy; // for accessing motion vector components
    int i = 0,j = 0; // for accessing prevBlocks MVs

    //newFrame = UMat::zeros(Size(FRAME_WIDTH * 2, FRAME_HEIGHT * 2), CV_8UC3); // create an empty frame initially
	//UMat frame = UMat::zeros(Size(FRAME_WIDTH, FRAME_HEIGHT), CV_8UC3);
	// adding 10% padding to the newFrame
    //copyMakeBorder( frame, newFrame, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, BORDER_CONSTANT);
    newFrame = UMat::zeros(Size(FRAME_WIDTH, FRAME_HEIGHT), CV_8UC3); // create an empty frame initially
    
    for ( y = 0; y < curr.rows - BLOCK_SIZE; y += BLOCK_SIZE)
    {
    	j = 0;
        for ( x = 0; x < curr.cols; x += BLOCK_SIZE)
        {
            // for each block in prevBlock, determine the block in the next frame
            // currRegion has the block corresponding to prevRegions[i][j]
            dx = (int)round(prevBlocksMV[i][j].x);
            dy = (int)round(prevBlocksMV[i][j].y);
            if ( !(x + dx >= curr.cols || x + dx <= -1*BLOCK_SIZE || y + dy >= curr.rows || y + dy <= -1*BLOCK_SIZE) )
            {
            	currRegion = getPaddedROI(curr, x + dx, y + dy, BLOCK_SIZE, BLOCK_SIZE);
            }
            // determining the interpolateRegion and interpolatedFrame
            addWeighted(prevBlocks[i][j], 0.5, currRegion, 0.5, 0.0, interpolatedRegion); // interpolatedRegion = 0.5*prevBlocks[i][j] + 0.5*currRegion
            interpolatedRegion.copyTo(newFrame(Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)));
            j++;
        }
        i++;
    }
    j = 0;
    i = NUM_BLOCKS_Y - 1;
    y = curr.rows - BLOCK_SIZE;
    for ( x = 0; x < curr.cols; x += BLOCK_SIZE)
    {
    		dx = (int)round(prevBlocksMV[i][j].x);
            dy = (int)round(prevBlocksMV[i][j].y);
			if ( !(x + dx >= curr.cols || x + dx <= -1*BLOCK_SIZE || y + dy >= curr.rows || y + dy <= -1*BLOCK_SIZE))
            {
            	currRegion = getPaddedROI(curr, x + dx, y + dy, BLOCK_SIZE, BLOCK_SIZE);
            }
            // determining the interpolateRegion and interpolatedFrame
            addWeighted(prevBlocks[i][j], 0.5, currRegion, 0.5, 0.0, interpolatedRegion); // interpolatedRegion = 0.5*prevBlocks[i][j] + 0.5*currRegion
            interpolatedRegion.copyTo(newFrame(Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)));
            j++;
    }
    //cout<<"Frame size :"<<newFrame.rows<<","<<newFrame.cols<<endl;
    //newFrame = newFrame(Rect(0, 0, FRAME_WIDTH, FRAME_HEIGHT));
}
