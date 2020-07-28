/*
****************************************
* This file contains the code to 
* contruct the interpolated frame.
* Author : Shehyaaz Khan Nayazi
****************************************
*/

#include "motion_compensation.hpp"
#include "constants.hpp"

using namespace cv;
using namespace std;

Mat bidirectionalMotionCompensation(vector<vector<Mat>> &prevBlocks, Mat curr, vector<vector<Point2f>> &prevBlocksMV)
{
    // creates the interpolated frame using bidirectional motion compensation
    Mat interpolatedFrame, currRegion(Size(BLOCK_SIZE, BLOCK_SIZE), CV_8U);
    int x, y;   // for accessing pixels in the x and y directions respectively
    int dx, dy; // for accessing motion vector components

    interpolatedFrame = Mat::zeros(Size(curr.cols * 2, curr.rows * 2), CV_8U); // create a frame with double the size initially

    for (int i = 0; i < prevBlocks.size(); i++)
    {
        for (int j = 0; j < prevBlocks[i].size(); j++)
        {
            // for each block in prevBlock, determine the block in the next frame
            x = BLOCK_SIZE * j; // column
            y = BLOCK_SIZE * i; // row
            dx = (int)round(prevBlocksMV[i][j].x);
            dy = (int)round(prevBlocksMV[i][j].y);

            for (int k = 0; k < BLOCK_SIZE; k++)
            {
                for (int l = 0; l < BLOCK_SIZE; l++)
                {
                    if ((k + y + dy) < 0 || (k + y + dy) >= curr.rows || (l + x + dx) < 0 || (l + x + dx) >= curr.cols)
                        currRegion.at<uchar>(k, l) = 0;
                    else
                        currRegion.at<uchar>(k, l) = curr.at<uchar>((k + y + dy), (l + x + dx));
                }
            }
            // currRegion has the block corresponding to prevRegions[i][j]

            // determining the block in interpolatedFrame
            for (int k = 0; k < BLOCK_SIZE; k++)
            {
                for (int l = 0; l < BLOCK_SIZE; l++)
                {
                    // averaging the vlaues of the corresponding blocks in the previous and current frame
                    if ((k + y + dy) < 0 || (k + y + dy) >= interpolatedFrame.rows || (l + x + dx) < 0 || (l + x + dx) >= interpolatedFrame.cols)
                        continue;
                    else
                        interpolatedFrame.at<uchar>((y + k + dy), (x + l + dx)) = (int)(0.5 * (prevBlocks[i][j].at<uchar>(k, l) + currRegion.at<uchar>(k, l)));
                }
            }
        }
    }

    // reduce interpolatedFrame to the original size, and return
    resize(interpolatedFrame, interpolatedFrame, Size(interpolatedFrame.cols * 0.5, interpolatedFrame.rows * 0.5), 0, 0, INTER_AREA);
    return interpolatedFrame;
}