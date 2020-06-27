/*
****************************************
* This file contains the constants used
* in the algorithm.
* Author : Shehyaaz Khan Nayazi
****************************************
*/

#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

// for gloabl region
#define GR_WIDTH 1024
#define GR_HEIGHT 512

// for local region
#define LR_WIDTH 256
#define LR_HEIGHT 128

// block size
#define BLOCK_SIZE 32

// number of global regions
#define NUM_GR_X 2
#define NUM_GR_Y 2

// number of local regions
#define NUM_LR_X 8
#define NUM_LR_Y 9

// number of blocks
#define NUM_BLOCKS_X 60 // 1920/BLOCK_SIZE
#define NUM_BLOCKS_Y 34 // 1080/BLOCK_SIZE

static const vector<vector<Point2f>> zeroes = vector<vector<Point2f>>(NUM_BLOCKS_Y, vector<Point2f>(NUM_BLOCKS_X, Point2f(0, 0)));

#endif
