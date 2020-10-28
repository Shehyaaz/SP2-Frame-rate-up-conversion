/*
****************************************
* This file contains the constants used
* in the algorithm.
* Author : Shehyaaz Khan Nayazi
****************************************
*/

#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#define FRAME_WIDTH 1920
#define FRAME_HEIGHT 1080

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
#define NUM_BLOCKS_X FRAME_WIDTH / BLOCK_SIZE // 1920/BLOCK_SIZE -> 60
#define NUM_BLOCKS_Y 34                       // 1080/BLOCK_SIZE -> 33.75 = 34 (approx.)

#define STANDARD_REGION_WIDTH 128
#define STANDARD_REGION_HEIGHT 64

#define INTERPOLATED_VIDEO "video/output.avi"

#endif
