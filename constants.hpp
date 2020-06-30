/*
****************************************
* This file contains the constants used
* in the algorithm.
* Author : Shehyaaz Khan Nayazi
****************************************
*/

#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#define IMAGE_WIDTH 1080
#define IMAGE_HEIGHT 1920

// for gloabl region
#define GR_WIDTH 512
#define GR_HEIGHT 1024

// for local region
#define LR_WIDTH 128
#define LR_HEIGHT 256

// block size
#define BLOCK_SIZE 32

// number of global regions
#define NUM_GR_X 2
#define NUM_GR_Y 2

// number of local regions
#define NUM_LR_X 9
#define NUM_LR_Y 8

// number of blocks
#define NUM_BLOCKS_X 34                        // 1080/BLOCK_SIZE -> 33.75 = 34 (approx.)
#define NUM_BLOCKS_Y IMAGE_HEIGHT / BLOCK_SIZE // 1920/BLOCK_SIZE -> 60

#endif
