/*
****************************************
* This file contains the constants used
* in the algorithm.
* Author : Shehyaaz Khan Nayazi
****************************************
*/

// for gloabl region
#ifndef GR_WIDTH
#define GR_WIDTH 1024
#endif
#ifndef GR_HEIGHT
#define GR_HEIGHT 512
#endif
// for local region
#ifndef LR_WIDTH
#define LR_WIDTH 256
#endif
#ifndef LR_HEIGHT
#define LR_HEIGHT 128
#endif
// block size
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
// number of global regions
#ifndef NUM_GR
#define NUM_GR 4
#endif
// number of local regions
#ifndef NUM_LR
#define NUM_LR 72
#endif
// number of blocks
#ifndef NUM_BLOCKS
#define NUM_BLOCKS 2025 // (1920*1080)/(BLOCK_SIZE*BLOCK_SIZE)
#endif
