/*
****************************************
* This file calls the BMC algorithm.
* Author : Shehyaaz Khan Nayazi
****************************************
*/

#include "bmc.hpp"

int main(int argc, char **argv)
{
    BlockMatchingCorrelation bmcObj = BlockMatchingCorrelation();
    bmcObj.interpolate();
    return 0;
}
/*
To compile from terminal, execute the following commad :
g++ *.cpp -o main `pkg-config --cflags --libs opencv4`
*/
