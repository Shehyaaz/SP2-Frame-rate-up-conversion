/*
****************************************
* This file calls the BMC algorithm.
* Author : Shehyaaz Khan Nayazi
****************************************
*/

#include "bmc.hpp"
#include "image_quality.hpp"

int main(int argc, char **argv)
{
    BlockMatchingCorrelation bmcObj = BlockMatchingCorrelation();
    bmcObj.interpolate();
    // calcuate the results and store
    calcQuality();
    return 0;
}
/*
To compile from terminal, execute the following commad :
g++ *.cpp -o main `pkg-config --cflags --libs opencv4`
*/