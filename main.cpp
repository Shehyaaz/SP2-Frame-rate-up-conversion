/*
****************************************
* This file calls the BMC algorithm.
* Author : Shehyaaz Khan Nayazi
****************************************
*/

#include "bmc.hpp"

int main(int argc, char **argv)
{
    if (argc == 1)
    {
        cout << "Please enter the path of the input video\nEnter 'help' for more info...\n";
    }
    if (argc >= 2)
    {
        BlockMatchingCorrelation bmcObj = BlockMatchingCorrelation(argv[1]);
        bmcObj.interpolate();
    }
    return 0;
}
/*
To compile from terminal, execute the following commad :
g++ *.cpp -o main `pkg-config --cflags --libs opencv4`

Usage :
./main path-of-input-video
*/
