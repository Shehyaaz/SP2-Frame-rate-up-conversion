/*
****************************************
* This file contains the implementation
* of the Block Matching Correlation
* (BMC) algorithm.
* Author : Shehyaaz Khan Nayazi
****************************************
*/

#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include "constants.hpp"
#include "opencv_methods.hpp"

using namespace cv;
using namespace std;

class BlockMatchingCorrelation
{
    // variable declarations
    vector<Mat> images;
    vector<vector<Point2d>> globalRegionMV;
    vector<vector<Point2d>> localRegionMV;
    vector<vector<Point2d>> blockMV;

public:
    // function declarations
    BlockMatchingCorrelation()
        : globalRegionMV(NUM_GR),
          localRegionMV(NUM_LR),
          blockMV(NUM_BLOCKS)
    {
        // initialization of variables
    }
    void readImg();
    vector<Point2d> phaseCorr(InputArray _src1, InputArray _src2, InputArray _window, double *response);
    vector<Mat> divideIntoGlobal(Mat inpImg);
    vector<Mat> divideIntoLocal(Mat inpImg);
    vector<Mat> divideIntoBlocks(Mat inpImg);
    void BMC(Mat prev, Mat curr);
    Point2d blockMatching(vector<Point2d> &candidateMV);
    void interpolate();
};

void BlockMatchingCorrelation::readImg()
{
    /* reads the images from the video folder */
    vector<cv::String> fn;
    glob("video/*.jpg", fn, false);

    size_t count = fn.size(); //number of jpg files in video folder
    for (size_t i = 0; i < count; i++)
        images.push_back(imread(fn[i], IMREAD_GRAYSCALE));
}

vector<Point2d> BlockMatchingCorrelation::phaseCorr(InputArray _src1, InputArray _src2, InputArray _window, double *response)
{
    /* performs customised phase plane correlation on the input frames */

    Mat src1 = _src1.getMat();
    Mat src2 = _src2.getMat();
    Mat window = _window.getMat();

    CV_Assert(src1.type() == src2.type());
    CV_Assert(src1.type() == CV_32FC1 || src1.type() == CV_64FC1);
    CV_Assert(src1.size == src2.size);

    if (!window.empty())
    {
        CV_Assert(src1.type() == window.type());
        CV_Assert(src1.size == window.size);
    }

    int M = getOptimalDFTSize(src1.rows);
    int N = getOptimalDFTSize(src1.cols);

    Mat padded1, padded2, paddedWin;

    if (M != src1.rows || N != src1.cols)
    {
        copyMakeBorder(src1, padded1, 0, M - src1.rows, 0, N - src1.cols, BORDER_CONSTANT, Scalar::all(0));
        copyMakeBorder(src2, padded2, 0, M - src2.rows, 0, N - src2.cols, BORDER_CONSTANT, Scalar::all(0));

        if (!window.empty())
        {
            copyMakeBorder(window, paddedWin, 0, M - window.rows, 0, N - window.cols, BORDER_CONSTANT, Scalar::all(0));
        }
    }
    else
    {
        padded1 = src1;
        padded2 = src2;
        paddedWin = window;
    }

    Mat FFT1, FFT2, P, Pm, C;

    // perform window multiplication if available
    if (!paddedWin.empty())
    {
        // apply window to both images before proceeding...
        multiply(paddedWin, padded1, padded1);
        multiply(paddedWin, padded2, padded2);
    }

    // execute phase correlation equation
    // Reference: http://en.wikipedia.org/wiki/Phase_correlation
    dft(padded1, FFT1, DFT_REAL_OUTPUT);
    dft(padded2, FFT2, DFT_REAL_OUTPUT);

    mulSpectrums(FFT1, FFT2, P, 0, true);

    magSpectrums(P, Pm);
    divSpectrums(P, Pm, C, 0, false); // FF* / |FF*| (phase correlation equation completed here...)

    idft(C, C); // gives us the nice peak shift location...

    fftShift(C); // shift the energy to the center of the frame.

    // locate the highest peak
    Point peakLoc;
    // return two peak locations
    minMaxLoc(C, NULL, NULL, NULL, &peakLoc);

    // get the phase shift with sub-pixel accuracy, 5x5 window seems about right here...
    Point2d t1, t2;
    t1 = weightedCentroid(C, peakLoc, Size(5, 5), response);
    C.at<float>(peakLoc) = 0;                 // set the value at peakLoc to 0
    minMaxLoc(C, NULL, NULL, NULL, &peakLoc); // find second peakLoc
    t2 = weightedCentroid(C, peakLoc, Size(5, 5), response);

    // max response is M*N (not exactly, might be slightly larger due to rounding errors)
    if (response)
        *response /= M * N;

    // adjust shift relative to image center...
    Point2d center((double)padded1.cols / 2.0, (double)padded1.rows / 2.0);

    return {(center - t1), (center - t2)};
}

vector<Mat> BlockMatchingCorrelation::divideIntoGlobal(Mat inpImg)
{
    vector<Mat> globalRegions(NUM_GR);
    Size globalRegionSize(GR_WIDTH, GR_HEIGHT);
    for (int y = 0; y < inpImg.rows - GR_HEIGHT; y += globalRegionSize.height)
    {
        for (int x = 0; x < inpImg.cols - GR_WIDTH; x += inpImg.cols - GR_WIDTH)
        {
            Rect rect = Rect(x, y, globalRegionSize.width, globalRegionSize.height);
            globalRegions.push_back(Mat(inpImg, rect));
        }
    }
    Rect rect = Rect(inpImg.cols - GR_WIDTH, inpImg.rows - GR_HEIGHT, globalRegionSize.width, globalRegionSize.height);
    globalRegions.push_back(Mat(inpImg, rect));
    return globalRegions;
}

vector<Mat> BlockMatchingCorrelation::divideIntoLocal(Mat inpImg)
{
    vector<Mat> localRegions(NUM_LR);
    Size localRegionSize(LR_WIDTH, LR_HEIGHT);
    for (int y = 0; y < inpImg.rows - LR_HEIGHT; y += localRegionSize.height)
    {
        for (int x = 0; x < inpImg.cols - LR_WIDTH; x += localRegionSize.width)
        {
            Rect rect = Rect(x, y, localRegionSize.width, localRegionSize.height);
            localRegions.push_back(Mat(inpImg, rect));
        }
    }
    Rect rect = Rect(inpImg.cols - LR_WIDTH, inpImg.rows - LR_HEIGHT, localRegionSize.width, localRegionSize.height);
    localRegions.push_back(Mat(inpImg, rect));
    return localRegions;
}

vector<Mat> BlockMatchingCorrelation::divideIntoBlocks(Mat inpImg)
{
    vector<Mat> blockRegions(NUM_BLOCKS);
    Size blockRegionSize(BLOCK_SIZE, BLOCK_SIZE);
    for (int y = 0; y < inpImg.rows - BLOCK_SIZE; y += blockRegionSize.height)
    {
        for (int x = 0; x < inpImg.cols - BLOCK_SIZE; x += blockRegionSize.width)
        {
            Rect rect = Rect(x, y, blockRegionSize.width, blockRegionSize.height);
            blockRegions.push_back(Mat(inpImg, rect));
        }
    }
    Rect rect = Rect(inpImg.cols - BLOCK_SIZE, inpImg.rows - BLOCK_SIZE, blockRegionSize.width, blockRegionSize.height);
    blockRegions.push_back(Mat(inpImg, rect));
    return blockRegions;
}

void BlockMatchingCorrelation::BMC(Mat prev, Mat curr)
{
}

Point2d BlockMatchingCorrelation::blockMatching(vector<Point2d> &candidateMV)
{
}

void BlockMatchingCorrelation::interpolate()
{
}

int main(int argc, char **argv)
{
    //vector<Mat> images = readImg(); // read images from video folder
    // TODO : CPPC function
    // TODO : BM function
    // TODO : call Bidirectional motion estimation function
    // TODO : compute results and accuracy
    return 0;
}

/*
cv::Point2d cv::phaseCorrelate(InputArray _src1, InputArray _src2, InputArray _window, double* response)
{
    Mat src1 = _src1.getMat();
    Mat src2 = _src2.getMat();
    Mat window = _window.getMat();

    CV_Assert( src1.type() == src2.type());
    CV_Assert( src1.type() == CV_32FC1 || src1.type() == CV_64FC1 );
    CV_Assert( src1.size == src2.size);

    if(!window.empty())
    {
        CV_Assert( src1.type() == window.type());
        CV_Assert( src1.size == window.size);
    }

    int M = getOptimalDFTSize(src1.rows);
    int N = getOptimalDFTSize(src1.cols);

    Mat padded1, padded2, paddedWin;

    if(M != src1.rows || N != src1.cols)
    {
        copyMakeBorder(src1, padded1, 0, M - src1.rows, 0, N - src1.cols, BORDER_CONSTANT, Scalar::all(0));
        copyMakeBorder(src2, padded2, 0, M - src2.rows, 0, N - src2.cols, BORDER_CONSTANT, Scalar::all(0));

        if(!window.empty())
        {
            copyMakeBorder(window, paddedWin, 0, M - window.rows, 0, N - window.cols, BORDER_CONSTANT, Scalar::all(0));
        }
    }
    else
    {
        padded1 = src1;
        padded2 = src2;
        paddedWin = window;
    }

    Mat FFT1, FFT2, P, Pm, C;

    // perform window multiplication if available
    if(!paddedWin.empty())
    {
        // apply window to both images before proceeding...
        multiply(paddedWin, padded1, padded1);
        multiply(paddedWin, padded2, padded2);
    }

    // execute phase correlation equation
    // Reference: http://en.wikipedia.org/wiki/Phase_correlation
    dft(padded1, FFT1, DFT_REAL_OUTPUT);
    dft(padded2, FFT2, DFT_REAL_OUTPUT);

    mulSpectrums(FFT1, FFT2, P, 0, true);

    magSpectrums(P, Pm);
    divSpectrums(P, Pm, C, 0, false); // FF* / |FF*| (phase correlation equation completed here...)

    idft(C, C); // gives us the nice peak shift location...

    fftShift(C); // shift the energy to the center of the frame.

    // locate the highest peak
    Point peakLoc;
    minMaxLoc(C, NULL, NULL, NULL, &peakLoc);

    // get the phase shift with sub-pixel accuracy, 5x5 window seems about right here...
    Point2d t;
    t = weightedCentroid(C, peakLoc, Size(5, 5), response);

    // max response is M*N (not exactly, might be slightly larger due to rounding errors)
    if(response)
        *response /= M*N;

    // adjust shift relative to image center...
    Point2d center((double)padded1.cols / 2.0, (double)padded1.rows / 2.0);

    return (center - t);
}

*/
