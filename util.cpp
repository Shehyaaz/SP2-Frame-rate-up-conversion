/*
****************************************
* This file contains the definitions of
* the utility functions used in the 
* algorithm.
* Author : Shehyaaz Khan Nayazi
****************************************
*/

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include "constants.hpp"
#include "opencv_methods.hpp"
#include "util.hpp"

using namespace cv;
using namespace std;

void readFrames(String videoFile, vector<UMat> &frames)
{
    /* reads the images from the video folder */
    VideoCapture cap(videoFile);
    // check if video opened successfully
    if (!cap.isOpened())
    {
        cout << "Error opening video stream or file" << endl;
        exit(-1);
    }

    while (1)
    {
        UMat fr;
        cap >> fr;
        // If the frame is empty, break immediately
        if (fr.empty())
            break;

        frames.push_back(fr);
        // Press  ESC on keyboard to  exit
        char c = (char)waitKey(1);
        if (c == 27)
            break;
    }
    // release the video
    cap.release();
}

vector<Point2f> phaseCorr(InputArray _src1, InputArray _src2, InputArray _window, double *response = 0)
{
    /* performs customised phase plane correlation on the input frames */
    UMat src1 = _src1.getMat().getUMat(ACCESS_WRITE);
    UMat src2 = _src2.getMat().getUMat(ACCESS_WRITE);
    UMat window = _window.getMat().getUMat(ACCESS_WRITE);

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

    UMat padded1, padded2, paddedWin;

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

    UMat FFT1, FFT2, P, Pm, C;

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
    Point2f t1, t2;
    t1 = weightedCentroid(C, peakLoc, Size(5, 5), response);
    C.getMat(ACCESS_WRITE).at<float>(peakLoc) = 0; // set the value at peakLoc to 0
    minMaxLoc(C, NULL, NULL, NULL, &peakLoc);      // find second peakLoc
    t2 = weightedCentroid(C, peakLoc, Size(5, 5), response);

    // max response is M*N (not exactly, might be slightly larger due to rounding errors)
    if (response)
        *response /= M * N;

    // adjust shift relative to image center...
    Point2f center((double)padded1.cols / 2.0, (double)padded1.rows / 2.0);

    return {(center - t1), (center - t2)};
}

float calcSAD(const UMat &prevBlock, int rowpos, int colpos, const UMat &curr, float dx, float dy)
{
    CV_Assert(prevBlock.type() == curr.type());
    CV_Assert(prevBlock.type() == CV_32FC1 || prevBlock.type() == CV_64FC1);

    UMat currBlock, absDiff;
    float SAD = 0.0; // to store SAD value
    int dx_int = (int)round(dx);
    int dy_int = (int)round(dy);
    int x = colpos * BLOCK_SIZE;
    int y = rowpos * BLOCK_SIZE;

    currBlock = getPaddedROI(curr, x + dx_int, y + dy_int, BLOCK_SIZE, BLOCK_SIZE);
    absdiff(prevBlock, currBlock, absDiff); // absDiff = prevBlock - currBlock
    SAD = sum(absDiff)[0];

    return SAD;
}

Point2f medianNeighbor(int rowpos, int colpos, vector<vector<Point2f>> &prevBlockMV)
{
    // median of Point(x,y) = {median of x-coordinates, median of y-coordinates}
    vector<Point2f> neighborMV(3, Point2f(0, 0));
    vector<float> x_coord(3, 0.0);
    vector<float> y_coord(3, 0.0);

    /* we are considering the three nearest neighbors here */
    if (rowpos - 1 < 0 && colpos - 1 < 0)
    {
        // block is in the top-left corner
        neighborMV[0] = prevBlockMV[rowpos][colpos + 1];
        neighborMV[1] = prevBlockMV[rowpos + 1][colpos];
        neighborMV[2] = prevBlockMV[rowpos + 1][colpos + 1];
    }
    else if (colpos - 1 < 0)
    {
        // block is along the left edge
        neighborMV[0] = prevBlockMV[rowpos - 1][colpos];
        neighborMV[1] = prevBlockMV[rowpos - 1][colpos + 1];
        neighborMV[2] = prevBlockMV[rowpos][colpos + 1];
    }
    else if (rowpos - 1 < 0)
    {
        // block is in the top-right corner
        neighborMV[0] = prevBlockMV[rowpos][colpos - 1];
        neighborMV[1] = prevBlockMV[rowpos + 1][colpos - 1];
        neighborMV[2] = prevBlockMV[rowpos + 1][colpos];
    }
    else
    {
        // block is in the middle region
        neighborMV[0] = prevBlockMV[rowpos - 1][colpos - 1];
        neighborMV[1] = prevBlockMV[rowpos - 1][colpos];
        neighborMV[2] = prevBlockMV[rowpos][colpos - 1];
    }
    // find the median
    for (auto point : neighborMV)
    {
        x_coord.push_back(point.x);
        y_coord.push_back(point.y);
    }
    // median of x-coordinates
    const auto median_it_x = x_coord.begin() + x_coord.size() / 2;
    std::nth_element(x_coord.begin(), median_it_x, x_coord.end());
    // median of y-coordinates
    const auto median_it_y = y_coord.begin() + y_coord.size() / 2;
    std::nth_element(y_coord.begin(), median_it_y, y_coord.end());

    Point2f median = Point2f(*median_it_x, *median_it_y);
    return median;
}

void writeToFile(ofstream &file, chrono::milliseconds duration)
{
    if (!file)
    {
        cout << "Could not open the file\n";
        exit(-1);
    }
    file << "Interpolated frame in :" << duration.count() << " milliseconds \n";
}

UMat getPaddedROI(const UMat &input, int top_left_x, int top_left_y, int width, int height, Scalar paddingColor)
{
    int bottom_right_x = top_left_x + width;
    int bottom_right_y = top_left_y + height;

    UMat output;
    if (top_left_x < 0 || top_left_y < 0 || bottom_right_x > input.cols || bottom_right_y > input.rows)
    {
        // border padding will be required
        int border_left = 0, border_right = 0, border_top = 0, border_bottom = 0;

        if (top_left_x < 0)
        {
            width = width + top_left_x;
            border_left = -1 * top_left_x;
            top_left_x = 0;
        }
        if (top_left_y < 0)
        {
            height = height + top_left_y;
            border_top = -1 * top_left_y;
            top_left_y = 0;
        }
        if (bottom_right_x > input.cols)
        {
            width = width - (bottom_right_x - input.cols);
            border_right = bottom_right_x - input.cols;
        }
        if (bottom_right_y > input.rows)
        {
            height = height - (bottom_right_y - input.rows);
            border_bottom = bottom_right_y - input.rows;
        }

        Rect R(top_left_x, top_left_y, width, height);
        copyMakeBorder(input(R), output, border_top, border_bottom, border_left, border_right, BORDER_CONSTANT, paddingColor);
    }
    else
    {
        // no border padding required
        Rect R(top_left_x, top_left_y, width, height);
        output = input(R);
    }
    return output;
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
