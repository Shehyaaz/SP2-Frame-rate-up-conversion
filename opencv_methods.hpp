/*
****************************************
* These methods are from the opencv
* source code.
****************************************
*/

#include <opencv2/core.hpp>
using namespace cv;

void magSpectrums(InputArray _src, OutputArray _dst);
void divSpectrums(InputArray _srcA, InputArray _srcB, OutputArray _dst, int flags, bool conjB);
void fftShift(InputOutputArray _out);
Point2d weightedCentroid(InputArray _src, cv::Point peakLocation, cv::Size weightBoxSize, double *response);