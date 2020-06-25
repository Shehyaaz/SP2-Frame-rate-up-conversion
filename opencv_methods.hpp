/*
****************************************
* These methods are from the opencv
* source code.
****************************************
*/

#include "opencv2/core.hpp";
using namespace cv;

static void magSpectrums(InputArray _src, OutputArray _dst);
static void divSpectrums(InputArray _srcA, InputArray _srcB, OutputArray _dst, int flags, bool conjB);
static void fftShift(InputOutputArray _out);
static Point2d weightedCentroid(InputArray _src, cv::Point peakLocation, cv::Size weightBoxSize, double *response);