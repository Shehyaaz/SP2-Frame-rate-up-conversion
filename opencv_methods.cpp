/*
****************************************
* These methods are from the opencv
* source code.
****************************************
*/

#include "opencv_methods.hpp"

static void magSpectrums(InputArray _src, OutputArray _dst)
{
    Mat src = _src.getMat();
    int depth = src.depth(), cn = src.channels(), type = src.type();
    int rows = src.rows, cols = src.cols;
    int j, k;

    CV_Assert(type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2);

    if (src.depth() == CV_32F)
        _dst.create(src.rows, src.cols, CV_32FC1);
    else
        _dst.create(src.rows, src.cols, CV_64FC1);

    Mat dst = _dst.getMat();
    dst.setTo(0); //Mat elements are not equal to zero by default!

    bool is_1d = (rows == 1 || (cols == 1 && src.isContinuous() && dst.isContinuous()));

    if (is_1d)
        cols = cols + rows - 1, rows = 1;

    int ncols = cols * cn;
    int j0 = cn == 1;
    int j1 = ncols - (cols % 2 == 0 && cn == 1);

    if (depth == CV_32F)
    {
        const float *dataSrc = src.ptr<float>();
        float *dataDst = dst.ptr<float>();

        size_t stepSrc = src.step / sizeof(dataSrc[0]);
        size_t stepDst = dst.step / sizeof(dataDst[0]);

        if (!is_1d && cn == 1)
        {
            for (k = 0; k < (cols % 2 ? 1 : 2); k++)
            {
                if (k == 1)
                    dataSrc += cols - 1, dataDst += cols - 1;
                dataDst[0] = dataSrc[0] * dataSrc[0];
                if (rows % 2 == 0)
                    dataDst[(rows - 1) * stepDst] = dataSrc[(rows - 1) * stepSrc] * dataSrc[(rows - 1) * stepSrc];

                for (j = 1; j <= rows - 2; j += 2)
                {
                    dataDst[j * stepDst] = (float)std::sqrt((double)dataSrc[j * stepSrc] * dataSrc[j * stepSrc] +
                                                            (double)dataSrc[(j + 1) * stepSrc] * dataSrc[(j + 1) * stepSrc]);
                }

                if (k == 1)
                    dataSrc -= cols - 1, dataDst -= cols - 1;
            }
        }

        for (; rows--; dataSrc += stepSrc, dataDst += stepDst)
        {
            if (is_1d && cn == 1)
            {
                dataDst[0] = dataSrc[0] * dataSrc[0];
                if (cols % 2 == 0)
                    dataDst[j1] = dataSrc[j1] * dataSrc[j1];
            }

            for (j = j0; j < j1; j += 2)
            {
                dataDst[j] = (float)std::sqrt((double)dataSrc[j] * dataSrc[j] + (double)dataSrc[j + 1] * dataSrc[j + 1]);
            }
        }
    }
    else
    {
        const double *dataSrc = src.ptr<double>();
        double *dataDst = dst.ptr<double>();

        size_t stepSrc = src.step / sizeof(dataSrc[0]);
        size_t stepDst = dst.step / sizeof(dataDst[0]);

        if (!is_1d && cn == 1)
        {
            for (k = 0; k < (cols % 2 ? 1 : 2); k++)
            {
                if (k == 1)
                    dataSrc += cols - 1, dataDst += cols - 1;
                dataDst[0] = dataSrc[0] * dataSrc[0];
                if (rows % 2 == 0)
                    dataDst[(rows - 1) * stepDst] = dataSrc[(rows - 1) * stepSrc] * dataSrc[(rows - 1) * stepSrc];

                for (j = 1; j <= rows - 2; j += 2)
                {
                    dataDst[j * stepDst] = std::sqrt(dataSrc[j * stepSrc] * dataSrc[j * stepSrc] +
                                                     dataSrc[(j + 1) * stepSrc] * dataSrc[(j + 1) * stepSrc]);
                }

                if (k == 1)
                    dataSrc -= cols - 1, dataDst -= cols - 1;
            }
        }

        for (; rows--; dataSrc += stepSrc, dataDst += stepDst)
        {
            if (is_1d && cn == 1)
            {
                dataDst[0] = dataSrc[0] * dataSrc[0];
                if (cols % 2 == 0)
                    dataDst[j1] = dataSrc[j1] * dataSrc[j1];
            }

            for (j = j0; j < j1; j += 2)
            {
                dataDst[j] = std::sqrt(dataSrc[j] * dataSrc[j] + dataSrc[j + 1] * dataSrc[j + 1]);
            }
        }
    }
}

static void divSpectrums(InputArray _srcA, InputArray _srcB, OutputArray _dst, int flags, bool conjB)
{
    Mat srcA = _srcA.getMat(), srcB = _srcB.getMat();
    int depth = srcA.depth(), cn = srcA.channels(), type = srcA.type();
    int rows = srcA.rows, cols = srcA.cols;
    int j, k;

    CV_Assert(type == srcB.type() && srcA.size() == srcB.size());
    CV_Assert(type == CV_32FC1 || type == CV_32FC2 || type == CV_64FC1 || type == CV_64FC2);

    _dst.create(srcA.rows, srcA.cols, type);
    Mat dst = _dst.getMat();

    CV_Assert(dst.data != srcA.data); // non-inplace check
    CV_Assert(dst.data != srcB.data); // non-inplace check

    bool is_1d = (flags & DFT_ROWS) || (rows == 1 || (cols == 1 &&
                                                      srcA.isContinuous() && srcB.isContinuous() && dst.isContinuous()));

    if (is_1d && !(flags & DFT_ROWS))
        cols = cols + rows - 1, rows = 1;

    int ncols = cols * cn;
    int j0 = cn == 1;
    int j1 = ncols - (cols % 2 == 0 && cn == 1);

    if (depth == CV_32F)
    {
        const float *dataA = srcA.ptr<float>();
        const float *dataB = srcB.ptr<float>();
        float *dataC = dst.ptr<float>();
        float eps = FLT_EPSILON; // prevent div0 problems

        size_t stepA = srcA.step / sizeof(dataA[0]);
        size_t stepB = srcB.step / sizeof(dataB[0]);
        size_t stepC = dst.step / sizeof(dataC[0]);

        if (!is_1d && cn == 1)
        {
            for (k = 0; k < (cols % 2 ? 1 : 2); k++)
            {
                if (k == 1)
                    dataA += cols - 1, dataB += cols - 1, dataC += cols - 1;
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if (rows % 2 == 0)
                    dataC[(rows - 1) * stepC] = dataA[(rows - 1) * stepA] / (dataB[(rows - 1) * stepB] + eps);
                if (!conjB)
                    for (j = 1; j <= rows - 2; j += 2)
                    {
                        double denom = (double)dataB[j * stepB] * dataB[j * stepB] +
                                       (double)dataB[(j + 1) * stepB] * dataB[(j + 1) * stepB] + (double)eps;

                        double re = (double)dataA[j * stepA] * dataB[j * stepB] +
                                    (double)dataA[(j + 1) * stepA] * dataB[(j + 1) * stepB];

                        double im = (double)dataA[(j + 1) * stepA] * dataB[j * stepB] -
                                    (double)dataA[j * stepA] * dataB[(j + 1) * stepB];

                        dataC[j * stepC] = (float)(re / denom);
                        dataC[(j + 1) * stepC] = (float)(im / denom);
                    }
                else
                    for (j = 1; j <= rows - 2; j += 2)
                    {

                        double denom = (double)dataB[j * stepB] * dataB[j * stepB] +
                                       (double)dataB[(j + 1) * stepB] * dataB[(j + 1) * stepB] + (double)eps;

                        double re = (double)dataA[j * stepA] * dataB[j * stepB] -
                                    (double)dataA[(j + 1) * stepA] * dataB[(j + 1) * stepB];

                        double im = (double)dataA[(j + 1) * stepA] * dataB[j * stepB] +
                                    (double)dataA[j * stepA] * dataB[(j + 1) * stepB];

                        dataC[j * stepC] = (float)(re / denom);
                        dataC[(j + 1) * stepC] = (float)(im / denom);
                    }
                if (k == 1)
                    dataA -= cols - 1, dataB -= cols - 1, dataC -= cols - 1;
            }
        }

        for (; rows--; dataA += stepA, dataB += stepB, dataC += stepC)
        {
            if (is_1d && cn == 1)
            {
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if (cols % 2 == 0)
                    dataC[j1] = dataA[j1] / (dataB[j1] + eps);
            }

            if (!conjB)
                for (j = j0; j < j1; j += 2)
                {
                    double denom = (double)(dataB[j] * dataB[j] + dataB[j + 1] * dataB[j + 1] + eps);
                    double re = (double)(dataA[j] * dataB[j] + dataA[j + 1] * dataB[j + 1]);
                    double im = (double)(dataA[j + 1] * dataB[j] - dataA[j] * dataB[j + 1]);
                    dataC[j] = (float)(re / denom);
                    dataC[j + 1] = (float)(im / denom);
                }
            else
                for (j = j0; j < j1; j += 2)
                {
                    double denom = (double)(dataB[j] * dataB[j] + dataB[j + 1] * dataB[j + 1] + eps);
                    double re = (double)(dataA[j] * dataB[j] - dataA[j + 1] * dataB[j + 1]);
                    double im = (double)(dataA[j + 1] * dataB[j] + dataA[j] * dataB[j + 1]);
                    dataC[j] = (float)(re / denom);
                    dataC[j + 1] = (float)(im / denom);
                }
        }
    }
    else
    {
        const double *dataA = srcA.ptr<double>();
        const double *dataB = srcB.ptr<double>();
        double *dataC = dst.ptr<double>();
        double eps = DBL_EPSILON; // prevent div0 problems

        size_t stepA = srcA.step / sizeof(dataA[0]);
        size_t stepB = srcB.step / sizeof(dataB[0]);
        size_t stepC = dst.step / sizeof(dataC[0]);

        if (!is_1d && cn == 1)
        {
            for (k = 0; k < (cols % 2 ? 1 : 2); k++)
            {
                if (k == 1)
                    dataA += cols - 1, dataB += cols - 1, dataC += cols - 1;
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if (rows % 2 == 0)
                    dataC[(rows - 1) * stepC] = dataA[(rows - 1) * stepA] / (dataB[(rows - 1) * stepB] + eps);
                if (!conjB)
                    for (j = 1; j <= rows - 2; j += 2)
                    {
                        double denom = dataB[j * stepB] * dataB[j * stepB] +
                                       dataB[(j + 1) * stepB] * dataB[(j + 1) * stepB] + eps;

                        double re = dataA[j * stepA] * dataB[j * stepB] +
                                    dataA[(j + 1) * stepA] * dataB[(j + 1) * stepB];

                        double im = dataA[(j + 1) * stepA] * dataB[j * stepB] -
                                    dataA[j * stepA] * dataB[(j + 1) * stepB];

                        dataC[j * stepC] = re / denom;
                        dataC[(j + 1) * stepC] = im / denom;
                    }
                else
                    for (j = 1; j <= rows - 2; j += 2)
                    {
                        double denom = dataB[j * stepB] * dataB[j * stepB] +
                                       dataB[(j + 1) * stepB] * dataB[(j + 1) * stepB] + eps;

                        double re = dataA[j * stepA] * dataB[j * stepB] -
                                    dataA[(j + 1) * stepA] * dataB[(j + 1) * stepB];

                        double im = dataA[(j + 1) * stepA] * dataB[j * stepB] +
                                    dataA[j * stepA] * dataB[(j + 1) * stepB];

                        dataC[j * stepC] = re / denom;
                        dataC[(j + 1) * stepC] = im / denom;
                    }
                if (k == 1)
                    dataA -= cols - 1, dataB -= cols - 1, dataC -= cols - 1;
            }
        }

        for (; rows--; dataA += stepA, dataB += stepB, dataC += stepC)
        {
            if (is_1d && cn == 1)
            {
                dataC[0] = dataA[0] / (dataB[0] + eps);
                if (cols % 2 == 0)
                    dataC[j1] = dataA[j1] / (dataB[j1] + eps);
            }

            if (!conjB)
                for (j = j0; j < j1; j += 2)
                {
                    double denom = dataB[j] * dataB[j] + dataB[j + 1] * dataB[j + 1] + eps;
                    double re = dataA[j] * dataB[j] + dataA[j + 1] * dataB[j + 1];
                    double im = dataA[j + 1] * dataB[j] - dataA[j] * dataB[j + 1];
                    dataC[j] = re / denom;
                    dataC[j + 1] = im / denom;
                }
            else
                for (j = j0; j < j1; j += 2)
                {
                    double denom = dataB[j] * dataB[j] + dataB[j + 1] * dataB[j + 1] + eps;
                    double re = dataA[j] * dataB[j] - dataA[j + 1] * dataB[j + 1];
                    double im = dataA[j + 1] * dataB[j] + dataA[j] * dataB[j + 1];
                    dataC[j] = re / denom;
                    dataC[j + 1] = im / denom;
                }
        }
    }
}

static void fftShift(InputOutputArray _out)
{
    Mat out = _out.getMat();

    if (out.rows == 1 && out.cols == 1)
    {
        // trivially shifted.
        return;
    }

    std::vector<Mat> planes;
    split(out, planes);

    int xMid = out.cols >> 1;
    int yMid = out.rows >> 1;

    bool is_1d = xMid == 0 || yMid == 0;

    if (is_1d)
    {
        int is_odd = (xMid > 0 && out.cols % 2 == 1) || (yMid > 0 && out.rows % 2 == 1);
        xMid = xMid + yMid;

        for (size_t i = 0; i < planes.size(); i++)
        {
            Mat tmp;
            Mat half0(planes[i], Rect(0, 0, xMid + is_odd, 1));
            Mat half1(planes[i], Rect(xMid + is_odd, 0, xMid, 1));

            half0.copyTo(tmp);
            half1.copyTo(planes[i](Rect(0, 0, xMid, 1)));
            tmp.copyTo(planes[i](Rect(xMid, 0, xMid + is_odd, 1)));
        }
    }
    else
    {
        int isXodd = out.cols % 2 == 1;
        int isYodd = out.rows % 2 == 1;
        for (size_t i = 0; i < planes.size(); i++)
        {
            // perform quadrant swaps...
            Mat q0(planes[i], Rect(0, 0, xMid + isXodd, yMid + isYodd));
            Mat q1(planes[i], Rect(xMid + isXodd, 0, xMid, yMid + isYodd));
            Mat q2(planes[i], Rect(0, yMid + isYodd, xMid + isXodd, yMid));
            Mat q3(planes[i], Rect(xMid + isXodd, yMid + isYodd, xMid, yMid));

            if (!(isXodd || isYodd))
            {
                Mat tmp;
                q0.copyTo(tmp);
                q3.copyTo(q0);
                tmp.copyTo(q3);

                q1.copyTo(tmp);
                q2.copyTo(q1);
                tmp.copyTo(q2);
            }
            else
            {
                Mat tmp0, tmp1, tmp2, tmp3;
                q0.copyTo(tmp0);
                q1.copyTo(tmp1);
                q2.copyTo(tmp2);
                q3.copyTo(tmp3);

                tmp0.copyTo(planes[i](Rect(xMid, yMid, xMid + isXodd, yMid + isYodd)));
                tmp3.copyTo(planes[i](Rect(0, 0, xMid, yMid)));

                tmp1.copyTo(planes[i](Rect(0, yMid, xMid, yMid + isYodd)));
                tmp2.copyTo(planes[i](Rect(xMid, 0, xMid + isXodd, yMid)));
            }
        }
    }

    merge(planes, out);
}
