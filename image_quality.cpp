/*
****************************************
* This file contains the code to 
* analyse the frames produced by
* the Block Matching Correlation
* (BMC) algorithm. Two metrics are used
* here :
* 1. Peak Signal to Noise Ratio (PSNR)
* 2. Structural Similarity Index (SSIM)
* This code has been taken from :
* docs.opencv.org
****************************************
*/

#include "image_quality.hpp"

double getPSNR(const UMat &I1, const UMat &I2)
{
    Mat s1;
    absdiff(I1, I2, s1);      // |I1 - I2|
    s1.convertTo(s1, CV_32F); // cannot make a square on 8 bits
    s1 = s1.mul(s1);          // |I1 - I2|^2

    Scalar s = sum(s1); // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if (sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double mse = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

Scalar getMSSIM(const UMat &i1, const UMat &i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d = CV_32F;

    Mat I1, I2;
    i1.convertTo(I1, d); // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    Mat I2_2 = I2.mul(I2);  // I2^2
    Mat I1_2 = I1.mul(I1);  // I1^2
    Mat I1_I2 = I1.mul(I2); // I1 * I2

    /*************************** END INITS **********************************/

    Mat mu1, mu2; // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);

    Mat mu1_2 = mu1.mul(mu1);
    Mat mu2_2 = mu2.mul(mu2);
    Mat mu1_mu2 = mu1.mul(mu2);

    Mat sigma1_2, sigma2_2, sigma12;

    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    subtract(sigma1_2, mu1_2, sigma1_2); // sigma1_2 -= mu1_2;

    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    subtract(sigma2_2, mu2_2, sigma2_2); // sigma2_2 -= mu2_2;

    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    subtract(sigma12, mu1_mu2, sigma12); // sigma12 -= mu1_mu2;

    ///////////////////////////////// FORMULA ////////////////////////////////
    Mat t1, t2, t3;
    add(mu1_mu2.mul(2), C1, t1); // // t1 = 2 * mu1_mu2 + C1;
    add(sigma12.mul(2), C2, t2); // t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);             // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    add(mu1_2, mu2_2, t1);
    add(t1, C1, t1); //t1 = mu1_2 + mu2_2 + C1;
    add(sigma1_2, sigma2_2, t2);
    add(t2, C2, t2); // t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2); // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    Mat ssim_map;
    divide(t3, t1, ssim_map); // ssim_map =  t3./t1;

    Scalar mssim = mean(ssim_map); // mssim = average of ssim map
    return mssim;
}

void writeValuesToFile(ofstream &resFile, int frameNo, double psnr, Scalar mssim)
{
    if (!resFile)
    {
        cout << "Could not open file\n";
        exit(-1);
    }
    resFile << "Frame " << frameNo << "\t\t" << psnr << " dB\t\t";
    resFile << " R " << mssim.val[2] * 100 << "%";
    resFile << " G " << mssim.val[1] * 100 << "%";
    resFile << " B " << mssim.val[0] * 100 << "%";
    resFile << endl;
}

void calcQuality()
{
    // To be changed
    vector<UMat> interpolatedFrames, originalFrames;
    size_t count;
    double psnr;
    Scalar mssim;
    // reading original frames
    originalFrames = readFrames(INPUT_VIDEO);
    // reading interpolated frames
    interpolatedFrames = readFrames(INTERPOLATED_VIDEO);

    // for (size_t i = 1; i < count; i+=2)
    // {
    //     img = imread(fn1[i], IMREAD_GRAYSCALE);
    //     if (!img.data)
    //     {
    //         cout << "Could not open or find the image" << std::endl;
    //         exit(-1);
    //     }
    //     interpolatedImages.push_back(img);
    // }

    // // reading actual frames
    // glob("video/*.jpg", fn2, false);
    // count = fn2.size();
    // for (size_t i = 1; i < 20; i += 2)
    // {
    //     img = imread(fn2[i], IMREAD_GRAYSCALE);
    //     if (!img.data)
    //     {
    //         cout << "Could not open or find the image" << std::endl;
    //         exit(-1);
    //     }
    //     originalImages.push_back(img);
    // }

    // performing analysis and storing the result
    if (interpolatedFrames.size() != originalFrames.size())
    {
        cout << "Number of images is different\n";
        exit(-1);
    }
    count = originalFrames.size();
    // open file
    ofstream resFile(ANALYSIS_FILE, ios_base::app);
    resFile << "Frame   \t\tPSNR      \t\tSSIM\n";
    for (size_t i = 1; i < count - 1; i += 2)
    {
        psnr = getPSNR(interpolatedFrames[i], originalFrames[i]);
        mssim = getMSSIM(interpolatedFrames[i], originalFrames[i]);
        writeValuesToFile(resFile, i + 1, psnr, mssim);
    }
    resFile.close();
}