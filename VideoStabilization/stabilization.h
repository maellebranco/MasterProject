#ifndef STABILIZATION_H
#define STABILIZATION_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>

using std::string;
using std::vector;
using cv::Mat;

class Stabilization
{
public:
    // single pole low pass RII filter
    static void filter(float freq, vector<double> inSignal, vector<double> &outSignal);

    // four stage low-pass RII filter
    static void filter4Stages(float freq, vector<double> inSignal, vector<double> &outSignal);

    // offset filter delay
    static void offsetDelay(int offsetFrames, vector<double> inSignal, vector<double> &outSignal);   

    // compute filtered/wanted motions and thus unwanted motions (only translations)
    static void calcUnwantedMotion(float freq, vector<double> translationsX, vector<double> translationsY,
                                   vector<double> &filteredTransX, vector<double> &filteredTransY,
                                   vector<double> &unwantedTransX, vector<double> &unwantedTransY);

    // compute filtered/wanted motions and thus unwanted motions
    static void calcUnwantedMotion(float freq, vector<double> affinesA, vector<double> affinesB, vector<double> affinesC, vector<double> affinesD,
                                   vector<double> affinesTx, vector<double> affinesTy, vector<double> affinesSkew, vector<double> affinesRatio,
                                   vector<double> &filteredAffinesA, vector<double> &filteredAffinesB, vector<double> &filteredAffinesC, vector<double> &filteredAffinesD,
                                   vector<double> &filteredAffinesTx, vector<double> &filteredAffinesTy, vector<double> &filteredAffinesSkew, vector<double> &filteredAffinesRatio,
                                   vector<double> &unwantedAffinesA, vector<double> &unwantedAffinesB, vector<double> &unwantedAffinesC, vector<double> &unwantedAffinesD,
                                   vector<double> &unwantedAffinesTx, vector<double> &unwantedAffinesTy, vector<double> &unwantedAffinesSkew, vector<double> &unwantedAffinesRatio);

        // compose transformation matrices of unwanted translations
    static void unwantedTransMotions(vector<double> unwantedTransX, vector<double> unwantedTransY,vector<Mat> &unwantedMotions);

    // compose transformation matrices of unwanted affines transformations
    static void unwantedAffineMotions(vector<double> unwantedAffinesA, vector<double> unwantedAffinesB, vector<double> unwantedAffinesC, vector<double> unwantedAffinesD,
                                      vector<double> unwantedAffinesTx, vector<double> unwantedAffinesTy, vector<double> unwantedAffinesSkew, vector<double> unwantedAffinesRatio,
                                      vector<Mat> &unwantedMotions);

    // compute crop window size from maximum translation correction
    static void cropSize(vector<double> unwantedTransX, vector<double> unwantedTransY, int &cropX, int &cropY);

    // wrap transformation matrices of unwanted motions to video and write stabilized video file
    static int stabilizeVideo(string folderName, string videoName, string stableVideoName, int nbFrames, vector<Mat> unwantedMotions, int cropX, int cropY);

    // [not working...] compute optimal camera path using linear programming
    static void optimalPathEstimation(int nbFrames, vector<Mat> globalMotions);
};

#endif // STABILIZATION_H
