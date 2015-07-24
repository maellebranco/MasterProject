#ifndef VIDEOMOTIONESTIMATION_H
#define VIDEOMOTIONESTIMATION_H

#include "trajectory.h"

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/videostab.hpp>

using std::string;
using std::vector;
using cv::Mat;

class VideoMotionEstimation
{
public:
   // for duplicates removing algorithms
    static bool point2fLessOperatorX(cv::Point2f pt1,cv::Point2f pt2);
    static bool point2fLessOperatorY(cv::Point2f pt1,cv::Point2f pt2);
    static bool point2fDuplicate(cv::Point2f pt1,cv::Point2f pt2);

    // read the video
    static int readingVideo(string videoPath);

    // calculate trajectories using SIFT descriptors and Brute Force matching
    static int calcTrajectoriesSIFT(vector<Trajectory> &trajectories, int &nbFrames, string videoPath);

    // calculate trajectories using KLT sparse optical flow (with Harris corners)
    static int calcTrajectoriesKLT(vector<Trajectory> &trajectories, int &nbFrames, string videoPath, int maxInitialCorners, int maxNewCorners);

    // calculate global motions (affine transformation or homography) using RANSAC and with trajectory ponderation
    static void calcGlobalMotions(vector<Mat> &globalMotions, cv::videostab::MotionModel model, vector<Trajectory> &trajectories, int nbFrames,
                                  bool discardUniqueMatches=true, bool safePoints=true);

    // convert Translation matrices to vectors and scale timestamps
    static void convertMatrixData(int nbFrames, vector<long int> timestamps, vector<Mat> translations,
                                  vector<double> &scaledTimestamps, vector<double> &translationsX, vector<double> &translationsY);

    // convert Affine matrices to vectors
    static void convertAffineData(int nbFrames, vector<Mat> affines, vector<double> &affinesA, vector<double> &affinesB, vector<double> &affinesC, vector<double> &affinesD,
                                  vector<double> &affinesTx, vector<double> &affinesTy, vector<double> &affinesSkew, vector<double> &affinesRatio);    

    // limit the skew and ratio transformations
    static void limitationFilter(vector<double> &affinesTx, vector<double> &affinesTy, vector<double> &affinesSkew, vector<double> &affinesRatio);
};

#endif // VIDEOMOTIONESTIMATION_H
