#ifndef VIDEOMOTIONESTIMATION_H
#define VIDEOMOTIONESTIMATION_H

#include "trajectory.h"

#include <string>
#include <vector>
#include <opencv2/core.hpp>

using std::string;
using std::vector;
using cv::Mat;

class videoMotionEstimation
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
    static void calcGlobalMotions(vector<Mat> &globalMotions, vector<Trajectory> &trajectories, int nbFrames, bool discardUniqueMatches, bool safePoints, bool homography);
};

#endif // VIDEOMOTIONESTIMATION_H
