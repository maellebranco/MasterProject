#ifndef TRAJECTORY_H
#define TRAJECTORY_H

#include <vector>
#include <opencv2/opencv.hpp>

class Trajectory
{
private:
    std::vector<cv::Point2f> points;
    int startFrame;
public:
    Trajectory(int,cv::Point2f,cv::Point2f);
    void addPoint(cv::Point2f);
    void showTrajectory() const;
    int getSize() const;
    int getStart() const;
    int getEnd() const;
    cv::Point2f getPoint(int) const;
};

#endif // TRAJECTORY_H
