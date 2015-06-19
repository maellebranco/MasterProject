#include "trajectory.h"

Trajectory::Trajectory(int frameNb, cv::Point2f pt1, cv::Point2f pt2)
{
    startFrame = frameNb;
    points.push_back(pt1);
    points.push_back(pt2);
}

void Trajectory::addPoint(cv::Point2f pt)
{
    points.push_back(pt);
}

void Trajectory::showTrajectory() const
{
    std::cout << "Start frame: " << startFrame << std::endl;
    std::cout << "Points: " << points << ";" << std::endl;
}

int Trajectory::getSize() const
{
    return points.size();
}

int Trajectory::getStart() const
{
    return startFrame;
}

int Trajectory::getEnd() const
{
    return startFrame+points.size()-1;
}

cv::Point2f Trajectory::getPoint(int frameNb) const
{
    return points.at(frameNb-startFrame);
}
