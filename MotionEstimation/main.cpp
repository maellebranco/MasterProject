#include "trajectory.h"

#include <QApplication>
#include <iostream>
#include <vector>
#include <map>

#include <stack>
#include <ctime>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/videostab.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;
using namespace std;

// for timing
stack<clock_t> tictoc_stack;
void tic() { tictoc_stack.push(clock()); }
void toc() {
    cout << "Time elapsed: " << ((double)(clock()-tictoc_stack.top()))/CLOCKS_PER_SEC << " seconds" << endl;
    tictoc_stack.pop();
}

// for duplicates removing algorithms
bool point2fLessOperatorX(Point2f pt1,Point2f pt2)
{
    if(pt1.x!=pt2.x)
        return (pt1.x<pt2.x);
    else
        return (pt1.y<pt2.y);
}
bool point2fLessOperatorY(Point2f pt1,Point2f pt2)
{
    if(pt1.y!=pt2.y)
        return (pt1.y<pt2.y);
    else
        return (pt1.x<pt2.x);
}
bool point2fDuplicate(Point2f pt1,Point2f pt2)
{
    int d = 10;
    return ((-d<(pt1.x-pt2.x) && (pt1.x-pt2.x)<d) && (-d<(pt1.y-pt2.y) && (pt1.y-pt2.y)<d));
}

// read the video
int readingVideo(String videoPath)
{
    VideoCapture cap(videoPath);    // open the video file for reading

    if(!cap.isOpened())  // if not success, exit
    {
        cout << "Cannot open the video file!" << endl;
        return -1;
    }

    double fps = cap.get(CV_CAP_PROP_FPS);  // get the frames per seconds of the video
    cout << "Frame per seconds : " << fps << endl;
    int nbFrames = cap.get(CV_CAP_PROP_FRAME_COUNT);    // get approximate number of frammes of the video
    cout << "Number of frames : " << nbFrames << endl;

    Mat frame;

    while(true) // for each frame
    {
        bool bSuccess = cap.read(frame); // read a new frame from video
        if (!bSuccess) // if not success, break loop
        {
            int pos = cap.get(CV_CAP_PROP_POS_FRAMES);
            if(pos<(nbFrames-10)) cout << "Cannot read the frame " << pos << "from video file!" << endl;
            else cout << "End of video file" << endl;
            break;
        }
        namedWindow("MyVideo",WINDOW_NORMAL);
        imshow("MyVideo", frame);   // show the frame in "MyVideo" window
        if(waitKey((int)(1000/fps)) == 27)  // wait between frames, if 'esc' key is pressed, break loop
        {
            cout << "Visualization ended by user" << endl;
            break;
        }
    }

    return 0;
}

// calculate trajectories using SIFT descriptors and Brute Force matching
int calcTrajectoriesSIFT(vector<Trajectory> &trajectories, int &nbFrames, String videoPath)
{
    VideoCapture cap(videoPath);    // open the video file for reading

    if(!cap.isOpened())  // if not success, exit
    {
        cout << "Cannot open the video file!" << endl;
        return -1;
    }

    double fps = cap.get(CV_CAP_PROP_FPS);  // get the frames per seconds of the video
    cout << "Frame per seconds : " << fps << endl;
    nbFrames = cap.get(CV_CAP_PROP_FRAME_COUNT);    // get approximate number of frammes of the video
    cout << "Number of frames : " << nbFrames << endl;

    Mat frame1, frame2, frameOut;
    vector<KeyPoint> keypoint1, keypoint2;
    Mat descriptor1, descriptor2;
    map<int,Point2f> currentTrajectories;
    bool init = true;
    int frameNb=0;

    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create(0,3,0.08,5,1.6);  // sift keypoint extracter with default parameter but with filterx2
    BFMatcher matcher(NORM_L2, true);   // matcher declaration, with crossCheck to limit outliers

    while(true) // for each frame
    {
        if(init)    // if first frame of video
        {
            if(!cap.read(frame1)) break;    // read first frame
            sift->detect(frame1,keypoint1); // detect keypoints
            sift->compute(frame1,keypoint1,descriptor1);    // compute the keypoints descriptors
            //drawKeypoints(frame1,keypoint1,frameOut,Scalar::all(-1),DrawMatchesFlags::DRAW_OVER_OUTIMG);
            //imshow("Keypoints frame1",frame1);
            //if(waitKey((int)(1000/fps)) == 27) break;
        }

        if(!cap.read(frame2)) break;    // read next frame
        sift->detect(frame2,keypoint2); // detect keypoints
        sift->compute(frame2,keypoint2,descriptor2);    // compute the keypoints descriptors
        //drawKeypoints(frame2,keypoint2,frameOut,Scalar::all(-1),DrawMatchesFlags::DRAW_OVER_OUTIMG);
        //imshow("Keypoints frame2",frame2);
        //if(waitKey((int)(1000/fps)) == 27) break;

        vector<DMatch> matches;
        matcher.match(descriptor1,descriptor2,matches,noArray());   // compute matching
        /* alternative matching for ratio control
        vector<vector<DMatch> > possibleMatches;
        matcher.knnMatch(descriptor1,descriptor2,possibleMatches,2,noArray(),false);
        for(vector<vector<DMatch>>::const_iterator it=possibleMatches.begin(); it!=possibleMatches.end(); ++it)
        {
            DMatch match = it->at(0);
            if(match.distance <= (it->at(1)).distance*0.6) // ratio test
              matches.push_back(match);
        }
        */
        //drawMatches(frame1,keypoint1,frame2,keypoint2,matches,frameOut);

        // better flow visualization
        frame2.copyTo(frameOut);
        for(vector<DMatch>::const_iterator it_match=matches.begin(); it_match!=matches.end(); ++it_match)
        {
                Point pt1 = keypoint1[(*it_match).queryIdx].pt;
                Point pt2 = keypoint2[(*it_match).trainIdx].pt;
                circle(frameOut,pt1,3,Scalar(0,0,255),1);
                circle(frameOut,pt2,3,Scalar(255,0,0),1);
                line(frameOut,pt1,pt2,Scalar(0,255,0),2,8,0);
        }
        namedWindow("Matches",WINDOW_NORMAL);
        imshow("Matches",frameOut);
        if(waitKey((int)(1000/fps)) == 27) break;
        //

        if(init)    // if first frame of video
        {
            for(vector<DMatch>::const_iterator it_match=matches.begin(); it_match!=matches.end(); ++it_match)   // for each match
            {
                Point2f pt1 = keypoint1[(*it_match).queryIdx].pt;   // retrieve point of first keypoint
                Point2f pt2 = keypoint2[(*it_match).trainIdx].pt;   // retrieve point of second keypoint
                Trajectory trajectory(frameNb,pt1,pt2);             // create new trajectory
                trajectories.push_back(trajectory);                 // add the new trajectory
                currentTrajectories.insert(pair<int,Point2f>(trajectories.size()-1,pt2)); // add a map reference to the new trajectory
            }
            init=false;
        }
        else
        {
            map<int,Point2f> newCurrentTrajectories;
            for(vector<DMatch>::const_iterator it_match=matches.begin(); it_match!=matches.end() ; ++it_match)   // for each match
            {
                if((*it_match).distance<200)    // if good enought match
                {
                    Point2f pt1 = keypoint1[(*it_match).queryIdx].pt;
                    Point2f pt2 = keypoint2[(*it_match).trainIdx].pt;
                    //cout << pt1 << " " << pt2 << " (" << (*it_match).distance << ")" << endl;
                    int key = -1;
                    for(map<int,Point2f>::const_iterator it=currentTrajectories.begin(); it!=currentTrajectories.end() && key==-1; ++it)
                    {                           // search corresponding trajectory in the reference map
                        if(it->second == pt1)   // if found
                        {
                            key = it->first;
                            trajectories.at(key).addPoint(pt2); // add the new point to the trajectory
                            newCurrentTrajectories.insert(pair<int,Point2f>(key,pt2));  // replace the map preference
                        }
                    }
                    if(key==-1) // if not found
                    {
                        Trajectory trajectory(frameNb,pt1,pt2); // create new trajectory
                        trajectories.push_back(trajectory);     // add the new trajectory
                        newCurrentTrajectories.insert(pair<int,Point2f>(trajectories.size()-1,pt2)); // add a map reference to the new trajectory
                    }
                }
            }
            currentTrajectories=newCurrentTrajectories; // update the map reference to current trajectories
        }

        cout << "Current frame: " << ++frameNb << endl;
        frame2.copyTo(frame1);
        keypoint1 = keypoint2;
        descriptor1 = descriptor2;
    }
    nbFrames = frameNb+1; // get real number of frames

    return 0;
}

// calculate trajectories using KLT sparse optical flow (with Harris corners)
int calcTrajectoriesKLT(vector<Trajectory> &trajectories, int &nbFrames, String videoPath, int maxInitialCorners=2000, int maxNewCorners=500)
{
    VideoCapture cap(videoPath);    // open the video file for reading

    if(!cap.isOpened())  // if not success, exit
    {
        cout << "Cannot open the video file!" << endl;
        return -1;
    }

    double fps = cap.get(CV_CAP_PROP_FPS);  // get the frames per seconds of the video
    cout << "Frame per seconds : " << fps << endl;
    nbFrames = cap.get(CV_CAP_PROP_FRAME_COUNT);    // get approximate number of frammes of the video
    cout << "Number of frames : " << nbFrames << endl;

    Mat frame1, frame2, gray1, gray2, frameOut, errors;
    vector<Point2f> previousPoints, nextPoints, newPoints;
    vector<unsigned char> status;
    map<int,Point2f> currentTrajectories;
    bool init = true;
    int frameNb=0;

    while(true) // for each frame
    {

        if(init)    // if first frame of video
        {
            if(!cap.read(frame1)) break;    // read first frame
            cvtColor(frame1,gray1,COLOR_BGR2GRAY);  // convert it to greyscale
            goodFeaturesToTrack(gray1,previousPoints,maxInitialCorners,0.01,5,noArray(),3,true,0.04);   // detect strong corners (Harris)
            //cout << previousPoints << endl;
            init=false;
        }

        if(!cap.read(frame2)) break;    // read next frame
        cvtColor(frame2,gray2,COLOR_BGR2GRAY);  // convert it to greyscale
        calcOpticalFlowPyrLK(gray1,gray2,previousPoints,nextPoints,status,errors);  // calculate sparse optical flow
        //cout << nextPoints << endl;

        // optical flow visulalization
        int valid=0;
        frame2.copyTo(frameOut);
        for(vector<unsigned char>::iterator it=status.begin(); it!=status.end(); ++it)
        {
            if((*it)==1)
            {
                ++valid;
                Point pt1 = previousPoints.at(distance(status.begin(),it));
                Point pt2 = nextPoints.at(distance(status.begin(),it));
                circle(frameOut,pt1,3,Scalar(0,0,255),1);
                circle(frameOut,pt2,3,Scalar(255,0,0),1);
                line(frameOut,pt1,pt2,Scalar(0,255,0),2,8,0);
            }
        }
        namedWindow("Matches",WINDOW_NORMAL);
        imshow("Matches",frameOut);
        if(waitKey(10*(int)(1000/fps)) == 27) break;
        cout << valid << " valid matches" << endl;

        goodFeaturesToTrack(gray2,newPoints,maxNewCorners,0.01,5,noArray(),3,true,0.04);    // detect new strong corners (Harris)

        map<int,Point2f> newCurrentTrajectories;
        for(vector<unsigned char>::iterator it=status.begin(); it!=status.end(); ++it)   // for each match
        {
            if((*it)==1)    // if valid
            {
                Point2f pt1 = previousPoints.at(distance(status.begin(),it));   // retrieve first point
                Point2f pt2 = nextPoints.at(distance(status.begin(),it));       // retrieve second point
                //cout << pt1 << " " << pt2 << endl;
                int key = -1;
                for(map<int,Point2f>::const_iterator traj_it=currentTrajectories.begin(); traj_it!=currentTrajectories.end() && key==-1; ++traj_it)
                {                           // search corresponding trajectory in the reference map
                    if(traj_it->second == pt1)   // if found
                    {
                        key = traj_it->first;
                        trajectories.at(key).addPoint(pt2); // add the new point to the trajectory
                        newCurrentTrajectories.insert(pair<int,Point2f>(key,pt2));  // replace the map preference
                    }
                }
                if(key==-1) // if not found
                {
                    Trajectory trajectory(frameNb,pt1,pt2); // create new trajectory
                    trajectories.push_back(trajectory);     // add the new trajectory
                    newCurrentTrajectories.insert(pair<int,Point2f>(trajectories.size()-1,pt2)); // add a map reference to the new trajectory
                }
                newPoints.push_back(nextPoints.at(distance(status.begin(),it))); // add to new points the current valid ones
            }
        }
        currentTrajectories=newCurrentTrajectories; // update the map reference to current trajectories

        previousPoints = newPoints; // transfert and remove duplicates
        sort(previousPoints.begin(),previousPoints.end(),point2fLessOperatorY);
        previousPoints.erase(unique(previousPoints.begin(),previousPoints.end(),point2fDuplicate),previousPoints.end());
        sort(previousPoints.begin(),previousPoints.end(),point2fLessOperatorX);
        previousPoints.erase(unique(previousPoints.begin(),previousPoints.end(),point2fDuplicate),previousPoints.end());
        cout << previousPoints.size() << " tracked points" <<endl;
        //cout << previousPoints << endl;

        cout << "Current frame: " << ++frameNb << endl;
        frame2.copyTo(frame1);
        gray2.copyTo(gray1);
    }
    nbFrames = frameNb+1; // get real number of frames

    return 0;
}

// calculate global motions (affine transformation or homography) using RANSAC and with trajectory ponderation
void calcGlobalMotions(vector<Mat> &globalMotions, vector<Trajectory> &trajectories, int nbFrames, bool discardUniqueMatches=true, bool safePoints=true, bool homography=false)
{
    cout << trajectories.size() << " trajectories" << endl;
    //for(vector<Trajectory>::const_iterator it=trajectories.begin(); it!=trajectories.end(); ++it) it->showTrajectory();

    if(discardUniqueMatches)    // discard unique match (often error)
    {
        for(vector<Trajectory>::iterator it=trajectories.begin(); it!=trajectories.end();)
        {
            if(it->getSize()<3) trajectories.erase(it);
            else ++it;
        }
        cout << trajectories.size() << " remaining trajectories" << endl;
        //for(vector<Trajectory>::const_iterator it=trajectories.begin(); it!=trajectories.end(); ++it) it->showTrajectory();
    }

    Mat globalMotion;
    for(int i=1; i<nbFrames; ++i)   // for each frame
    {
        map<int,int> weights;
        for(vector<Trajectory>::iterator it=trajectories.begin(); it!=trajectories.end(); ++it)
        {
            int start = it->getStart(); // get starting frame
            int end = it->getEnd();     // get ending frame
            if(start < i && i < end)    // search corresponding trajectories
            {
                int weight = min(i-start,end-i);    // weight calculation (simple triangle function, reward long trajectories and points in middle of the trajectory
                weights.insert(pair<int,int>(distance(trajectories.begin(),it),weight)); // create a reference
            }
        }

        vector<Point2f> points1, points2;

        if(safePoints)  // to avoid errors of opencv estimate motion function
        {
            points1.push_back(Point2f(0,0));
            points1.push_back(Point2f(0,1));
            points1.push_back(Point2f(1,0));
            points2.push_back(Point2f(0,0));
            points2.push_back(Point2f(0,1));
            points2.push_back(Point2f(1,0));
        }

        for(map<int,int>::const_iterator it=weights.begin(); it!=weights.end(); ++it)   // for each corresponding trajectory
        {
            Point2f pt1 = trajectories.at(it->first).getPoint(i-1);
            Point2f pt2 = trajectories.at(it->first).getPoint(i);
            //cout << pt1 << " " << pt2 << " (" << it->second << ")" << endl;

            for(int w=0; w<(it->second); ++w)   // weight applied by duplication of points
            {
                points1.push_back(pt1); // add the point of the previous frame
                points2.push_back(pt2); // add the point of the current frame
            }
        }

        if(homography)  // homography or affine transformation
            globalMotion = findHomography(points1,points2,FM_RANSAC);    // estimate the global motion using RANSAC
        else
            globalMotion = videostab::estimateGlobalMotionRansac(points1,points2,videostab::MM_AFFINE,videostab::RansacParams::default2dMotion(videostab::MM_AFFINE),0,0);
            //globalMotion = videostab::estimateGlobalMotionLeastSquares(points1,points2,videostab::MM_AFFINE,0);   // using Least Squares (less accurate)

        cout << "Frame " << i << endl << globalMotion << endl;
        globalMotions.push_back(globalMotion);  // add the global motion between the two frames to the vector of global motions
    }
}

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    tic();
    vector<Trajectory> trajectories;
    int nbFrames;

    //calcTrajectoriesSIFT(trajectories,nbFrames,"/home/maelle/Desktop/VideosTest/stable200.mp4");
    calcTrajectoriesKLT(trajectories,nbFrames,"/home/maelle/Desktop/VideosTest/stable100.mp4",2000,500);
    toc();

    tic();
    vector<Mat> globalMotions;   // output of motion estimation step

    calcGlobalMotions(globalMotions,trajectories,nbFrames,true,true,false);
    toc();

    return a.exec();
}
