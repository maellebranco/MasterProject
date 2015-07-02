#include "mainwindow.h"
#include "trajectory.h"
#include "videomotionestimation.h"

#include <QApplication>
#include <iostream>
#include <vector>
#include <map>

#include <stack>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/videostab.hpp>
#include <opencv2/calib3d.hpp>

#include <ClpSimplex.hpp>

using namespace cv;
using namespace std;

// for timing
stack<clock_t> tictoc_stack;
void tic() { tictoc_stack.push(clock()); }
void toc() {
    cout << "Time elapsed: " << ((double)(clock()-tictoc_stack.top()))/CLOCKS_PER_SEC << " seconds" << endl;
    tictoc_stack.pop();
}

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    //MainWindow w;
    //w.show();

    string videoName = "shake100";

    tic();
    vector<Trajectory> trajectories;
    int nbFrames;

    //calcTrajectoriesSIFT(trajectories,nbFrames,"/home/maelle/Desktop/VideosTest/trans100.avi");
    videoMotionEstimation::calcTrajectoriesKLT(trajectories,nbFrames,"/home/maelle/Desktop/VideosTest/"+videoName+".avi",2000,500);
    toc();

    tic();
    vector<Mat> globalMotions;   // output of motion estimation step

    videoMotionEstimation::calcGlobalMotions(globalMotions,trajectories,nbFrames,true,true,false);
    toc();

    tic();
    // wrapping
    VideoCapture cap("/home/maelle/Desktop/VideosTest/"+videoName+".avi");  // open the original video

    if (!cap.isOpened())    // if not success, exit
    {
        cout << "Cannot open the video file!" << endl;
        return -1;
    }

    double fps = cap.get(CAP_PROP_FPS); // get framerate of the video
    Size frameSize(1920,1080);          // set frames size
    int fourcc = VideoWriter::fourcc('H','2','6','4');  // set codec code
    //cout << "Video parameters: " << frameSize << ", codec:" << fourcc << ", fps:" << fps << endl;

    VideoWriter videoWriter("/home/maelle/Desktop/VideosOut/"+videoName+"_StableS.avi",fourcc,fps,frameSize,true);  // initialize the VideoWriter object

    if (!videoWriter.isOpened())    // if not success, exit
    {
        cout << "Failed to write the video!" << endl;
        return -1;
    }

    Mat frame, stableFrame, warp;
    if(!cap.read(frame))    // first frame
        videoWriter.write(stableFrame);
    for(int i=0; i<nbFrames-1; ++i)   // for each frame
    {
        if(!cap.read(frame)) break;    // read frame

        cout << "Current frame: " << i+1 << endl;

        warp = globalMotions.at(i).inv();
        cout << warp << endl;

        warpPerspective(frame,stableFrame,warp,frameSize);   // apply transformation on the frame (with default options)

        /*
        Mat croppedRef(stableFrame,Rect(96,54,1920-96,1080-54));
        croppedRef.copyTo(stableFrame);
        resize(stableFrame,stableFrame,frameSize,0,0,CV_INTER_CUBIC);
        */

        namedWindow("Stabilized Video",WINDOW_NORMAL);
        imshow("Stabilized Video", stableFrame);
        if(waitKey(10*(int)(1000/fps)) == 27) break;

        videoWriter.write(stableFrame);   // write the frame into the file
    }

    toc();

    return a.exec();
}

/* void optimalPathEstimation()    // not working...
{
    double w = 1920;   // frame weight
    double h = 1080;   // frame height
    double cm = 20;     // crop marge factor (1/cm croped)
    double corners[8] = {w/cm,h/cm,w-(w/cm),h/cm,w/cm,h-(h/cm),w-(w/cm),h-(h/cm)}; // (x,y) top-left, top-right, bottom-left, bottom-right

    // empty model
    ClpSimplex  model;

    // objective
    double objValueBase1[6] = {1,1,100,100,100,100}; // affine part vs translation part (100:1)
    double objValueBase2[18] = {10,10,10,10,10,10,1,1,1,1,1,1,100,100,100,100,100,100}; // derivative D1/D2/D3 weights (10:1:100)

    // columns bounds
    double ColumnLowerBase[24] = {-w,-h,0.9,-0.1,-0.1,0.9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}; // proximity constraints and positive slack variables
    double ColumnUpperBase[24] = {w,h,1.1,0.1,0.1,1.1,COIN_DBL_MAX,COIN_DBL_MAX,COIN_DBL_MAX,COIN_DBL_MAX,COIN_DBL_MAX,COIN_DBL_MAX,COIN_DBL_MAX,COIN_DBL_MAX,
                                  COIN_DBL_MAX,COIN_DBL_MAX,COIN_DBL_MAX,COIN_DBL_MAX,COIN_DBL_MAX,COIN_DBL_MAX,COIN_DBL_MAX,COIN_DBL_MAX,COIN_DBL_MAX,COIN_DBL_MAX};
    // create space for columns
    model.resize(0,24*nbFrames);

    // fill in
    for(int f=0;f<nbFrames;++f)
    {
        int i;
        for(i=0;i<6;++i)    // objective (pt parameters)
            model.setObjectiveCoefficient(f*6+i,objValueBase1[i]);
        for(i=0;i<18;++i)   // objective (slack variables)
            model.setObjectiveCoefficient(nbFrames*6+f*18+i,objValueBase2[i]);
        for (i =0;i<24;++i) // columns bounds
        {
             model.setColumnLower(i,ColumnLowerBase[i]);
             model.setColumnUpper(i,ColumnUpperBase[i]);
        }

        // proximity constraint rows (2)
        int row1Index[2] = {f*6+3,f*6+4};
        double row1Value[2] = {1,1};
        model.addRow(2,row1Index,row1Value,-0.05,0.05);
        int row2Index[2] = {f*6+2,f*6+5};
        double row2Value[2] = {1,-1};
        model.addRow(2,row2Index,row2Value,-0.1,0.1);

        // inclusion constraint rows (8)
        int rowIncl1Index[3] = {f*6+0,f*6+2,f*6+3};
        int rowIncl2Index[3] = {f*6+1,f*6+4,f*6+5};
        for(i=0;i<8;i+=2)
        {
            double rowValue[3] = {1,corners[i],corners[i+1]};
            model.addRow(3,rowIncl1Index,rowValue,0,w);
            model.addRow(3,rowIncl2Index,rowValue,0,h);
        }

        // smoothness parameters (known camera path)
        double fdx[3];
        double fdy[3];
        double fa[3];
        double fb[3];
        double fc[3];
        double fd[3];
        if(f==nbFrames-1)   // last frame, use identity transformation for "next" frames
        {
            for(i=0;i<3;++i)
            {
                fdx[i] = 0;
                fdy[i] = 0;
                fa[i] = 1;
                fb[i] = 0;
                fc[i] = 0;
                fd[i] = 1;
            }
        }
        else if(f==nbFrames-2)  // before last frame, identity for Ft+2 and Ft+3
        {
            fdx[0] = globalMotions.at(f).at<float>(Point(2,0));
            fdy[0] = globalMotions.at(f).at<float>(Point(2,1));
            fa[0] = globalMotions.at(f).at<float>(Point(0,0));
            fb[0] = globalMotions.at(f).at<float>(Point(1,0));
            fc[0] = globalMotions.at(f).at<float>(Point(0,1));
            fd[0] = globalMotions.at(f).at<float>(Point(1,1));
            for(i=1;i<3;++i)
            {
                fdx[i] = 0;
                fdy[i] = 0;
                fa[i] = 1;
                fb[i] = 0;
                fc[i] = 0;
                fd[i] = 1;
            }
        }
        else if(f==nbFrames-3)  // before before last frame, identity for Ft+3
        {
            for(i=0;i<2;++i)
            {
                fdx[i] = globalMotions.at(f+i).at<double>(2,0);
                fdy[i] = globalMotions.at(f+i).at<double>(2,1);
                fa[i] = globalMotions.at(f+i).at<double>(0,0);
                fb[i] = globalMotions.at(f+i).at<double>(1,0);
                fc[i] = globalMotions.at(f+i).at<double>(0,1);
                fd[i] = globalMotions.at(f+i).at<double>(1,1);
            }
            fdx[2] = 0;
            fdy[2] = 0;
            fa[2] = 1;
            fb[2] = 0;
            fc[2] = 0;
            fd[2] = 1;
        }
        else
        {
            for(i=0;i<3;++i)
            {
                fdx[i] = globalMotions.at(f+i).at<double>(2,0);
                fdy[i] = globalMotions.at(f+i).at<double>(2,1);
                fa[i] = globalMotions.at(f+i).at<double>(0,0);
                fb[i] = globalMotions.at(f+i).at<double>(1,0);
                fc[i] = globalMotions.at(f+i).at<double>(0,1);
                fd[i] = globalMotions.at(f+i).at<double>(1,1);
            }
        }

        // smoothness D1 rows (12)
        int rowD1Index1[4] = {f*6+0,f*6+6+0,f*6+6+1,nbFrames*6+f*18+0};
        int rowD1Index2[4] = {f*6+1,f*6+6+0,f*6+6+1,nbFrames*6+f*18+1};
        int rowD1Index3[4] = {f*6+2,f*6+6+2,f*6+6+4,nbFrames*6+f*18+2};
        int rowD1Index4[4] = {f*6+3,f*6+6+3,f*6+6+5,nbFrames*6+f*18+3};
        int rowD1Index5[4] = {f*6+4,f*6+6+2,f*6+6+4,nbFrames*6+f*18+4};
        int rowD1Index6[4] = {f*6+5,f*6+6+3,f*6+6+5,nbFrames*6+f*18+5};
        double rowD1Value1[4] = {-1,fa[0],fb[0],-1};
        double rowD1Value2[4] = {-1,fa[0],fb[0],1};
        double rowD1Value3[4] = {-1,fc[0],fd[0],-1};
        double rowD1Value4[4] = {-1,fc[0],fd[0],1};
        model.addRow(4,rowD1Index1,rowD1Value1,-COIN_DBL_MAX,-fdx[0]);
        model.addRow(4,rowD1Index1,rowD1Value2,-fdx[0],COIN_DBL_MAX);
        model.addRow(4,rowD1Index2,rowD1Value3,-COIN_DBL_MAX,-fdy[0]);
        model.addRow(4,rowD1Index2,rowD1Value4,-fdy[0],COIN_DBL_MAX);
        model.addRow(4,rowD1Index3,rowD1Value1,-COIN_DBL_MAX,0);
        model.addRow(4,rowD1Index3,rowD1Value2,0,COIN_DBL_MAX);
        model.addRow(4,rowD1Index4,rowD1Value1,-COIN_DBL_MAX,0);
        model.addRow(4,rowD1Index4,rowD1Value2,0,COIN_DBL_MAX);
        model.addRow(4,rowD1Index5,rowD1Value3,-COIN_DBL_MAX,0);
        model.addRow(4,rowD1Index5,rowD1Value4,0,COIN_DBL_MAX);
        model.addRow(4,rowD1Index6,rowD1Value3,-COIN_DBL_MAX,0);
        model.addRow(4,rowD1Index6,rowD1Value4,0,COIN_DBL_MAX);

        // smoothness D2 rows (12)
        int rowD2Index1[6] = {f*6+0,f*6+6+0,f*6+6+1,f*6+12+0,f*6+12+1,nbFrames*6+f*18+6+0};
        int rowD2Index2[6] = {f*6+1,f*6+6+0,f*6+6+1,f*6+12+0,f*6+12+1,nbFrames*6+f*18+6+1};
        int rowD2Index3[6] = {f*6+2,f*6+6+3,f*6+6+4,f*6+12+2,f*6+12+4,nbFrames*6+f*18+6+2};
        int rowD2Index4[6] = {f*6+3,f*6+6+3,f*6+6+5,f*6+12+3,f*6+12+5,nbFrames*6+f*18+6+3};
        int rowD2Index5[6] = {f*6+4,f*6+6+2,f*6+6+4,f*6+12+2,f*6+12+4,nbFrames*6+f*18+6+4};
        int rowD2Index6[6] = {f*6+5,f*6+6+3,f*6+6+5,f*6+12+3,f*6+12+5,nbFrames*6+f*18+6+5};
        double rowD2Value1[6] = {1,-1-fa[0],-fb[0],fa[1],fb[1],-1};
        double rowD2Value2[6] = {1,-1-fa[0],-fb[0],fa[1],fb[1],1};
        double rowD2Value3[6] = {1,-fc[0],-1-fd[0],fc[1],fd[1],-1};
        double rowD2Value4[6] = {1,-fc[0],-1-fd[0],fc[1],fd[1],1};
        model.addRow(6,rowD2Index1,rowD2Value1,-COIN_DBL_MAX,fdx[0]-fdx[1]);
        model.addRow(6,rowD2Index1,rowD2Value2,fdx[0]-fdx[1],COIN_DBL_MAX);
        model.addRow(6,rowD2Index2,rowD2Value3,-COIN_DBL_MAX,fdy[0]-fdy[1]);
        model.addRow(6,rowD2Index2,rowD2Value4,fdy[0]-fdy[1],COIN_DBL_MAX);
        model.addRow(6,rowD2Index3,rowD2Value1,-COIN_DBL_MAX,0);
        model.addRow(6,rowD2Index3,rowD2Value2,0,COIN_DBL_MAX);
        model.addRow(6,rowD2Index4,rowD2Value1,-COIN_DBL_MAX,0);
        model.addRow(6,rowD2Index4,rowD2Value2,0,COIN_DBL_MAX);
        model.addRow(6,rowD2Index5,rowD2Value3,-COIN_DBL_MAX,0);
        model.addRow(6,rowD2Index5,rowD2Value4,0,COIN_DBL_MAX);
        model.addRow(6,rowD2Index6,rowD2Value3,-COIN_DBL_MAX,0);
        model.addRow(6,rowD2Index6,rowD2Value4,0,COIN_DBL_MAX);

        // smoothness D3 rows (12)
        int rowD3Index1[8] = {f*6+0,f*6+6+0,f*6+6+1,f*6+12+0,f*6+12+1,f*6+18+0,f*6+18+1,nbFrames*6+f*18+12+0};
        int rowD3Index2[8] = {f*6+1,f*6+6+0,f*6+6+1,f*6+12+0,f*6+12+1,f*6+18+0,f*6+18+1,nbFrames*6+f*18+12+1};
        int rowD3Index3[8] = {f*6+2,f*6+6+3,f*6+6+4,f*6+12+2,f*6+12+4,f*6+18+2,f*6+18+4,nbFrames*6+f*18+12+2};
        int rowD3Index4[8] = {f*6+3,f*6+6+3,f*6+6+5,f*6+12+3,f*6+12+5,f*6+18+3,f*6+18+5,nbFrames*6+f*18+12+3};
        int rowD3Index5[8] = {f*6+4,f*6+6+2,f*6+6+4,f*6+12+2,f*6+12+4,f*6+18+2,f*6+18+4,nbFrames*6+f*18+12+4};
        int rowD3Index6[8] = {f*6+5,f*6+6+3,f*6+6+5,f*6+12+3,f*6+12+5,f*6+18+3,f*6+18+5,nbFrames*6+f*18+12+5};
        double rowD3Value1[8] = {-1,2+fa[0],fb[0],-1-2*fa[1],-2*fb[1],fa[2],fb[2],-1};
        double rowD3Value2[8] = {-1,2+fa[0],fb[0],-1-2*fa[1],-2*fb[1],fa[2],fb[2],1};
        double rowD3Value3[8] = {-1,fc[0],2+fd[0],-2*fc[1],-1-2*fd[1],fc[2],fd[2],-1};
        double rowD3Value4[8] = {-1,fc[0],2+fd[0],-2*fc[1],-1-2*fd[1],fc[2],fd[2],1};
        model.addRow(8,rowD3Index1,rowD3Value1,-COIN_DBL_MAX,fdx[2]-2*fdx[1]+fdx[0]);
        model.addRow(8,rowD3Index1,rowD3Value2,fdx[2]-2*fdx[1]+fdx[0],COIN_DBL_MAX);
        model.addRow(8,rowD3Index2,rowD3Value3,-COIN_DBL_MAX,fdy[2]-2*fdy[1]+fdy[0]);
        model.addRow(8,rowD3Index2,rowD3Value4,fdy[2]-2*fdy[1]+fdy[0],COIN_DBL_MAX);
        model.addRow(8,rowD3Index3,rowD3Value1,-COIN_DBL_MAX,0);
        model.addRow(8,rowD3Index3,rowD3Value2,0,COIN_DBL_MAX);
        model.addRow(8,rowD3Index4,rowD3Value1,-COIN_DBL_MAX,0);
        model.addRow(8,rowD3Index4,rowD3Value2,0,COIN_DBL_MAX);
        model.addRow(8,rowD3Index5,rowD3Value3,-COIN_DBL_MAX,0);
        model.addRow(8,rowD3Index5,rowD3Value4,0,COIN_DBL_MAX);
        model.addRow(8,rowD3Index6,rowD3Value3,-COIN_DBL_MAX,0);
        model.addRow(8,rowD3Index6,rowD3Value4,0,COIN_DBL_MAX);
    }

    // solve problem using simplex algorithm
    model.dual();

    // print solution
    const double * solution = model.primalColumnSolution();
    for(int i=0;i<(nbFrames*6);++i)
        cout << solution[i] << endl;
}
*/
