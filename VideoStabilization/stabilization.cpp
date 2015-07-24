#include "stabilization.h"

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ClpSimplex.hpp>

using namespace cv;
using namespace std;

// single pole low pass RII filter
void Stabilization::filter(float freq, vector<double> inSignal, vector<double> &outSignal)
{
    double x = exp(-2*M_PI*freq);
    double value = 0;
    for(vector<double>::const_iterator it=inSignal.begin(); it!=inSignal.end(); ++it)
    {
        value = x*value + (1-x)*(*it);
        outSignal.push_back(value);
    }
}

// four stage low-pass RII filter
void Stabilization::filter4Stages(float freq, vector<double> inSignal, vector<double> &outSignal)
{
    double x = exp(-14.445*freq);
    double value,v1,v2,v3,v4;
    v4 = (1-x)*(1-x)*(1-x)*(1-x)*inSignal[0];
    v3 = (1-x)*(1-x)*(1-x)*(1-x)*inSignal[1] +4*x*v4;
    v2 = (1-x)*(1-x)*(1-x)*(1-x)*inSignal[2] +4*x*v3 -6*x*x*v4;
    v1 = (1-x)*(1-x)*(1-x)*(1-x)*inSignal[3] +4*x*v2 -6*x*x*v3 +4*x*x*x*v4;
    outSignal.push_back(v4);
    outSignal.push_back(v3);
    outSignal.push_back(v2);
    outSignal.push_back(v1);
    for(unsigned int i=4; i<inSignal.size(); ++i)
    {
        value = (1-x)*(1-x)*(1-x)*(1-x)*inSignal[i] +4*x*v1 -6*x*x*v2 +4*x*x*x*v3 -x*x*x*x*v4;
        outSignal.push_back(value);
        v4=v3;
        v3=v2;
        v2=v1;
        v1=value;
    }
}

// offset filter delay
void Stabilization::offsetDelay(int offsetFrames, vector<double> inSignal, vector<double> &outSignal)
{
    for(unsigned int i=offsetFrames; i<inSignal.size(); ++i)
        outSignal.push_back(inSignal.at(i));
    for(int i=1; i<=offsetFrames; ++i)
         outSignal.push_back(inSignal.back()-i*(inSignal.back()/offsetFrames));   // linear interpolation
}

// compute filtered/wanted motions and thus unwanted motions (only translations)
void Stabilization::calcUnwantedMotion(float freq, vector<double> translationsX, vector<double> translationsY,
                        vector<double> &filteredTransX, vector<double> &filteredTransY,
                        vector<double> &unwantedTransX, vector<double> &unwantedTransY)
{
    float f = freq/30; // f = fc/fs (fc ?Hz, fs 30fps)
    vector<double> fTranslationsX,fTranslationsY;
    filter4Stages(f,translationsX,fTranslationsX);
    filter4Stages(f,translationsY,fTranslationsY);

    float d = freq*30; // d = fc*fs (fc ?Hz, fs 30fps)
    offsetDelay(d,fTranslationsX,filteredTransX);
    offsetDelay(d,fTranslationsY,filteredTransY);

    for(unsigned int i=0; i<translationsX.size(); ++i)
    {
        unwantedTransX.push_back(translationsX.at(i) - filteredTransX.at(i));
        unwantedTransY.push_back(translationsY.at(i) - filteredTransY.at(i));
    }
}

// compute filtered/wanted motions and thus unwanted motions
void Stabilization::calcUnwantedMotion(float freq, vector<double> affinesA, vector<double> affinesB, vector<double> affinesC, vector<double> affinesD,
                                       vector<double> affinesTx, vector<double> affinesTy, vector<double> affinesSkew, vector<double> affinesRatio,
                                       vector<double> &filteredAffinesA, vector<double> &filteredAffinesB, vector<double> &filteredAffinesC, vector<double> &filteredAffinesD,
                                       vector<double> &filteredAffinesTx, vector<double> &filteredAffinesTy, vector<double> &filteredAffinesSkew, vector<double> &filteredAffinesRatio,
                                       vector<double> &unwantedAffinesA, vector<double> &unwantedAffinesB, vector<double> &unwantedAffinesC, vector<double> &unwantedAffinesD,
                                       vector<double> &unwantedAffinesTx, vector<double> &unwantedAffinesTy, vector<double> &unwantedAffinesSkew, vector<double> &unwantedAffinesRatio)
{
    float f = (freq*4)/30; // f = fc/fs (fc ?Hz, fs 30fps)
    vector<double> fAffinesA, fAffinesB, fAffinesC, fAffinesD, fAffinesTx, fAffinesTy, fAffinesSkew, fAffinesRatio;
    filter4Stages(f,affinesA,fAffinesA);
    filter4Stages(f,affinesB,fAffinesB);
    filter4Stages(f,affinesC,fAffinesC);
    filter4Stages(f,affinesD,fAffinesD);
    filter4Stages(f,affinesTx,fAffinesTx);
    filter4Stages(f,affinesTy,fAffinesTy);
    filter4Stages(f,affinesSkew,fAffinesSkew);
    filter4Stages(f,affinesRatio,fAffinesRatio);

    float d = (freq/4)*30; // d = fc*fs (fc ?Hz, fs 30fps)
    offsetDelay(d,fAffinesA,filteredAffinesA);
    offsetDelay(d,fAffinesB,filteredAffinesB);
    offsetDelay(d,fAffinesC,filteredAffinesC);
    offsetDelay(d,fAffinesD,filteredAffinesD);
    offsetDelay(d,fAffinesTx,filteredAffinesTx);
    offsetDelay(d,fAffinesTy,filteredAffinesTy);
    offsetDelay(d,fAffinesSkew,filteredAffinesSkew);
    offsetDelay(d,fAffinesRatio,filteredAffinesRatio);

    for(unsigned int i=0; i<affinesA.size(); ++i)
    {
        unwantedAffinesA.push_back(affinesA.at(i) - filteredAffinesA.at(i));
        unwantedAffinesB.push_back(affinesB.at(i) - filteredAffinesB.at(i));
        unwantedAffinesC.push_back(affinesC.at(i) - filteredAffinesC.at(i));
        unwantedAffinesD.push_back(affinesD.at(i) - filteredAffinesD.at(i));
        unwantedAffinesTx.push_back(affinesTx.at(i) - filteredAffinesTx.at(i));
        unwantedAffinesTy.push_back(affinesTy.at(i) - filteredAffinesTy.at(i));
        unwantedAffinesSkew.push_back(affinesSkew.at(i) - filteredAffinesSkew.at(i));
        unwantedAffinesRatio.push_back(affinesRatio.at(i) - filteredAffinesRatio.at(i));
    }
}

// compose transformation matrices of unwanted translations
void Stabilization::unwantedTransMotions(vector<double> unwantedTransX, vector<double> unwantedTransY, vector<Mat> &unwantedMotions)
{
    for(unsigned int i=0; i<unwantedTransX.size(); ++i)
    {
        Mat unwantedMotion = (Mat_<double>(3,3) << 1,0,unwantedTransX.at(i), 0,1,unwantedTransY.at(i), 0,0,1);
        unwantedMotions.push_back(unwantedMotion);
    }
}

// compose transformation matrices of unwanted affines transformations
void Stabilization::unwantedAffineMotions(vector<double> unwantedAffinesA, vector<double> unwantedAffinesB, vector<double> unwantedAffinesC, vector<double> unwantedAffinesD,
                                          vector<double> unwantedAffinesTx, vector<double> unwantedAffinesTy, vector<double> unwantedAffinesSkew, vector<double> unwantedAffinesRatio,
                                          vector<Mat> &unwantedMotions)
{
    for(unsigned int i=0; i<unwantedAffinesA.size(); ++i)
    {
        Mat unwantedMotion = (Mat_<double>(3,3) << 1,unwantedAffinesSkew.at(i),unwantedAffinesTx.at(i)/2,
                                                   0,1+unwantedAffinesRatio.at(i),unwantedAffinesTy.at(i)/2, 0,0,1);
        unwantedMotions.push_back(unwantedMotion);
    }
}

// compute 16:9 crop window size from maximum translation correction (max 10% > 96:54)
void Stabilization::cropSize(vector<double> unwantedTransX, vector<double> unwantedTransY, int &cropX, int &cropY)
{
    int maxX = ceil(max(*max_element(unwantedTransX.begin(),unwantedTransX.end()),-*min_element(unwantedTransX.begin(),unwantedTransX.end())));
    int maxY = ceil(max(*max_element(unwantedTransY.begin(),unwantedTransY.end()),-*min_element(unwantedTransY.begin(),unwantedTransY.end())));

    if(maxX>80 || maxY>45)
    {
        cropX = 96;
        cropY = 54;
    }
    else if(maxX>64 || maxY>36)
    {
        cropX = 80;
        cropY = 45;
    }
    else if(maxX>48 || maxY>27)
    {
        cropX = 64;
        cropY = 36;
    }
    else if(maxX>32 || maxY>18)
    {
        cropX = 48;
        cropY = 27;
    }
    else if(maxX>16 || maxY>9)
    {
        cropX = 32;
        cropY = 18;
    }
    else
    {
        cropX = 16;
        cropY = 9;
    }
}

// wrap optimal camera path to video and write stabilized video file
int Stabilization::stabilizeVideo(string folderName, string videoName, string stableVideoName, int nbFrames, vector<Mat> unwantedMotions, int cropX, int cropY)
{
    VideoCapture cap("/home/maelle/Desktop/Samples/"+folderName+"/"+videoName+".avi");  // open the original video

    if (!cap.isOpened())    // if not success, exit
    {
        cout << "Cannot open the video file!" << endl;
        return -1;
    }

    double fps = cap.get(CAP_PROP_FPS); // get framerate of the video
    Size frameSize(1920,1080);          // set frames size
    int fourcc = VideoWriter::fourcc('H','2','6','4');  // set codec code
    //cout << "Video parameters: " << frameSize << ", codec:" << fourcc << ", fps:" << fps << endl;

    VideoWriter videoWriter("/home/maelle/Desktop/Samples/"+folderName+"/"+stableVideoName+".avi",fourcc,fps,frameSize,true);  // initialize the VideoWriter object

    if (!videoWriter.isOpened())    // if not success, exit
    {
        cout << "Failed to write the video!" << endl;
        return -1;
    }

    Mat frame, stableFrame, warp;
    if(cap.read(frame))    // first frame
    {
        Mat croppedRef(frame,Rect(cropX,cropY,1920-2*cropX,1080-2*cropY));
        croppedRef.copyTo(frame);
        resize(frame,frame,frameSize,0,0,CV_INTER_CUBIC);
        videoWriter.write(frame);
    }
    for(int i=0; i<nbFrames-1; ++i)   // for each frame (except first)
    {
        if(!cap.read(frame))    // read frame
        {
            cout << "Failed to write the frame " << i+1 << endl;
            return -1;
        }
        cout << "Current frame: " << i+1 << endl;

        warp = unwantedMotions.at(i).inv();   // compute the inverse matrix
        //cout << warp << endl;

        warpPerspective(frame,stableFrame,warp,frameSize);   // apply transformation on the frame (with default options)

        Mat croppedRef(stableFrame,Rect(cropX,cropY,1920-2*cropX,1080-2*cropY));
        croppedRef.copyTo(stableFrame);
        resize(stableFrame,stableFrame,frameSize,0,0,CV_INTER_CUBIC);

        videoWriter.write(stableFrame);   // write the frame into the file

        namedWindow("Stabilized Video",WINDOW_NORMAL);
        imshow("Stabilized Video", stableFrame);
        if(waitKey((int)(1000/fps)) == 27) break;
    }

    return 0;
}

// [not working...] compute optimal camera path using linear programming
void Stabilization::optimalPathEstimation(int nbFrames, vector<Mat> globalMotions)
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
