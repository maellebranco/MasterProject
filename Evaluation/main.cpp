#include <QApplication>
#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

#include <qwt_plot.h>
#include <qwt_plot_curve.h>
#include <qwt_legend.h>
#include <qwt_plot_marker.h>

#include <stack>
#include <ctime>

using namespace std;
using namespace cv;

// for timing
stack<clock_t> tictoc_stack;
void tic() { tictoc_stack.push(clock()); }
void toc() {
    cout << "Time elapsed: " << ((double)(clock()-tictoc_stack.top()))/CLOCKS_PER_SEC << " seconds" << endl;
    tictoc_stack.pop();
}

static int readPixelRMSE(string videoPath,vector<double> &RMSE,vector<double> &frames,double &avrRMSE)
{
    VideoCapture cap(videoPath);
    if(!cap.isOpened()) return 1;

    Mat next,prev;

    if(!cap.read(prev)) return 1;
    cvtColor(prev,prev,COLOR_BGR2GRAY);

    for(int i=1;;)
    {
        if(!cap.read(next)) break;
        cout << "Current frame: " << i << endl;

        cvtColor(next,next,COLOR_BGR2GRAY);

        double squaredError=0;
        for(int y=0; y<next.rows; ++y)
            for(int x=0; x<next.cols; ++x)
            {
                double pxPrev = prev.at<uchar>(y,x);
                double pxNext = next.at<uchar>(y,x);
                squaredError += (pxPrev-pxNext)*(pxPrev-pxNext);
            }
        double rmse = sqrt(squaredError/(next.rows*next.cols));
        RMSE.push_back(rmse);
        frames.push_back(i++);
        avrRMSE += rmse;

        swap(prev,next);
    }
    avrRMSE /= RMSE.size();
    cout << "Average Root Mean Squared Error: " << avrRMSE << endl;

    return 0;
}

static int readFlows(string videoPath,vector<double> &xflows,vector<double> &yflows,vector<double> &frames,double &avrx,double &avry,bool visualization=true)
{
    VideoCapture cap(videoPath);
    if(!cap.isOpened()) return 1;

    Mat flow,cflow,next,prev;

    if(!cap.read(prev)) return 1;
    cvtColor(prev,prev,COLOR_BGR2GRAY);

    for(int i=1;;)
    {
        if(!cap.read(next)) break;
        cout << "Current frame: " << i << endl;

        cvtColor(next,next,COLOR_BGR2GRAY);
        calcOpticalFlowFarneback(prev,next,flow,0.5,3,15,3,5,1.1,0);

        if(visualization)
        {
            cvtColor(prev,cflow,COLOR_GRAY2BGR);
            for(int y=0; y<cflow.rows; y+=20)
                for(int x=0; x<cflow.cols; x+=20)
                {
                    Point2f fxy = flow.at<Point2f>(y,x);
                    line(cflow,Point(x,y),Point(cvRound(x+fxy.x),cvRound(y+fxy.y)),Scalar(0,255,0));
                    circle(cflow,Point(x,y),2,Scalar(0,255,0),-1);
                }
            namedWindow("flow",WINDOW_NORMAL);
            imshow("flow",cflow);
            if(waitKey(1)==27) break;
        }

        double xflow=0, yflow=0;
        for(int y=0; y<flow.rows; ++y)
            for(int x=0; x<flow.cols; ++x)
            {
                Point2f pt = flow.at<Point2f>(y,x);
                xflow += pt.x;
                yflow += pt.y;
            }
        double nb = flow.rows*flow.cols;
        xflows.push_back((xflow/nb)*(xflow/nb));
        yflows.push_back((yflow/nb)*(yflow/nb));
        frames.push_back(i++);
        avrx += xflows.back();
        avry += yflows.back();

        swap(prev,next);
    }
    avrx /= xflows.size();
    avry /= yflows.size();
    cout << "Mean Squared Average Flow: " << avrx << " " << avry << endl;

    return 0;
}

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    string videoName = "test";

//*** Pixel variation error

//  original video pixel variation (RMSE) reading
tic();
    vector<double> RMSEO,framesO;
    double avrRMSEO;
    if(readPixelRMSE("/home/maelle/Desktop/Samples/"+videoName+"/"+videoName+".avi",RMSEO,framesO,avrRMSEO)>0) return -1;
toc();

//  fusion stabilized video pixel variation (RMSE) reading
tic();
    vector<double> RMSEF,framesF;
    double avrRMSEF;
    if(readPixelRMSE("/home/maelle/Desktop/Samples/"+videoName+"/"+videoName+"_stable_F.avi",RMSEF,framesF,avrRMSEF)>0) return -1;
toc();

//  video processing stabilized video pixel variation (RMSE) reading
tic();
    vector<double> RMSEV,framesV;
    double avrRMSEV;
    if(readPixelRMSE("/home/maelle/Desktop/Samples/"+videoName+"/"+videoName+"_stable_VP.avi",RMSEV,framesV,avrRMSEV)>0) return -1;
toc();

//  plot
    QwtPlot plotRMSE;
    plotRMSE.setTitle("Root Mean Squared pixel variation per frame");
    plotRMSE.setCanvasBackground(Qt::white);
    plotRMSE.insertLegend(new QwtLegend());
    plotRMSE.setAxisTitle(QwtPlot::yLeft,"RMSE (px)");
    plotRMSE.setAxisTitle(QwtPlot::xBottom,"Frame");
    QwtPlotMarker *mAO=new QwtPlotMarker();
        mAO->setLinePen(QPen(Qt::darkBlue));
        mAO->setLineStyle(QwtPlotMarker::HLine);
        mAO->setValue(0,avrRMSEO);
        mAO->attach(&plotRMSE);
    QwtPlotMarker *mAF=new QwtPlotMarker();
        mAF->setLinePen(QPen(Qt::darkRed));
        mAF->setLineStyle(QwtPlotMarker::HLine);
        mAF->setValue(0,avrRMSEF);
        mAF->attach(&plotRMSE);
    QwtPlotMarker *mAV=new QwtPlotMarker();
        mAV->setLinePen(QPen(Qt::darkGreen));
        mAV->setLineStyle(QwtPlotMarker::HLine);
        mAV->setValue(0,avrRMSEV);
        mAV->attach(&plotRMSE);
    QwtPlotCurve *curveRMSEO = new QwtPlotCurve();
        curveRMSEO->setTitle("Original");
        curveRMSEO->setPen(Qt::blue,2);
        curveRMSEO->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveRMSEO->setRawSamples(framesO.data(),RMSEO.data(),framesO.size());
        curveRMSEO->attach(&plotRMSE);
    QwtPlotCurve *curveRMSEF = new QwtPlotCurve();
        curveRMSEF->setTitle("Fusion stabilized");
        curveRMSEF->setPen(Qt::red,2);
        curveRMSEF->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveRMSEF->setRawSamples(framesF.data(),RMSEF.data(),framesF.size());
        curveRMSEF->attach(&plotRMSE);
    QwtPlotCurve *curveRMSEV = new QwtPlotCurve();
        curveRMSEV->setTitle("Video Processing stabilized");
        curveRMSEV->setPen(Qt::green,2);
        curveRMSEV->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveRMSEV->setRawSamples(framesV.data(),RMSEV.data(),framesV.size());
        curveRMSEV->attach(&plotRMSE);
    plotRMSE.resize(600,400);
    plotRMSE.show();

//*** Flow
/*
//  original video motion flows reading
tic();
    vector<double> xflowsO,yflowsO,framesFlowO;
    double avrxO=0,avryO=0;
    if(readFlows("/home/maelle/Desktop/Samples/"+videoName+"/"+videoName+".avi",xflowsO,yflowsO,framesFlowO,avrxO,avryO,true)>0) return -1;
toc();

//  fusion stabilized video motion flows reading
tic();
    vector<double> xflowsF,yflowsF,framesFlowF;
    double avrxF=0,avryF=0;
    if(readFlows("/home/maelle/Desktop/Samples/"+videoName+"/"+videoName+"_stable_F.avi",xflowsF,yflowsF,framesFlowF,avrxF,avryF,true)>0) return -1;
toc();

//  video processing stabilized video motion flows reading
tic();
    vector<double> xflowsV,yflowsV,framesFlowV;
    double avrxV=0,avryV=0;
    if(readFlows("/home/maelle/Desktop/Samples/"+videoName+"/"+videoName+"_stable_VP.avi",xflowsV,yflowsV,framesFlowV,avrxV,avryV,true)>0) return -1;
toc();

//  plots
    QwtPlot plotxFlow;
    plotxFlow.setTitle("Average horizontal flow per frame");
    plotxFlow.setCanvasBackground(Qt::white);
    plotxFlow.insertLegend(new QwtLegend());
    plotxFlow.setAxisTitle(QwtPlot::yLeft,"Flow (px)");
    plotxFlow.setAxisTitle(QwtPlot::xBottom,"Frame");
    QwtPlotMarker *mxAO=new QwtPlotMarker();
        mxAO->setLinePen(QPen(Qt::darkBlue));
        mxAO->setLineStyle(QwtPlotMarker::HLine);
        mxAO->setValue(0,avrxO);
        mxAO->attach(&plotxFlow);
    QwtPlotMarker *mxAF=new QwtPlotMarker();
        mxAF->setLinePen(QPen(Qt::darkRed));
        mxAF->setLineStyle(QwtPlotMarker::HLine);
        mxAF->setValue(0,avrxF);
        mxAF->attach(&plotxFlow);
    QwtPlotMarker *mxAV=new QwtPlotMarker();
        mxAV->setLinePen(QPen(Qt::darkGreen));
        mxAV->setLineStyle(QwtPlotMarker::HLine);
        mxAV->setValue(0,avrxV);
        mxAV->attach(&plotxFlow);
    QwtPlotCurve *curvexFlowO = new QwtPlotCurve();
        curvexFlowO->setTitle("Original");
        curvexFlowO->setPen(Qt::blue,2);
        curvexFlowO->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curvexFlowO->setRawSamples(framesFlowO.data(),xflowsO.data(),framesFlowO.size());
        curvexFlowO->attach(&plotxFlow);
    QwtPlotCurve *curvexFlowF = new QwtPlotCurve();
        curvexFlowF->setTitle("Fusion stabilized");
        curvexFlowF->setPen(Qt::red,2);
        curvexFlowF->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curvexFlowF->setRawSamples(framesFlowF.data(),xflowsF.data(),framesFlowF.size());
        curvexFlowF->attach(&plotxFlow);
    QwtPlotCurve *curvexFlowV = new QwtPlotCurve();
        curvexFlowV->setTitle("Video Processing stabilized");
        curvexFlowV->setPen(Qt::green,2);
        curvexFlowV->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curvexFlowV->setRawSamples(framesFlowV.data(),xflowsV.data(),framesFlowV.size());
        curvexFlowV->attach(&plotxFlow);
    plotxFlow.resize(600,400);
    plotxFlow.show();

    QwtPlot plotyFlow;
    plotyFlow.setTitle("Average vertical flow per frame");
    plotyFlow.setCanvasBackground(Qt::white);
    plotyFlow.insertLegend(new QwtLegend());
    plotyFlow.setAxisTitle(QwtPlot::yLeft,"Flow (px)");
    plotyFlow.setAxisTitle(QwtPlot::xBottom,"Frame");
    QwtPlotMarker *myF=new QwtPlotMarker();
        myF->setLineStyle(QwtPlotMarker::HLine);
        myF->setValue(0,0);
        myF->attach(&plotyFlow);
    QwtPlotMarker *myAO=new QwtPlotMarker();
        myAO->setLinePen(QPen(Qt::darkBlue));
        myAO->setLineStyle(QwtPlotMarker::HLine);
        myAO->setValue(0,avryO);
        myAO->attach(&plotyFlow);
    QwtPlotMarker *myAF=new QwtPlotMarker();
        myAF->setLinePen(QPen(Qt::darkRed));
        myAF->setLineStyle(QwtPlotMarker::HLine);
        myAF->setValue(0,avryF);
        myAF->attach(&plotyFlow);
    QwtPlotMarker *myAV=new QwtPlotMarker();
        myAV->setLinePen(QPen(Qt::darkGreen));
        myAV->setLineStyle(QwtPlotMarker::HLine);
        myAV->setValue(0,avryV);
        myAV->attach(&plotyFlow);
    QwtPlotCurve *curveyFlowO = new QwtPlotCurve();
        curveyFlowO->setTitle("Original");
        curveyFlowO->setPen(Qt::blue,2);
        curveyFlowO->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveyFlowO->setRawSamples(framesFlowO.data(),yflowsO.data(),framesFlowO.size());
        curveyFlowO->attach(&plotyFlow);
    QwtPlotCurve *curveyFlowF = new QwtPlotCurve();
        curveyFlowF->setTitle("Fusion stabilized");
        curveyFlowF->setPen(Qt::red,2);
        curveyFlowF->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveyFlowF->setRawSamples(framesFlowF.data(),yflowsF.data(),framesFlowF.size());
        curveyFlowF->attach(&plotyFlow);
    QwtPlotCurve *curveyFlowV = new QwtPlotCurve();
        curveyFlowV->setTitle("Video Processing stabilized");
        curveyFlowV->setPen(Qt::green,2);
        curveyFlowV->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveyFlowV->setRawSamples(framesFlowV.data(),yflowsV.data(),framesFlowV.size());
        curveyFlowV->attach(&plotyFlow);
    plotyFlow.resize(600,400);
    plotyFlow.show();
*/
    return a.exec();
}
