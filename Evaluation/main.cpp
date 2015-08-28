#include <QApplication>
#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

#include <qwt_plot.h>
#include <qwt_plot_curve.h>
#include <qwt_legend.h>
#include <qwt_plot_marker.h>
#include <qwt_plot_renderer.h>

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

    string videoName = "static";
//
//*** Pixel variation error

//  original video pixel variation (RMSE) reading
tic();
    vector<double> RMSEO,framesO;
    double avrRMSEO;
    if(readPixelRMSE("/home/maelle/Desktop/Samples/Static/"+videoName+"/"+videoName+".avi",RMSEO,framesO,avrRMSEO)>0) return -1;
toc();

//  fusion stabilized video pixel variation (RMSE) reading
tic();
    vector<double> RMSEF,framesF;
    double avrRMSEF;
    if(readPixelRMSE("/home/maelle/Desktop/Samples/Static/"+videoName+"/"+videoName+"_stable_F.avi",RMSEF,framesF,avrRMSEF)>0) return -1;
toc();

//  video processing stabilized video pixel variation (RMSE) reading
tic();
    vector<double> RMSEV,framesV;
    double avrRMSEV;
    if(readPixelRMSE("/home/maelle/Desktop/Samples/Static/"+videoName+"/"+videoName+"_stable_VP.avi",RMSEV,framesV,avrRMSEV)>0) return -1;
toc();
//  second round fusion stabilized video pixel variation (RMSE) reading
tic();
    vector<double> RMSEF2,framesF2;
    double avrRMSEF2;
    if(readPixelRMSE("/home/maelle/Desktop/Samples/Static/"+videoName+"/"+videoName+"_stable2_F.avi",RMSEF2,framesF2,avrRMSEF2)>0) return -1;
toc();

//  second round video processing stabilized video pixel variation (RMSE) reading
tic();
    vector<double> RMSEV2,framesV2;
    double avrRMSEV2;
    if(readPixelRMSE("/home/maelle/Desktop/Samples/Static/"+videoName+"/"+videoName+"_stable2_VP.avi",RMSEV2,framesV2,avrRMSEV2)>0) return -1;
toc();

//  plot first round
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

//  plot second round
    QwtPlot plotRMSE2;
    plotRMSE2.setTitle("Root Mean Squared pixel variation per frame (second round)");
    plotRMSE2.setCanvasBackground(Qt::white);
    plotRMSE2.insertLegend(new QwtLegend());
    plotRMSE2.setAxisTitle(QwtPlot::yLeft,"RMSE (px)");
    plotRMSE2.setAxisTitle(QwtPlot::xBottom,"Frame");
    QwtPlotMarker *mAO2=new QwtPlotMarker();
        mAO2->setLinePen(QPen(Qt::darkBlue));
        mAO2->setLineStyle(QwtPlotMarker::HLine);
        mAO2->setValue(0,avrRMSEO);
        mAO2->attach(&plotRMSE2);
    QwtPlotMarker *mAOF=new QwtPlotMarker();
        mAOF->setLinePen(QPen(Qt::darkCyan));
        mAOF->setLineStyle(QwtPlotMarker::HLine);
        mAOF->setValue(0,avrRMSEF);
        mAOF->attach(&plotRMSE2);
    QwtPlotMarker *mAF2=new QwtPlotMarker();
        mAF2->setLinePen(QPen(Qt::darkRed));
        mAF2->setLineStyle(QwtPlotMarker::HLine);
        mAF2->setValue(0,avrRMSEF2);
        mAF2->attach(&plotRMSE2);
    QwtPlotMarker *mAV2=new QwtPlotMarker();
        mAV2->setLinePen(QPen(Qt::darkGreen));
        mAV2->setLineStyle(QwtPlotMarker::HLine);
        mAV2->setValue(0,avrRMSEV2);
        mAV2->attach(&plotRMSE2);
    QwtPlotCurve *curveRMSEO2 = new QwtPlotCurve();
        curveRMSEO2->setTitle("Original");
        curveRMSEO2->setPen(Qt::blue,2);
        curveRMSEO2->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveRMSEO2->setRawSamples(framesO.data(),RMSEO.data(),framesO.size());
        curveRMSEO2->attach(&plotRMSE2);
    QwtPlotCurve *curveRMSEOF = new QwtPlotCurve();
        curveRMSEOF->setTitle("First round fusion stabilized");
        curveRMSEOF->setPen(Qt::cyan,2);
        curveRMSEOF->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveRMSEOF->setRawSamples(framesF.data(),RMSEF.data(),framesF.size());
        curveRMSEOF->attach(&plotRMSE2);
    QwtPlotCurve *curveRMSEF2 = new QwtPlotCurve();
        curveRMSEF2->setTitle("Fusion stabilized");
        curveRMSEF2->setPen(Qt::red,2);
        curveRMSEF2->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveRMSEF2->setRawSamples(framesF2.data(),RMSEF2.data(),framesF2.size());
        curveRMSEF2->attach(&plotRMSE2);
    QwtPlotCurve *curveRMSEV2 = new QwtPlotCurve();
        curveRMSEV2->setTitle("Video Processing stabilized");
        curveRMSEV2->setPen(Qt::green,2);
        curveRMSEV2->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveRMSEV2->setRawSamples(framesV2.data(),RMSEV2.data(),framesV2.size());
        curveRMSEV2->attach(&plotRMSE2);
    plotRMSE2.resize(600,400);
    plotRMSE2.show();

//
//*** Flow

//  original video motion flows reading
tic();
    vector<double> xflowsO,yflowsO,framesFlowO;
    double avrxO=0,avryO=0;
    if(readFlows("/home/maelle/Desktop/Samples/Static/"+videoName+"/"+videoName+".avi",xflowsO,yflowsO,framesFlowO,avrxO,avryO,true)>0) return -1;
toc();

//  fusion stabilized video motion flows reading
tic();
    vector<double> xflowsF,yflowsF,framesFlowF;
    double avrxF=0,avryF=0;
    if(readFlows("/home/maelle/Desktop/Samples/Static/"+videoName+"/"+videoName+"_stable_F.avi",xflowsF,yflowsF,framesFlowF,avrxF,avryF,true)>0) return -1;
toc();

//  video processing stabilized video motion flows reading
tic();
    vector<double> xflowsV,yflowsV,framesFlowV;
    double avrxV=0,avryV=0;
    if(readFlows("/home/maelle/Desktop/Samples/Static/"+videoName+"/"+videoName+"_stable_VP.avi",xflowsV,yflowsV,framesFlowV,avrxV,avryV,true)>0) return -1;
toc();

//  second round fusion stabilized video motion flows reading
tic();
    vector<double> xflowsF2,yflowsF2,framesFlowF2;
    double avrxF2=0,avryF2=0;
    if(readFlows("/home/maelle/Desktop/Samples/Static/"+videoName+"/"+videoName+"_stable2_F.avi",xflowsF2,yflowsF2,framesFlowF2,avrxF2,avryF2,true)>0) return -1;
toc();

//  second round video processing stabilized video motion flows reading
tic();
    vector<double> xflowsV2,yflowsV2,framesFlowV2;
    double avrxV2=0,avryV2=0;
    if(readFlows("/home/maelle/Desktop/Samples/Static/"+videoName+"/"+videoName+"_stable2_VP.avi",xflowsV2,yflowsV2,framesFlowV2,avrxV2,avryV2,true)>0) return -1;
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

    QwtPlot plotxFlow2;
    plotxFlow2.setTitle("Average horizontal flow per frame (second round)");
    plotxFlow2.setCanvasBackground(Qt::white);
    plotxFlow2.insertLegend(new QwtLegend());
    plotxFlow2.setAxisTitle(QwtPlot::yLeft,"Flow (px)");
    plotxFlow2.setAxisTitle(QwtPlot::xBottom,"Frame");
    QwtPlotMarker *mxAO2=new QwtPlotMarker();
        mxAO2->setLinePen(QPen(Qt::darkBlue));
        mxAO2->setLineStyle(QwtPlotMarker::HLine);
        mxAO2->setValue(0,avrxO);
        mxAO2->attach(&plotxFlow2);
    QwtPlotMarker *mxAF0=new QwtPlotMarker();
        mxAF0->setLinePen(QPen(Qt::darkCyan));
        mxAF0->setLineStyle(QwtPlotMarker::HLine);
        mxAF0->setValue(0,avrxF);
        mxAF0->attach(&plotxFlow2);
    QwtPlotMarker *mxAF2=new QwtPlotMarker();
        mxAF2->setLinePen(QPen(Qt::darkRed));
        mxAF2->setLineStyle(QwtPlotMarker::HLine);
        mxAF2->setValue(0,avrxF2);
        mxAF2->attach(&plotxFlow2);
    QwtPlotMarker *mxAV2=new QwtPlotMarker();
        mxAV2->setLinePen(QPen(Qt::darkGreen));
        mxAV2->setLineStyle(QwtPlotMarker::HLine);
        mxAV2->setValue(0,avrxV2);
        mxAV2->attach(&plotxFlow2);
    QwtPlotCurve *curvexFlowO2 = new QwtPlotCurve();
        curvexFlowO2->setTitle("Original");
        curvexFlowO2->setPen(Qt::blue,2);
        curvexFlowO2->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curvexFlowO2->setRawSamples(framesFlowO.data(),xflowsO.data(),framesFlowO.size());
        curvexFlowO2->attach(&plotxFlow2);
    QwtPlotCurve *curvexFlowFO = new QwtPlotCurve();
        curvexFlowFO->setTitle("First round fusion stabilized");
        curvexFlowFO->setPen(Qt::cyan,2);
        curvexFlowFO->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curvexFlowFO->setRawSamples(framesFlowF.data(),xflowsF.data(),framesFlowF.size());
        curvexFlowFO->attach(&plotxFlow2);
    QwtPlotCurve *curvexFlowF2 = new QwtPlotCurve();
        curvexFlowF2->setTitle("Fusion stabilized");
        curvexFlowF2->setPen(Qt::red,2);
        curvexFlowF2->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curvexFlowF2->setRawSamples(framesFlowF2.data(),xflowsF2.data(),framesFlowF2.size());
        curvexFlowF2->attach(&plotxFlow2);
    QwtPlotCurve *curvexFlowV2 = new QwtPlotCurve();
        curvexFlowV2->setTitle("Video Processing stabilized");
        curvexFlowV2->setPen(Qt::green,2);
        curvexFlowV2->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curvexFlowV2->setRawSamples(framesFlowV2.data(),xflowsV2.data(),framesFlowV2.size());
        curvexFlowV2->attach(&plotxFlow2);
    plotxFlow2.resize(600,400);
    plotxFlow2.show();

    QwtPlot plotyFlow;
    plotyFlow.setTitle("Average vertical flow per frame");
    plotyFlow.setCanvasBackground(Qt::white);
    plotyFlow.insertLegend(new QwtLegend());
    plotyFlow.setAxisTitle(QwtPlot::yLeft,"Flow (px)");
    plotyFlow.setAxisTitle(QwtPlot::xBottom,"Frame");
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

    QwtPlot plotyFlow2;
    plotyFlow2.setTitle("Average vertical flow per frame (second round)");
    plotyFlow2.setCanvasBackground(Qt::white);
    plotyFlow2.insertLegend(new QwtLegend());
    plotyFlow2.setAxisTitle(QwtPlot::yLeft,"Flow (px)");
    plotyFlow2.setAxisTitle(QwtPlot::xBottom,"Frame");
    QwtPlotMarker *myAO2=new QwtPlotMarker();
        myAO2->setLinePen(QPen(Qt::darkBlue));
        myAO2->setLineStyle(QwtPlotMarker::HLine);
        myAO2->setValue(0,avryO);
        myAO2->attach(&plotyFlow2);
    QwtPlotMarker *myAF0=new QwtPlotMarker();
        myAF0->setLinePen(QPen(Qt::darkCyan));
        myAF0->setLineStyle(QwtPlotMarker::HLine);
        myAF0->setValue(0,avryF);
        myAF0->attach(&plotyFlow2);
    QwtPlotMarker *myAF2=new QwtPlotMarker();
        myAF2->setLinePen(QPen(Qt::darkRed));
        myAF2->setLineStyle(QwtPlotMarker::HLine);
        myAF2->setValue(0,avryF2);
        myAF2->attach(&plotyFlow2);
    QwtPlotMarker *myAV2=new QwtPlotMarker();
        myAV2->setLinePen(QPen(Qt::darkGreen));
        myAV2->setLineStyle(QwtPlotMarker::HLine);
        myAV2->setValue(0,avryV2);
        myAV2->attach(&plotyFlow2);
    QwtPlotCurve *curveyFlowO2 = new QwtPlotCurve();
        curveyFlowO2->setTitle("Original");
        curveyFlowO2->setPen(Qt::blue,2);
        curveyFlowO2->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveyFlowO2->setRawSamples(framesFlowO.data(),yflowsO.data(),framesFlowO.size());
        curveyFlowO2->attach(&plotyFlow2);
    QwtPlotCurve *curveyFlowFO = new QwtPlotCurve();
        curveyFlowFO->setTitle("First round fusion stabilized");
        curveyFlowFO->setPen(Qt::cyan,2);
        curveyFlowFO->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveyFlowFO->setRawSamples(framesFlowF.data(),yflowsF.data(),framesFlowF.size());
        curveyFlowFO->attach(&plotyFlow2);
    QwtPlotCurve *curveyFlowF2 = new QwtPlotCurve();
        curveyFlowF2->setTitle("Fusion stabilized");
        curveyFlowF2->setPen(Qt::red,2);
        curveyFlowF2->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveyFlowF2->setRawSamples(framesFlowF2.data(),yflowsF2.data(),framesFlowF2.size());
        curveyFlowF2->attach(&plotyFlow2);
    QwtPlotCurve *curveyFlowV2 = new QwtPlotCurve();
        curveyFlowV2->setTitle("Video Processing stabilized");
        curveyFlowV2->setPen(Qt::green,2);
        curveyFlowV2->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveyFlowV2->setRawSamples(framesFlowV2.data(),yflowsV2.data(),framesFlowV2.size());
        curveyFlowV2->attach(&plotyFlow2);
    plotyFlow2.resize(600,400);
    plotyFlow2.show();
//
    return a.exec();
}
