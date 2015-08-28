#include "trajectory.h"
#include "videomotionestimation.h"
#include "sensormotionestimation.h"
#include "stabilization.h"
#include "fusion.h"

#include <QApplication>
#include <iostream>
#include <string.h>
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

#include <libxml2/libxml/xmlreader.h>

#include <qwt_plot.h>
#include <qwt_plot_curve.h>
#include <qwt_legend.h>
#include <qwt_plot_marker.h>
#include <qwt_symbol.h>

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

    string videoName = "static";

    //VideoMotionEstimation::readingVideo("/home/maelle/Desktop/Samples/Static/"+videoName+"/"+videoName+".avi");

//*** First round

    // Read video frames and compute trajectories
tic();
    vector<Trajectory> trajectories;
    int nbFrames;

    VideoMotionEstimation::calcTrajectoriesKLT(trajectories,nbFrames,"/home/maelle/Desktop/Samples/Static/"+videoName+"/"+videoName+".avi",2000,500);
    //VideoMotionEstimation::calcTrajectoriesSIFT(trajectories,nbFrames,"/home/maelle/Desktop/Samples/Static/"+videoName+"/"+videoName+".avi");
toc();

    // Compute global motions (only translations) between each pair of frames
tic();
    vector<Mat> translations;

    VideoMotionEstimation::calcGlobalMotions(translations,videostab::MM_TRANSLATION,trajectories,nbFrames,true,true);
toc();

    // Read video timestamps from xml file
tic();
    vector<long int> videoTimestamps;
    string videoFilePath = "/home/maelle/Desktop/Samples/Static/"+videoName+"/video.txt";

    if(Fusion::videoTimestamps(videoFilePath.c_str(),videoTimestamps)<0)
        return -1;

    // Convert the global motions matrices in vectors (and scale the timestamps for plots)
    vector<double> scaledVideoTimestamps,translationsX,translationsY;
    VideoMotionEstimation::convertMatrixData(nbFrames,videoTimestamps,translations,scaledVideoTimestamps,translationsX,translationsY);
toc();
    // Read the sensor data from xml file
tic();
    string filePath = "/home/maelle/Desktop/Samples/Static/"+videoName+"/sensor.txt";
    vector<long int> timestamps;
    vector<float> temperatures;
    float sensitivityAcc, sensitivityGyro;
    vector<int> rawAccelerationsX, rawAccelerationsY, rawAccelerationsZ;
    vector<int> rawAngularRatesX, rawAngularRatesY, rawAngularRatesZ;

    if(SensorMotionEstimation::readSensorData(filePath.c_str(),timestamps,temperatures,sensitivityAcc,sensitivityGyro,
                                              rawAccelerationsX,rawAccelerationsY,rawAccelerationsZ,rawAngularRatesX,rawAngularRatesY,rawAngularRatesZ)<0)
        return -1;
toc();
    // Convert the sensor raw data into usable data vectors
tic();
    vector<double> scaledTimestamps;
    vector<double> accelerationsX, accelerationsY, accelerationsZ;
    vector<double> angularRatesX, angularRatesY, angularRatesZ;

    SensorMotionEstimation::convertRawData(timestamps,sensitivityAcc,sensitivityGyro,
                                           rawAccelerationsX,rawAccelerationsY,rawAccelerationsZ,rawAngularRatesX,rawAngularRatesY,rawAngularRatesZ,
                                           scaledTimestamps,accelerationsX,accelerationsY,accelerationsZ,angularRatesX,angularRatesY,angularRatesZ);

    // Compute roll, pitch and yaw data
    vector<double> roll,pitch,yaw;
    vector<double> rollGyro,pitchGyro;
    vector<double> rollAcc,pitchAcc;

    SensorMotionEstimation::computeRollPitchYaw(scaledTimestamps,accelerationsX,accelerationsY,accelerationsZ,angularRatesX,angularRatesY,angularRatesZ,
                                                roll,pitch,yaw,rollGyro,pitchGyro,rollAcc,pitchAcc);
toc();

    // Compute the synchronization between sensor data and video frames
tic();
    //Fusion::checkSynchro(timestamps,videoTimestamps);
    map<int,int> synchroMap;
    Fusion::buildSynchroMap(timestamps,videoTimestamps,synchroMap);

    vector<double> syncAngularRatesX, syncAngularRatesY, syncAngularRatesZ;
    Fusion::fusionSensorData(synchroMap,angularRatesX,angularRatesY,angularRatesZ,syncAngularRatesX,syncAngularRatesY,syncAngularRatesZ);

    // Correct the video processing global motions using the sensor data
    vector<double> corr1TranslationsX,corr1TranslationsY;
    Fusion::noMotionDetection(translationsX,translationsY,syncAngularRatesX,syncAngularRatesY,syncAngularRatesZ,
                              corr1TranslationsX,corr1TranslationsY);
    double mseH=0, mseV=0;
    vector<double> errorsH,errorsV;
    Fusion::errorMotions(corr1TranslationsX,corr1TranslationsY,syncAngularRatesY,syncAngularRatesZ,errorsH,errorsV,mseH,mseV);

    vector<double> corrTranslationsX,corrTranslationsY;
    Fusion::correctionMotions(corr1TranslationsX,corr1TranslationsY,errorsH,errorsV,mseH,mseV,corrTranslationsX,corrTranslationsY);
toc();

    // Compute the unwanted global motions using low pass filtering
tic();
    vector<double> filteredTransX,filteredTransY;
    vector<double> unwantedTransX,unwantedTransY;

    Stabilization::calcUnwantedMotion(0.45,translationsX,translationsY,filteredTransX,filteredTransY,unwantedTransX,unwantedTransY);

    // Compute the unwanted global motions using low pass filtering for fusion data
    vector<double> corrFilteredTransX,corrFilteredTransY;
    vector<double> corrUnwantedTransX,corrUnwantedTransY;

    Stabilization::calcUnwantedMotion(0.45,corrTranslationsX,corrTranslationsY,corrFilteredTransX,corrFilteredTransY,
                                      corrUnwantedTransX,corrUnwantedTransY);
toc();

    // Compute the transformation matrices for stabilization
tic();
    vector<Mat> unwantedMotions;
    Stabilization::unwantedTransMotions(unwantedTransX,unwantedTransY,unwantedMotions);

    // Compute the size of the crop window
    int cropX=0, cropY=0;
    Stabilization::cropSize(unwantedTransX,unwantedTransY,cropX,cropY);
    cout << "Crop size: " << cropX << ":" << cropY << endl;

    // Warp the transformation matrices to each frame to stabilize the video
    Stabilization::stabilizeVideo(videoName,videoName,videoName+"_stable_VP",nbFrames,unwantedMotions,cropX,cropY);
toc();

tic();
    // Compute the transformation matrices for stabilization using fusion data
    vector<Mat> corrUnwantedMotions;
    Stabilization::unwantedTransMotions(corrUnwantedTransX,corrUnwantedTransY,corrUnwantedMotions);

    // Compute the size of the crop window
    cropX=0; cropY=0;
    Stabilization::cropSize(corrUnwantedTransX,corrUnwantedTransY,cropX,cropY);
    cout << "Crop size: " << cropX << ":" << cropY << endl;

    // Warp the transformation matrices to each frame to stabilize the video
    Stabilization::stabilizeVideo(videoName,videoName,videoName+"_stable_F",nbFrames,corrUnwantedMotions,cropX,cropY);
toc();

    // Plots
//
    QwtPlot plotAccelerations;
    plotAccelerations.setTitle("Accelerations");
    plotAccelerations.setCanvasBackground(Qt::white);
    plotAccelerations.insertLegend(new QwtLegend());
    plotAccelerations.setAxisTitle(QwtPlot::yLeft,"Acceleration (g)");
    plotAccelerations.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotCurve *curveAccX = new QwtPlotCurve();
        curveAccX->setTitle("X");
        curveAccX->setPen(Qt::blue,2);
        curveAccX->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveAccX->setRawSamples(scaledTimestamps.data(),accelerationsX.data(),scaledTimestamps.size());
        curveAccX->attach(&plotAccelerations);
    QwtPlotCurve *curveAccY = new QwtPlotCurve();
        curveAccY->setTitle("Y");
        curveAccY->setPen(Qt::red,2);
        curveAccY->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveAccY->setRawSamples(scaledTimestamps.data(),accelerationsY.data(),scaledTimestamps.size());
        curveAccY->attach(&plotAccelerations);
    QwtPlotCurve *curveAccZ = new QwtPlotCurve();
        curveAccZ->setTitle("Z");
        curveAccZ->setPen(Qt::green,2);
        curveAccZ->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveAccZ->setRawSamples(scaledTimestamps.data(),accelerationsZ.data(),scaledTimestamps.size());
        curveAccZ->attach(&plotAccelerations);
    plotAccelerations.resize(600,400);
    plotAccelerations.show();

    QwtPlot plotAngularRates;
    plotAngularRates.setTitle("Angular Rates");
    plotAngularRates.setCanvasBackground(Qt::white);
    plotAngularRates.insertLegend(new QwtLegend());
    plotAngularRates.setAxisTitle(QwtPlot::yLeft,"Angular Rate (°/s)");
    plotAngularRates.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotMarker *m1=new QwtPlotMarker();
        m1->setLineStyle(QwtPlotMarker::HLine);
        m1->setValue(0,0.5);
        m1->attach(&plotAngularRates);
    QwtPlotMarker *m2=new QwtPlotMarker();
        m2->setLineStyle(QwtPlotMarker::HLine);
        m2->setValue(0,-0.5);
        m2->attach(&plotAngularRates);
    QwtPlotCurve *curveAngX = new QwtPlotCurve();
        curveAngX->setTitle("X");
        curveAngX->setPen(Qt::blue,2);
        curveAngX->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveAngX->setRawSamples(scaledTimestamps.data(),angularRatesX.data(),scaledTimestamps.size());
        curveAngX->attach(&plotAngularRates);
    QwtPlotCurve *curveAngY = new QwtPlotCurve();
        curveAngY->setTitle("Y");
        curveAngY->setPen(Qt::red,2);
        curveAngY->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveAngY->setRawSamples(scaledTimestamps.data(),angularRatesY.data(),scaledTimestamps.size());
        curveAngY->attach(&plotAngularRates);
    QwtPlotCurve *curveAngZ = new QwtPlotCurve();
        curveAngZ->setTitle("Z");
        curveAngZ->setPen(Qt::green,2);
        curveAngZ->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveAngZ->setRawSamples(scaledTimestamps.data(),angularRatesZ.data(),scaledTimestamps.size());
        curveAngZ->attach(&plotAngularRates);
    plotAngularRates.resize(600,400);
    plotAngularRates.show();

    QwtPlot plotRoll;
    plotRoll.setTitle("Roll");
    plotRoll.setCanvasBackground(Qt::white);
    plotRoll.insertLegend(new QwtLegend());
    plotRoll.setAxisTitle(QwtPlot::yLeft,"Roll (°)");
    plotRoll.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotCurve *curveRollAcc = new QwtPlotCurve();
        curveRollAcc->setTitle("Accelerometer");
        curveRollAcc->setPen(Qt::blue,2);
        curveRollAcc->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveRollAcc->setRawSamples(scaledTimestamps.data(),rollAcc.data(),scaledTimestamps.size());
        curveRollAcc->attach(&plotRoll);
    QwtPlotCurve *curveRollGyro = new QwtPlotCurve();
        curveRollGyro->setTitle("Gyroscope");
        curveRollGyro->setPen(Qt::red,2);
        curveRollGyro->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveRollGyro->setRawSamples(scaledTimestamps.data(),rollGyro.data(),scaledTimestamps.size());
        curveRollGyro->attach(&plotRoll);
    QwtPlotCurve *curveRoll = new QwtPlotCurve();
        curveRoll->setTitle("Combined");
        curveRoll->setPen(Qt::green,2);
        curveRoll->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveRoll->setRawSamples(scaledTimestamps.data(),roll.data(),scaledTimestamps.size());
        curveRoll->attach(&plotRoll);
    plotRoll.resize(600,400);
    plotRoll.show();

    QwtPlot plotPitch;
    plotPitch.setTitle("Pitch");
    plotPitch.setCanvasBackground(Qt::white);
    plotPitch.insertLegend(new QwtLegend());
    plotPitch.setAxisTitle(QwtPlot::yLeft,"Pitch (°)");
    plotPitch.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotCurve *curvePitchAcc = new QwtPlotCurve();
        curvePitchAcc->setTitle("Accelerometer");
        curvePitchAcc->setPen(Qt::blue,2);
        curvePitchAcc->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curvePitchAcc->setRawSamples(scaledTimestamps.data(),pitchAcc.data(),scaledTimestamps.size());
        curvePitchAcc->attach(&plotPitch);
    QwtPlotCurve *curvePitchGyro = new QwtPlotCurve();
        curvePitchGyro->setTitle("Gyroscope");
        curvePitchGyro->setPen(Qt::red,2);
        curvePitchGyro->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curvePitchGyro->setRawSamples(scaledTimestamps.data(),pitchGyro.data(),scaledTimestamps.size());
        curvePitchGyro->attach(&plotPitch);
    QwtPlotCurve *curvePitch = new QwtPlotCurve();
        curvePitch->setTitle("Combined");
        curvePitch->setPen(Qt::green,2);
        curvePitch->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curvePitch->setRawSamples(scaledTimestamps.data(),pitch.data(),scaledTimestamps.size());
        curvePitch->attach(&plotPitch);
    plotPitch.resize(600,400);
    plotPitch.show();

    QwtPlot plotYaw;
    plotYaw.setTitle("Yaw");
    plotYaw.setCanvasBackground(Qt::white);
    plotYaw.insertLegend(new QwtLegend());
    plotYaw.setAxisTitle(QwtPlot::yLeft,"Yaw (°)");
    plotYaw.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotCurve *curveYaw = new QwtPlotCurve();
        curveYaw->setTitle("Gyroscope");
        curveYaw->setPen(Qt::red,2);
        curveYaw->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveYaw->setRawSamples(scaledTimestamps.data(),yaw.data(),scaledTimestamps.size());
        curveYaw->attach(&plotYaw);
    plotYaw.resize(600,400);
    plotYaw.show();
//
//
    QwtPlot plotHrelations;
    plotHrelations.setTitle("Horizontales relations");
    plotHrelations.setCanvasBackground(Qt::white);
    plotHrelations.insertLegend(new QwtLegend());
    plotHrelations.setAxisTitle(QwtPlot::yLeft," ");
    plotHrelations.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotMarker *mH=new QwtPlotMarker();
        mH->setLineStyle(QwtPlotMarker::HLine);
        mH->setValue(0,0);
        mH->attach(&plotHrelations);
    QwtPlotCurve *curveFrameH = new QwtPlotCurve();
        curveFrameH->setTitle("Frame translation");
        curveFrameH->setPen(Qt::red,2);
        curveFrameH->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveFrameH->setRawSamples(scaledVideoTimestamps.data(),translationsX.data(),scaledVideoTimestamps.size());
        curveFrameH->attach(&plotHrelations);
        //QwtSymbol *symbol = new QwtSymbol(QwtSymbol::Diamond,QBrush(),QPen(Qt::black,2),QSize(2,2));
        //curveFrameH->setSymbol(symbol);
    QwtPlotCurve *curveGyroH = new QwtPlotCurve();
        curveGyroH->setTitle("Gyro rotation");
        curveGyroH->setPen(Qt::blue,2);
        curveGyroH->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveGyroH->setRawSamples(scaledTimestamps.data(),angularRatesZ.data(),scaledTimestamps.size());
        curveGyroH->attach(&plotHrelations);
        //QwtSymbol *symbol2 = new QwtSymbol(QwtSymbol::Diamond,QBrush(),QPen(Qt::black,2),QSize(2,2));
        //curveGyroH->setSymbol(symbol2);
    QwtPlotCurve *curveGyroHsync = new QwtPlotCurve();
        curveGyroHsync->setTitle("Sync gyro rotation");
        curveGyroHsync->setPen(Qt::cyan,2);
        curveGyroHsync->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveGyroHsync->setRawSamples(scaledVideoTimestamps.data(),syncAngularRatesZ.data(),scaledVideoTimestamps.size());
        curveGyroHsync->attach(&plotHrelations);
        //QwtSymbol *symbol3 = new QwtSymbol(QwtSymbol::Diamond,QBrush(),QPen(Qt::black,2),QSize(2,2));
        //curveGyroHsync->setSymbol(symbol3);
    QwtPlotCurve *curveFrameHc = new QwtPlotCurve();
        curveFrameHc->setTitle("Frame translation corrected");
        curveFrameHc->setPen(Qt::magenta,2);
        curveFrameHc->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveFrameHc->setRawSamples(scaledVideoTimestamps.data(),corrTranslationsX.data(),scaledVideoTimestamps.size());
        curveFrameHc->attach(&plotHrelations);
    plotHrelations.resize(600,400);
    plotHrelations.show();

    QwtPlot plotVrelations;
    plotVrelations.setTitle("Verticales relations");
    plotVrelations.setCanvasBackground(Qt::white);
    plotVrelations.insertLegend(new QwtLegend());
    plotVrelations.setAxisTitle(QwtPlot::yLeft," ");
    plotVrelations.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotMarker *mV=new QwtPlotMarker();
        mV->setLineStyle(QwtPlotMarker::HLine);
        mV->setValue(0,0);
        mV->attach(&plotVrelations);
    QwtPlotCurve *curveFrameV = new QwtPlotCurve();
        curveFrameV->setTitle("Frame translation");
        curveFrameV->setPen(Qt::red,2);
        curveFrameV->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        vector<double> n_translationsY(translationsY);
        std::transform(translationsY.begin(),translationsY.end(),n_translationsY.begin(),bind1st(multiplies<double>(),-1));
        curveFrameV->setRawSamples(scaledVideoTimestamps.data(),n_translationsY.data(),scaledVideoTimestamps.size());
        curveFrameV->attach(&plotVrelations);
    QwtPlotCurve *curveGyroV = new QwtPlotCurve();
        curveGyroV->setTitle("Gyro rotation");
        curveGyroV->setPen(Qt::blue,2);
        curveGyroV->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveGyroV->setRawSamples(scaledTimestamps.data(),angularRatesY.data(),scaledTimestamps.size());
        curveGyroV->attach(&plotVrelations);
    QwtPlotCurve *curveGyroVsync = new QwtPlotCurve();
        curveGyroVsync->setTitle("Sync gyro rotation");
        curveGyroVsync->setPen(Qt::cyan,2);
        curveGyroVsync->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveGyroVsync->setRawSamples(scaledVideoTimestamps.data(),syncAngularRatesY.data(),scaledVideoTimestamps.size());
        curveGyroVsync->attach(&plotVrelations);
    QwtPlotCurve *curveFrameVc = new QwtPlotCurve();
        curveFrameVc->setTitle("Frame translation corrected");
        curveFrameVc->setPen(Qt::magenta,2);
        curveFrameVc->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        vector<double> n_corrTranslationsY(corrTranslationsY);
        std::transform(corrTranslationsY.begin(),corrTranslationsY.end(),n_corrTranslationsY.begin(),bind1st(multiplies<double>(),-1));
        curveFrameVc->setRawSamples(scaledVideoTimestamps.data(),n_corrTranslationsY.data(),scaledVideoTimestamps.size());
        curveFrameVc->attach(&plotVrelations);
    plotVrelations.resize(600,400);
    plotVrelations.show();
//
//
    QwtPlot plotTranslationsX;
    plotTranslationsX.setTitle("Translations horizontales in frame plan");
    plotTranslationsX.setCanvasBackground(Qt::white);
    plotTranslationsX.insertLegend(new QwtLegend());
    plotTranslationsX.setAxisTitle(QwtPlot::yLeft,"Translation (px/frame)");
    plotTranslationsX.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotMarker *mTx=new QwtPlotMarker();
        mTx->setLineStyle(QwtPlotMarker::HLine);
        mTx->setValue(0,0);
        mTx->attach(&plotTranslationsX);
    QwtPlotCurve *curveTranslationX = new QwtPlotCurve();
        curveTranslationX->setTitle("x (-Y)");
        curveTranslationX->setPen(Qt::blue,2);
        curveTranslationX->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveTranslationX->setRawSamples(scaledVideoTimestamps.data(),translationsX.data(),scaledVideoTimestamps.size());
        curveTranslationX->attach(&plotTranslationsX);
    QwtPlotCurve *curveFTranslationX = new QwtPlotCurve();
        curveFTranslationX->setTitle("Filtered x (-Y)");
        curveFTranslationX->setPen(Qt::darkBlue,2);
        curveFTranslationX->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveFTranslationX->setRawSamples(scaledVideoTimestamps.data(),filteredTransX.data(),scaledVideoTimestamps.size());
        curveFTranslationX->attach(&plotTranslationsX);
    QwtPlotCurve *curveUTranslationX = new QwtPlotCurve();
        curveUTranslationX->setTitle("Unwanted x (-Y)");
        curveUTranslationX->setPen(Qt::cyan,2);
        curveUTranslationX->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveUTranslationX->setRawSamples(scaledVideoTimestamps.data(),unwantedTransX.data(),scaledVideoTimestamps.size());
        curveUTranslationX->attach(&plotTranslationsX);
    plotTranslationsX.resize(600,400);
    plotTranslationsX.show();

    QwtPlot plotTranslationsY;
    plotTranslationsY.setTitle("Translations verticales in frame plan");
    plotTranslationsY.setCanvasBackground(Qt::white);
    plotTranslationsY.insertLegend(new QwtLegend());
    plotTranslationsY.setAxisTitle(QwtPlot::yLeft,"Translation (px/frame)");
    plotTranslationsY.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotMarker *mTy=new QwtPlotMarker();
        mTy->setLineStyle(QwtPlotMarker::HLine);
        mTy->setValue(0,0);
        mTy->attach(&plotTranslationsY);
    QwtPlotCurve *curveTranslationY = new QwtPlotCurve();
        curveTranslationY->setTitle("y (-Z)");
        curveTranslationY->setPen(Qt::red,2);
        curveTranslationY->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveTranslationY->setRawSamples(scaledVideoTimestamps.data(),translationsY.data(),scaledVideoTimestamps.size());
        curveTranslationY->attach(&plotTranslationsY);
    QwtPlotCurve *curveFTranslationY = new QwtPlotCurve();
        curveFTranslationY->setTitle("Filtered y (-Z)");
        curveFTranslationY->setPen(Qt::darkRed,2);
        curveFTranslationY->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveFTranslationY->setRawSamples(scaledVideoTimestamps.data(),filteredTransY.data(),scaledVideoTimestamps.size());
        curveFTranslationY->attach(&plotTranslationsY);
    QwtPlotCurve *curveUTranslationY = new QwtPlotCurve();
        curveUTranslationY->setTitle("Unwanted y (-Z)");
        curveUTranslationY->setPen(Qt::magenta,2);
        curveUTranslationY->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveUTranslationY->setRawSamples(scaledVideoTimestamps.data(),unwantedTransY.data(),scaledVideoTimestamps.size());
        curveUTranslationY->attach(&plotTranslationsY);
    plotTranslationsY.resize(600,400);
    plotTranslationsY.show();
//
//
    QwtPlot plotCTranslationsX;
    plotCTranslationsX.setTitle("Corrected Translations horizontales in frame plan");
    plotCTranslationsX.setCanvasBackground(Qt::white);
    plotCTranslationsX.insertLegend(new QwtLegend());
    plotCTranslationsX.setAxisTitle(QwtPlot::yLeft,"Translation (px/frame)");
    plotCTranslationsX.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotMarker *mCTx=new QwtPlotMarker();
        mCTx->setLineStyle(QwtPlotMarker::HLine);
        mCTx->setValue(0,0);
        mCTx->attach(&plotCTranslationsX);
    QwtPlotCurve *curveCTranslationX = new QwtPlotCurve();
        curveCTranslationX->setTitle("x (-Y)");
        curveCTranslationX->setPen(Qt::blue,2);
        curveCTranslationX->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveCTranslationX->setRawSamples(scaledVideoTimestamps.data(),corrTranslationsX.data(),scaledVideoTimestamps.size());
        curveCTranslationX->attach(&plotCTranslationsX);
    QwtPlotCurve *curveFCTranslationX = new QwtPlotCurve();
        curveFCTranslationX->setTitle("Filtered x (-Y)");
        curveFCTranslationX->setPen(Qt::darkBlue,2);
        curveFCTranslationX->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveFCTranslationX->setRawSamples(scaledVideoTimestamps.data(),corrFilteredTransX.data(),scaledVideoTimestamps.size());
        curveFCTranslationX->attach(&plotCTranslationsX);
    QwtPlotCurve *curveUCTranslationX = new QwtPlotCurve();
        curveUCTranslationX->setTitle("Unwanted x (-Y)");
        curveUCTranslationX->setPen(Qt::cyan,2);
        curveUCTranslationX->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveUCTranslationX->setRawSamples(scaledVideoTimestamps.data(),corrUnwantedTransX.data(),scaledVideoTimestamps.size());
        curveUCTranslationX->attach(&plotCTranslationsX);
    plotCTranslationsX.resize(600,400);
    plotCTranslationsX.show();

    QwtPlot plotCTranslationsY;
    plotCTranslationsY.setTitle("Corrected Translations verticales in frame plan");
    plotCTranslationsY.setCanvasBackground(Qt::white);
    plotCTranslationsY.insertLegend(new QwtLegend());
    plotCTranslationsY.setAxisTitle(QwtPlot::yLeft,"Translation (px/frame)");
    plotCTranslationsY.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotMarker *mCTy=new QwtPlotMarker();
        mCTy->setLineStyle(QwtPlotMarker::HLine);
        mCTy->setValue(0,0);
        mCTy->attach(&plotCTranslationsY);
    QwtPlotCurve *curveCTranslationY = new QwtPlotCurve();
        curveCTranslationY->setTitle("y (-Z)");
        curveCTranslationY->setPen(Qt::red,2);
        curveCTranslationY->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveCTranslationY->setRawSamples(scaledVideoTimestamps.data(),corrTranslationsY.data(),scaledVideoTimestamps.size());
        curveCTranslationY->attach(&plotCTranslationsY);
    QwtPlotCurve *curveFCTranslationY = new QwtPlotCurve();
        curveFCTranslationY->setTitle("Filtered y (-Z)");
        curveFCTranslationY->setPen(Qt::darkRed,2);
        curveFCTranslationY->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveFCTranslationY->setRawSamples(scaledVideoTimestamps.data(),corrFilteredTransY.data(),scaledVideoTimestamps.size());
        curveFCTranslationY->attach(&plotCTranslationsY);
    QwtPlotCurve *curveUCTranslationY = new QwtPlotCurve();
        curveUCTranslationY->setTitle("Unwanted y (-Z)");
        curveUCTranslationY->setPen(Qt::magenta,2);
        curveUCTranslationY->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveUCTranslationY->setRawSamples(scaledVideoTimestamps.data(),corrUnwantedTransY.data(),scaledVideoTimestamps.size());
        curveUCTranslationY->attach(&plotCTranslationsY);
    plotCTranslationsY.resize(600,400);
    plotCTranslationsY.show();
//

//*** Second round

    // Read video frames and compute trajectories
tic();
    vector<Trajectory> trajectories2;
    int nbFrames2;

    VideoMotionEstimation::calcTrajectoriesKLT(trajectories2,nbFrames2,"/home/maelle/Desktop/Samples/Static/"+videoName+"/"+videoName+"_stable_F.avi",2000,500);
toc();

tic();
    // Compute global motions (affine transformations) between each pair of frames
    vector<Mat> affines;

    VideoMotionEstimation::calcGlobalMotions(affines,videostab::MM_AFFINE,trajectories2,nbFrames2,true,true);

    // Convert the affine matrices into vectors and compute skew and ratio parameters
    vector<double> affinesA, affinesB, affinesC, affinesD, affinesTx, affinesTy, affinesSkew, affinesRatio;

    VideoMotionEstimation::convertAffineData(nbFrames2,affines,affinesA,affinesB,affinesC,affinesD,affinesTx,affinesTy,affinesSkew,affinesRatio);

    // Limit the skew and ratio distortion
    VideoMotionEstimation::limitationFilter(affinesTx,affinesTy,affinesSkew,affinesRatio);
toc();

tic();
    // Correct the video processing global motions using the sensor data
    vector<double> corrAffinesA, corrAffinesB, corrAffinesC, corrAffinesD, corrAffinesTx, corrAffinesTy, corrAffinesSkew, corrAffinesRatio;
    Fusion::noMotionDetection(affinesA,affinesB,affinesC,affinesD,affinesTx,affinesTy,affinesSkew,affinesRatio,syncAngularRatesX,syncAngularRatesY,syncAngularRatesZ,
                              corrAffinesA,corrAffinesB,corrAffinesC,corrAffinesD,corrAffinesTx,corrAffinesTy,corrAffinesSkew,corrAffinesRatio);
toc();

tic();
    // Compute the unwanted global motions using low pass filtering
    vector<double> filteredAffinesA, filteredAffinesB, filteredAffinesC, filteredAffinesD, filteredAffinesTx, filteredAffinesTy, filteredAffinesSkew, filteredAffinesRatio;
    vector<double> unwantedAffinesA, unwantedAffinesB, unwantedAffinesC, unwantedAffinesD, unwantedAffinesTx, unwantedAffinesTy, unwantedAffinesSkew, unwantedAffinesRatio;

    Stabilization::calcUnwantedMotion(0.45,affinesA,affinesB,affinesC,affinesD,affinesTx,affinesTy,affinesSkew,affinesRatio,
                                      filteredAffinesA,filteredAffinesB,filteredAffinesC,filteredAffinesD,filteredAffinesTx,filteredAffinesTy,filteredAffinesSkew,filteredAffinesRatio,
                                      unwantedAffinesA,unwantedAffinesB,unwantedAffinesC,unwantedAffinesD,unwantedAffinesTx,unwantedAffinesTy,unwantedAffinesSkew,unwantedAffinesRatio);

    // Compute the unwanted global motions using low pass filtering for fusion data
    vector<double> corrFilteredAffinesA, corrFilteredAffinesB, corrFilteredAffinesC, corrFilteredAffinesD, corrFilteredAffinesTx, corrFilteredAffinesTy, corrFilteredAffinesSkew, corrFilteredAffinesRatio;
    vector<double> corrUnwantedAffinesA, corrUnwantedAffinesB, corrUnwantedAffinesC, corrUnwantedAffinesD, corrUnwantedAffinesTx, corrUnwantedAffinesTy, corrUnwantedAffinesSkew, corrUnwantedAffinesRatio;

    Stabilization::calcUnwantedMotion(0.45,corrAffinesA,corrAffinesB,corrAffinesC,corrAffinesD,corrAffinesTx,corrAffinesTy,corrAffinesSkew,corrAffinesRatio,
                                      corrFilteredAffinesA,corrFilteredAffinesB,corrFilteredAffinesC,corrFilteredAffinesD,
                                      corrFilteredAffinesTx,corrFilteredAffinesTy,corrFilteredAffinesSkew,corrFilteredAffinesRatio,
                                      corrUnwantedAffinesA,corrUnwantedAffinesB,corrUnwantedAffinesC,corrUnwantedAffinesD,
                                      corrUnwantedAffinesTx,corrUnwantedAffinesTy,corrUnwantedAffinesSkew,corrUnwantedAffinesRatio);
toc();

tic();
    // Compute the transformation matrices for stabilization
    vector<Mat> unwantedAffineMotions;
    Stabilization::unwantedAffineMotions(unwantedAffinesA,unwantedAffinesB,unwantedAffinesC,unwantedAffinesD,unwantedAffinesTx,unwantedAffinesTy,
                                         unwantedAffinesSkew,unwantedAffinesRatio,unwantedAffineMotions);
    // Compute the size of the crop window
    cropX=0, cropY=0;
    Stabilization::cropSize(unwantedAffinesTx,unwantedAffinesTy,cropX,cropY);
    cout << "Crop size: " << cropX << ":" << cropY << endl;

    // Warp the transformation matrices to each frame to stabilize the video
    Stabilization::stabilizeVideo(videoName,videoName+"_stable_F",videoName+"_stable2_VP",nbFrames2,unwantedAffineMotions,cropX,cropY);
toc();

tic();
    // Compute the transformation matrices for stabilization using fusion data
    vector<Mat> corrUnwantedAffineMotions;
    Stabilization::unwantedAffineMotions(corrUnwantedAffinesA,corrUnwantedAffinesB,corrUnwantedAffinesC,corrUnwantedAffinesD,corrUnwantedAffinesTx,corrUnwantedAffinesTy,
                                         corrUnwantedAffinesSkew,corrUnwantedAffinesRatio,corrUnwantedAffineMotions);
    // Compute the size of the crop window
    cropX=0; cropY=0;
    Stabilization::cropSize(corrUnwantedAffinesTx,corrUnwantedAffinesTy,cropX,cropY);
    cout << "Crop size: " << cropX << ":" << cropY << endl;

    // Warp the transformation matrices to each frame to stabilize the video
    Stabilization::stabilizeVideo(videoName,videoName+"_stable_F",videoName+"_stable2_F",nbFrames2,corrUnwantedAffineMotions,cropX,cropY);
toc();

    // Plots
//
    QwtPlot plotAffinesA;
    plotAffinesA.setTitle("Affine transformation parameter a");
    plotAffinesA.setCanvasBackground(Qt::white);
    plotAffinesA.insertLegend(new QwtLegend());
    plotAffinesA.setAxisTitle(QwtPlot::yLeft,"Scale + Rotation");
    plotAffinesA.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotMarker *mAa=new QwtPlotMarker();
        mAa->setLineStyle(QwtPlotMarker::HLine);
        mAa->setValue(0,0);
        mAa->attach(&plotAffinesA);
    QwtPlotCurve *curveAffinesA = new QwtPlotCurve();
        curveAffinesA->setTitle("a");
        curveAffinesA->setPen(Qt::blue,2);
        curveAffinesA->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveAffinesA->setRawSamples(scaledVideoTimestamps.data(),affinesA.data(),scaledVideoTimestamps.size());
        curveAffinesA->attach(&plotAffinesA);
    QwtPlotCurve *curveFAffinesA = new QwtPlotCurve();
        curveFAffinesA->setTitle("Filtered a");
        curveFAffinesA->setPen(Qt::darkBlue,2);
        curveFAffinesA->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveFAffinesA->setRawSamples(scaledVideoTimestamps.data(),filteredAffinesA.data(),scaledVideoTimestamps.size());
        curveFAffinesA->attach(&plotAffinesA);
    QwtPlotCurve *curveUAffinesA = new QwtPlotCurve();
        curveUAffinesA->setTitle("Unwanted a");
        curveUAffinesA->setPen(Qt::cyan,2);
        curveUAffinesA->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveUAffinesA->setRawSamples(scaledVideoTimestamps.data(),unwantedAffinesA.data(),scaledVideoTimestamps.size());
        curveUAffinesA->attach(&plotAffinesA);
    plotAffinesA.resize(600,400);
    plotAffinesA.show();

    QwtPlot plotAffinesB;
    plotAffinesB.setTitle("Affine transformation parameter b");
    plotAffinesB.setCanvasBackground(Qt::white);
    plotAffinesB.insertLegend(new QwtLegend());
    plotAffinesB.setAxisTitle(QwtPlot::yLeft,"Skew + Rotation");
    plotAffinesB.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotMarker *mAb=new QwtPlotMarker();
        mAb->setLineStyle(QwtPlotMarker::HLine);
        mAb->setValue(0,0);
        mAb->attach(&plotAffinesB);
    QwtPlotCurve *curveAffinesB = new QwtPlotCurve();
        curveAffinesB->setTitle("b");
        curveAffinesB->setPen(Qt::blue,2);
        curveAffinesB->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveAffinesB->setRawSamples(scaledVideoTimestamps.data(),affinesB.data(),scaledVideoTimestamps.size());
        curveAffinesB->attach(&plotAffinesB);
    QwtPlotCurve *curveFAffinesB = new QwtPlotCurve();
        curveFAffinesB->setTitle("Filtered b");
        curveFAffinesB->setPen(Qt::darkBlue,2);
        curveFAffinesB->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveFAffinesB->setRawSamples(scaledVideoTimestamps.data(),filteredAffinesB.data(),scaledVideoTimestamps.size());
        curveFAffinesB->attach(&plotAffinesB);
    QwtPlotCurve *curveUAffinesB = new QwtPlotCurve();
        curveUAffinesB->setTitle("Unwanted b");
        curveUAffinesB->setPen(Qt::cyan,2);
        curveUAffinesB->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveUAffinesB->setRawSamples(scaledVideoTimestamps.data(),unwantedAffinesB.data(),scaledVideoTimestamps.size());
        curveUAffinesB->attach(&plotAffinesB);
    plotAffinesB.resize(600,400);
    plotAffinesB.show();

    QwtPlot plotAffinesC;
    plotAffinesC.setTitle("Affine transformation parameter c");
    plotAffinesC.setCanvasBackground(Qt::white);
    plotAffinesC.insertLegend(new QwtLegend());
    plotAffinesC.setAxisTitle(QwtPlot::yLeft,"Skew + Rotation");
    plotAffinesC.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotMarker *mAc=new QwtPlotMarker();
        mAc->setLineStyle(QwtPlotMarker::HLine);
        mAc->setValue(0,0);
        mAc->attach(&plotAffinesC);
    QwtPlotCurve *curveAffinesC = new QwtPlotCurve();
        curveAffinesC->setTitle("c");
        curveAffinesC->setPen(Qt::blue,2);
        curveAffinesC->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveAffinesC->setRawSamples(scaledVideoTimestamps.data(),affinesC.data(),scaledVideoTimestamps.size());
        curveAffinesC->attach(&plotAffinesC);
    QwtPlotCurve *curveFAffinesC = new QwtPlotCurve();
        curveFAffinesC->setTitle("Filtered c");
        curveFAffinesC->setPen(Qt::darkBlue,2);
        curveFAffinesC->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveFAffinesC->setRawSamples(scaledVideoTimestamps.data(),filteredAffinesC.data(),scaledVideoTimestamps.size());
        curveFAffinesC->attach(&plotAffinesC);
    QwtPlotCurve *curveUAffinesC = new QwtPlotCurve();
        curveUAffinesC->setTitle("Unwanted c");
        curveUAffinesC->setPen(Qt::cyan,2);
        curveUAffinesC->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveUAffinesC->setRawSamples(scaledVideoTimestamps.data(),unwantedAffinesC.data(),scaledVideoTimestamps.size());
        curveUAffinesC->attach(&plotAffinesC);
    plotAffinesC.resize(600,400);
    plotAffinesC.show();

    QwtPlot plotAffinesD;
    plotAffinesD.setTitle("Affine transformation parameter d");
    plotAffinesD.setCanvasBackground(Qt::white);
    plotAffinesD.insertLegend(new QwtLegend());
    plotAffinesD.setAxisTitle(QwtPlot::yLeft,"Scale + Rotation");
    plotAffinesD.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotMarker *mAd=new QwtPlotMarker();
        mAd->setLineStyle(QwtPlotMarker::HLine);
        mAd->setValue(0,0);
        mAd->attach(&plotAffinesD);
    QwtPlotCurve *curveAffinesD = new QwtPlotCurve();
        curveAffinesD->setTitle("d");
        curveAffinesD->setPen(Qt::blue,2);
        curveAffinesD->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveAffinesD->setRawSamples(scaledVideoTimestamps.data(),affinesD.data(),scaledVideoTimestamps.size());
        curveAffinesD->attach(&plotAffinesD);
    QwtPlotCurve *curveFAffinesD = new QwtPlotCurve();
        curveFAffinesD->setTitle("Filtered d");
        curveFAffinesD->setPen(Qt::darkBlue,2);
        curveFAffinesD->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveFAffinesD->setRawSamples(scaledVideoTimestamps.data(),filteredAffinesD.data(),scaledVideoTimestamps.size());
        curveFAffinesD->attach(&plotAffinesD);
    QwtPlotCurve *curveUAffinesD = new QwtPlotCurve();
        curveUAffinesD->setTitle("Unwanted d");
        curveUAffinesD->setPen(Qt::cyan,2);
        curveUAffinesD->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveUAffinesD->setRawSamples(scaledVideoTimestamps.data(),unwantedAffinesD.data(),scaledVideoTimestamps.size());
        curveUAffinesD->attach(&plotAffinesD);
    plotAffinesD.resize(600,400);
    plotAffinesD.show();

    QwtPlot plotAffinesTx;
    plotAffinesTx.setTitle("Affine transformation parameter Tx");
    plotAffinesTx.setCanvasBackground(Qt::white);
    plotAffinesTx.insertLegend(new QwtLegend());
    plotAffinesTx.setAxisTitle(QwtPlot::yLeft,"Translation (px)");
    plotAffinesTx.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotMarker *mATx=new QwtPlotMarker();
        mATx->setLineStyle(QwtPlotMarker::HLine);
        mATx->setValue(0,0);
        mATx->attach(&plotAffinesTx);
    QwtPlotCurve *curveAffinesTx = new QwtPlotCurve();
        curveAffinesTx->setTitle("Tx");
        curveAffinesTx->setPen(Qt::blue,2);
        curveAffinesTx->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveAffinesTx->setRawSamples(scaledVideoTimestamps.data(),affinesTx.data(),scaledVideoTimestamps.size());
        curveAffinesTx->attach(&plotAffinesTx);
    QwtPlotCurve *curveFAffinesTx = new QwtPlotCurve();
        curveFAffinesTx->setTitle("Filtered Tx");
        curveFAffinesTx->setPen(Qt::darkBlue,2);
        curveFAffinesTx->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveFAffinesTx->setRawSamples(scaledVideoTimestamps.data(),filteredAffinesTx.data(),scaledVideoTimestamps.size());
        curveFAffinesTx->attach(&plotAffinesTx);
    QwtPlotCurve *curveUAffinesTx = new QwtPlotCurve();
        curveUAffinesTx->setTitle("Unwanted Tx");
        curveUAffinesTx->setPen(Qt::cyan,2);
        curveUAffinesTx->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveUAffinesTx->setRawSamples(scaledVideoTimestamps.data(),unwantedAffinesTx.data(),scaledVideoTimestamps.size());
        curveUAffinesTx->attach(&plotAffinesTx);
    plotAffinesTx.resize(600,400);
    plotAffinesTx.show();

    QwtPlot plotAffinesTy;
    plotAffinesTy.setTitle("Affine transformation parameter Ty");
    plotAffinesTy.setCanvasBackground(Qt::white);
    plotAffinesTy.insertLegend(new QwtLegend());
    plotAffinesTy.setAxisTitle(QwtPlot::yLeft,"Translation (px)");
    plotAffinesTy.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotMarker *mATy=new QwtPlotMarker();
        mATy->setLineStyle(QwtPlotMarker::HLine);
        mATy->setValue(0,0);
        mATy->attach(&plotAffinesTy);
    QwtPlotCurve *curveAffinesTy = new QwtPlotCurve();
        curveAffinesTy->setTitle("Ty");
        curveAffinesTy->setPen(Qt::blue,2);
        curveAffinesTy->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveAffinesTy->setRawSamples(scaledVideoTimestamps.data(),affinesTy.data(),scaledVideoTimestamps.size());
        curveAffinesTy->attach(&plotAffinesTy);
    QwtPlotCurve *curveFAffinesTy = new QwtPlotCurve();
        curveFAffinesTy->setTitle("Filtered Ty");
        curveFAffinesTy->setPen(Qt::darkBlue,2);
        curveFAffinesTy->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveFAffinesTy->setRawSamples(scaledVideoTimestamps.data(),filteredAffinesTy.data(),scaledVideoTimestamps.size());
        curveFAffinesTy->attach(&plotAffinesTy);
    QwtPlotCurve *curveUAffinesTy = new QwtPlotCurve();
        curveUAffinesTy->setTitle("Unwanted Ty");
        curveUAffinesTy->setPen(Qt::cyan,2);
        curveUAffinesTy->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveUAffinesTy->setRawSamples(scaledVideoTimestamps.data(),unwantedAffinesTy.data(),scaledVideoTimestamps.size());
        curveUAffinesTy->attach(&plotAffinesTy);
    plotAffinesTy.resize(600,400);
    plotAffinesTy.show();

    QwtPlot plotAffinesSkew;
    plotAffinesSkew.setTitle("Affine transformation skew");
    plotAffinesSkew.setCanvasBackground(Qt::white);
    plotAffinesSkew.insertLegend(new QwtLegend());
    plotAffinesSkew.setAxisTitle(QwtPlot::yLeft,"Skew");
    plotAffinesSkew.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotMarker *mAs=new QwtPlotMarker();
        mAs->setLineStyle(QwtPlotMarker::HLine);
        mAs->setValue(0,0);
        mAs->attach(&plotAffinesSkew);
    QwtPlotCurve *curveAffinesSkew = new QwtPlotCurve();
        curveAffinesSkew->setTitle("skew");
        curveAffinesSkew->setPen(Qt::blue,2);
        curveAffinesSkew->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveAffinesSkew->setRawSamples(scaledVideoTimestamps.data(),affinesSkew.data(),scaledVideoTimestamps.size());
        curveAffinesSkew->attach(&plotAffinesSkew);
    QwtPlotCurve *curveFAffinesSkew = new QwtPlotCurve();
        curveFAffinesSkew->setTitle("Filtered skew");
        curveFAffinesSkew->setPen(Qt::darkBlue,2);
        curveFAffinesSkew->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveFAffinesSkew->setRawSamples(scaledVideoTimestamps.data(),filteredAffinesSkew.data(),scaledVideoTimestamps.size());
        curveFAffinesSkew->attach(&plotAffinesSkew);
    QwtPlotCurve *curveUAffinesSkew = new QwtPlotCurve();
        curveUAffinesSkew->setTitle("Unwanted skew");
        curveUAffinesSkew->setPen(Qt::cyan,2);
        curveUAffinesSkew->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveUAffinesSkew->setRawSamples(scaledVideoTimestamps.data(),unwantedAffinesSkew.data(),scaledVideoTimestamps.size());
        curveUAffinesSkew->attach(&plotAffinesSkew);
    plotAffinesSkew.resize(600,400);
    plotAffinesSkew.show();

    QwtPlot plotAffinesRatio;
    plotAffinesRatio.setTitle("Affine transformation ratio");
    plotAffinesRatio.setCanvasBackground(Qt::white);
    plotAffinesRatio.insertLegend(new QwtLegend());
    plotAffinesRatio.setAxisTitle(QwtPlot::yLeft,"Ratio");
    plotAffinesRatio.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotMarker *mAr=new QwtPlotMarker();
        mAr->setLineStyle(QwtPlotMarker::HLine);
        mAr->setValue(0,0);
        mAr->attach(&plotAffinesRatio);
    QwtPlotCurve *curveAffinesRatio = new QwtPlotCurve();
        curveAffinesRatio->setTitle("ratio");
        curveAffinesRatio->setPen(Qt::blue,2);
        curveAffinesRatio->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveAffinesRatio->setRawSamples(scaledVideoTimestamps.data(),affinesRatio.data(),scaledVideoTimestamps.size());
        curveAffinesRatio->attach(&plotAffinesRatio);
    QwtPlotCurve *curveFAffinesRatio = new QwtPlotCurve();
        curveFAffinesRatio->setTitle("Filtered ratio");
        curveFAffinesRatio->setPen(Qt::darkBlue,2);
        curveFAffinesRatio->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveFAffinesRatio->setRawSamples(scaledVideoTimestamps.data(),filteredAffinesRatio.data(),scaledVideoTimestamps.size());
        curveFAffinesRatio->attach(&plotAffinesRatio);
    QwtPlotCurve *curveUAffinesRatio = new QwtPlotCurve();
        curveUAffinesRatio->setTitle("Unwanted ratio");
        curveUAffinesRatio->setPen(Qt::cyan,2);
        curveUAffinesRatio->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveUAffinesRatio->setRawSamples(scaledVideoTimestamps.data(),unwantedAffinesRatio.data(),scaledVideoTimestamps.size());
        curveUAffinesRatio->attach(&plotAffinesRatio);
    plotAffinesRatio.resize(600,400);
    plotAffinesRatio.show();
//
//
    QwtPlot plotCAffinesA;
    plotCAffinesA.setTitle("Corrected Affine transformation parameter a");
    plotCAffinesA.setCanvasBackground(Qt::white);
    plotCAffinesA.insertLegend(new QwtLegend());
    plotCAffinesA.setAxisTitle(QwtPlot::yLeft,"Scale + Rotation");
    plotCAffinesA.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotMarker *mCAa=new QwtPlotMarker();
        mCAa->setLineStyle(QwtPlotMarker::HLine);
        mCAa->setValue(0,0);
        mCAa->attach(&plotCAffinesA);
    QwtPlotCurve *curveCAffinesA = new QwtPlotCurve();
        curveCAffinesA->setTitle("a");
        curveCAffinesA->setPen(Qt::blue,2);
        curveCAffinesA->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveCAffinesA->setRawSamples(scaledVideoTimestamps.data(),corrAffinesA.data(),scaledVideoTimestamps.size());
        curveCAffinesA->attach(&plotCAffinesA);
    QwtPlotCurve *curveFCAffinesA = new QwtPlotCurve();
        curveFCAffinesA->setTitle("Filtered a");
        curveFCAffinesA->setPen(Qt::darkBlue,2);
        curveFCAffinesA->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveFCAffinesA->setRawSamples(scaledVideoTimestamps.data(),corrFilteredAffinesA.data(),scaledVideoTimestamps.size());
        curveFCAffinesA->attach(&plotCAffinesA);
    QwtPlotCurve *curveUCAffinesA = new QwtPlotCurve();
        curveUCAffinesA->setTitle("Unwanted a");
        curveUCAffinesA->setPen(Qt::cyan,2);
        curveUCAffinesA->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveUCAffinesA->setRawSamples(scaledVideoTimestamps.data(),corrUnwantedAffinesA.data(),scaledVideoTimestamps.size());
        curveUCAffinesA->attach(&plotCAffinesA);
    plotCAffinesA.resize(600,400);
    plotCAffinesA.show();

    QwtPlot plotCAffinesB;
    plotCAffinesB.setTitle("Corrected Affine transformation parameter b");
    plotCAffinesB.setCanvasBackground(Qt::white);
    plotCAffinesB.insertLegend(new QwtLegend());
    plotCAffinesB.setAxisTitle(QwtPlot::yLeft,"Skew + Rotation");
    plotCAffinesB.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotMarker *mCAb=new QwtPlotMarker();
        mCAb->setLineStyle(QwtPlotMarker::HLine);
        mCAb->setValue(0,0);
        mCAb->attach(&plotCAffinesB);
    QwtPlotCurve *curveCAffinesB = new QwtPlotCurve();
        curveCAffinesB->setTitle("b");
        curveCAffinesB->setPen(Qt::blue,2);
        curveCAffinesB->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveCAffinesB->setRawSamples(scaledVideoTimestamps.data(),corrAffinesB.data(),scaledVideoTimestamps.size());
        curveCAffinesB->attach(&plotCAffinesB);
    QwtPlotCurve *curveFCAffinesB = new QwtPlotCurve();
        curveFCAffinesB->setTitle("Filtered b");
        curveFCAffinesB->setPen(Qt::darkBlue,2);
        curveFCAffinesB->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveFCAffinesB->setRawSamples(scaledVideoTimestamps.data(),corrFilteredAffinesB.data(),scaledVideoTimestamps.size());
        curveFCAffinesB->attach(&plotCAffinesB);
    QwtPlotCurve *curveUCAffinesB = new QwtPlotCurve();
        curveUCAffinesB->setTitle("Unwanted b");
        curveUCAffinesB->setPen(Qt::cyan,2);
        curveUCAffinesB->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveUCAffinesB->setRawSamples(scaledVideoTimestamps.data(),corrUnwantedAffinesB.data(),scaledVideoTimestamps.size());
        curveUCAffinesB->attach(&plotCAffinesB);
    plotCAffinesB.resize(600,400);
    plotCAffinesB.show();

    QwtPlot plotCaffinesC;
    plotCaffinesC.setTitle("Corrected Affine transformation parameter c");
    plotCaffinesC.setCanvasBackground(Qt::white);
    plotCaffinesC.insertLegend(new QwtLegend());
    plotCaffinesC.setAxisTitle(QwtPlot::yLeft,"Skew + Rotation");
    plotCaffinesC.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotMarker *mCAc=new QwtPlotMarker();
        mCAc->setLineStyle(QwtPlotMarker::HLine);
        mCAc->setValue(0,0);
        mCAc->attach(&plotCaffinesC);
    QwtPlotCurve *curveCaffinesC = new QwtPlotCurve();
        curveCaffinesC->setTitle("c");
        curveCaffinesC->setPen(Qt::blue,2);
        curveCaffinesC->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveCaffinesC->setRawSamples(scaledVideoTimestamps.data(),corrAffinesC.data(),scaledVideoTimestamps.size());
        curveCaffinesC->attach(&plotCaffinesC);
    QwtPlotCurve *curveFCaffinesC = new QwtPlotCurve();
        curveFCaffinesC->setTitle("Filtered c");
        curveFCaffinesC->setPen(Qt::darkBlue,2);
        curveFCaffinesC->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveFCaffinesC->setRawSamples(scaledVideoTimestamps.data(),corrFilteredAffinesC.data(),scaledVideoTimestamps.size());
        curveFCaffinesC->attach(&plotCaffinesC);
    QwtPlotCurve *curveUCaffinesC = new QwtPlotCurve();
        curveUCaffinesC->setTitle("Unwanted c");
        curveUCaffinesC->setPen(Qt::cyan,2);
        curveUCaffinesC->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveUCaffinesC->setRawSamples(scaledVideoTimestamps.data(),corrUnwantedAffinesC.data(),scaledVideoTimestamps.size());
        curveUCaffinesC->attach(&plotCaffinesC);
    plotCaffinesC.resize(600,400);
    plotCaffinesC.show();

    QwtPlot plotCaffinesD;
    plotCaffinesD.setTitle("Corrected Affine transformation parameter d");
    plotCaffinesD.setCanvasBackground(Qt::white);
    plotCaffinesD.insertLegend(new QwtLegend());
    plotCaffinesD.setAxisTitle(QwtPlot::yLeft,"Scale + Rotation");
    plotCaffinesD.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotMarker *mCAd=new QwtPlotMarker();
        mCAd->setLineStyle(QwtPlotMarker::HLine);
        mCAd->setValue(0,0);
        mCAd->attach(&plotCaffinesD);
    QwtPlotCurve *curveCaffinesD = new QwtPlotCurve();
        curveCaffinesD->setTitle("d");
        curveCaffinesD->setPen(Qt::blue,2);
        curveCaffinesD->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveCaffinesD->setRawSamples(scaledVideoTimestamps.data(),corrAffinesD.data(),scaledVideoTimestamps.size());
        curveCaffinesD->attach(&plotCaffinesD);
    QwtPlotCurve *curveFCaffinesD = new QwtPlotCurve();
        curveFCaffinesD->setTitle("Filtered d");
        curveFCaffinesD->setPen(Qt::darkBlue,2);
        curveFCaffinesD->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveFCaffinesD->setRawSamples(scaledVideoTimestamps.data(),corrFilteredAffinesD.data(),scaledVideoTimestamps.size());
        curveFCaffinesD->attach(&plotCaffinesD);
    QwtPlotCurve *curveUCaffinesD = new QwtPlotCurve();
        curveUCaffinesD->setTitle("Unwanted d");
        curveUCaffinesD->setPen(Qt::cyan,2);
        curveUCaffinesD->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveUCaffinesD->setRawSamples(scaledVideoTimestamps.data(),corrUnwantedAffinesD.data(),scaledVideoTimestamps.size());
        curveUCaffinesD->attach(&plotCaffinesD);
    plotCaffinesD.resize(600,400);
    plotCaffinesD.show();

    QwtPlot plotCaffinesTx;
    plotCaffinesTx.setTitle("Corrected Affine transformation parameter Tx");
    plotCaffinesTx.setCanvasBackground(Qt::white);
    plotCaffinesTx.insertLegend(new QwtLegend());
    plotCaffinesTx.setAxisTitle(QwtPlot::yLeft,"Translation (px)");
    plotCaffinesTx.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotMarker *mCATx=new QwtPlotMarker();
        mCATx->setLineStyle(QwtPlotMarker::HLine);
        mCATx->setValue(0,0);
        mCATx->attach(&plotCaffinesTx);
    QwtPlotCurve *curveCaffinesTx = new QwtPlotCurve();
        curveCaffinesTx->setTitle("Tx");
        curveCaffinesTx->setPen(Qt::blue,2);
        curveCaffinesTx->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveCaffinesTx->setRawSamples(scaledVideoTimestamps.data(),corrAffinesTx.data(),scaledVideoTimestamps.size());
        curveCaffinesTx->attach(&plotCaffinesTx);
    QwtPlotCurve *curveFCaffinesTx = new QwtPlotCurve();
        curveFCaffinesTx->setTitle("Filtered Tx");
        curveFCaffinesTx->setPen(Qt::darkBlue,2);
        curveFCaffinesTx->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveFCaffinesTx->setRawSamples(scaledVideoTimestamps.data(),corrFilteredAffinesTx.data(),scaledVideoTimestamps.size());
        curveFCaffinesTx->attach(&plotCaffinesTx);
    QwtPlotCurve *curveUCaffinesTx = new QwtPlotCurve();
        curveUCaffinesTx->setTitle("Unwanted Tx");
        curveUCaffinesTx->setPen(Qt::cyan,2);
        curveUCaffinesTx->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveUCaffinesTx->setRawSamples(scaledVideoTimestamps.data(),corrUnwantedAffinesTx.data(),scaledVideoTimestamps.size());
        curveUCaffinesTx->attach(&plotCaffinesTx);
    plotCaffinesTx.resize(600,400);
    plotCaffinesTx.show();

    QwtPlot plotCaffinesTy;
    plotCaffinesTy.setTitle("Corrected Affine transformation parameter Ty");
    plotCaffinesTy.setCanvasBackground(Qt::white);
    plotCaffinesTy.insertLegend(new QwtLegend());
    plotCaffinesTy.setAxisTitle(QwtPlot::yLeft,"Translation (px)");
    plotCaffinesTy.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotMarker *mCATy=new QwtPlotMarker();
        mCATy->setLineStyle(QwtPlotMarker::HLine);
        mCATy->setValue(0,0);
        mCATy->attach(&plotCaffinesTy);
    QwtPlotCurve *curveCaffinesTy = new QwtPlotCurve();
        curveCaffinesTy->setTitle("Ty");
        curveCaffinesTy->setPen(Qt::blue,2);
        curveCaffinesTy->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveCaffinesTy->setRawSamples(scaledVideoTimestamps.data(),corrAffinesTy.data(),scaledVideoTimestamps.size());
        curveCaffinesTy->attach(&plotCaffinesTy);
    QwtPlotCurve *curveFCaffinesTy = new QwtPlotCurve();
        curveFCaffinesTy->setTitle("Filtered Ty");
        curveFCaffinesTy->setPen(Qt::darkBlue,2);
        curveFCaffinesTy->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveFCaffinesTy->setRawSamples(scaledVideoTimestamps.data(),corrFilteredAffinesTy.data(),scaledVideoTimestamps.size());
        curveFCaffinesTy->attach(&plotCaffinesTy);
    QwtPlotCurve *curveUCaffinesTy = new QwtPlotCurve();
        curveUCaffinesTy->setTitle("Unwanted Ty");
        curveUCaffinesTy->setPen(Qt::cyan,2);
        curveUCaffinesTy->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveUCaffinesTy->setRawSamples(scaledVideoTimestamps.data(),corrUnwantedAffinesTy.data(),scaledVideoTimestamps.size());
        curveUCaffinesTy->attach(&plotCaffinesTy);
    plotCaffinesTy.resize(600,400);
    plotCaffinesTy.show();

    QwtPlot plotCAffinesSkew;
    plotCAffinesSkew.setTitle("Corrected Affine transformation skew");
    plotCAffinesSkew.setCanvasBackground(Qt::white);
    plotCAffinesSkew.insertLegend(new QwtLegend());
    plotCAffinesSkew.setAxisTitle(QwtPlot::yLeft,"Skew");
    plotCAffinesSkew.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotMarker *mCAs=new QwtPlotMarker();
        mCAs->setLineStyle(QwtPlotMarker::HLine);
        mCAs->setValue(0,0);
        mCAs->attach(&plotCAffinesSkew);
    QwtPlotCurve *curveCAffinesSkew = new QwtPlotCurve();
        curveCAffinesSkew->setTitle("skew");
        curveCAffinesSkew->setPen(Qt::blue,2);
        curveCAffinesSkew->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveCAffinesSkew->setRawSamples(scaledVideoTimestamps.data(),corrAffinesSkew.data(),scaledVideoTimestamps.size());
        curveCAffinesSkew->attach(&plotCAffinesSkew);
    QwtPlotCurve *curveFCAffinesSkew = new QwtPlotCurve();
        curveFCAffinesSkew->setTitle("Filtered skew");
        curveFCAffinesSkew->setPen(Qt::darkBlue,2);
        curveFCAffinesSkew->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveFCAffinesSkew->setRawSamples(scaledVideoTimestamps.data(),corrFilteredAffinesSkew.data(),scaledVideoTimestamps.size());
        curveFCAffinesSkew->attach(&plotCAffinesSkew);
    QwtPlotCurve *curveUCAffinesSkew = new QwtPlotCurve();
        curveUCAffinesSkew->setTitle("Unwanted skew");
        curveUCAffinesSkew->setPen(Qt::cyan,2);
        curveUCAffinesSkew->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveUCAffinesSkew->setRawSamples(scaledVideoTimestamps.data(),corrUnwantedAffinesSkew.data(),scaledVideoTimestamps.size());
        curveUCAffinesSkew->attach(&plotCAffinesSkew);
    plotCAffinesSkew.resize(600,400);
    plotCAffinesSkew.show();

    QwtPlot plotCAffinesRatio;
    plotCAffinesRatio.setTitle("Corrected Affine transformation ratio");
    plotCAffinesRatio.setCanvasBackground(Qt::white);
    plotCAffinesRatio.insertLegend(new QwtLegend());
    plotCAffinesRatio.setAxisTitle(QwtPlot::yLeft,"Ratio");
    plotCAffinesRatio.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotMarker *mCAr=new QwtPlotMarker();
        mCAr->setLineStyle(QwtPlotMarker::HLine);
        mCAr->setValue(0,0);
        mCAr->attach(&plotCAffinesRatio);
    QwtPlotCurve *curveCAffinesRatio = new QwtPlotCurve();
        curveCAffinesRatio->setTitle("ratio");
        curveCAffinesRatio->setPen(Qt::blue,2);
        curveCAffinesRatio->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveCAffinesRatio->setRawSamples(scaledVideoTimestamps.data(),corrAffinesRatio.data(),scaledVideoTimestamps.size());
        curveCAffinesRatio->attach(&plotCAffinesRatio);
    QwtPlotCurve *curveFCAffinesRatio = new QwtPlotCurve();
        curveFCAffinesRatio->setTitle("Filtered ratio");
        curveFCAffinesRatio->setPen(Qt::darkBlue,2);
        curveFCAffinesRatio->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveFCAffinesRatio->setRawSamples(scaledVideoTimestamps.data(),corrFilteredAffinesRatio.data(),scaledVideoTimestamps.size());
        curveFCAffinesRatio->attach(&plotCAffinesRatio);
    QwtPlotCurve *curveUCAffinesRatio = new QwtPlotCurve();
        curveUCAffinesRatio->setTitle("Unwanted ratio");
        curveUCAffinesRatio->setPen(Qt::cyan,2);
        curveUCAffinesRatio->setRenderHint(QwtPlotItem::RenderAntialiased,true);
        curveUCAffinesRatio->setRawSamples(scaledVideoTimestamps.data(),corrUnwantedAffinesRatio.data(),scaledVideoTimestamps.size());
        curveUCAffinesRatio->attach(&plotCAffinesRatio);
    plotCAffinesRatio.resize(600,400);
    plotCAffinesRatio.show();
//

    return a.exec();
}
