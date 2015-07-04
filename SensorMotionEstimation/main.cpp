#include <QApplication>
#include <iostream>
#include <string.h>
#include <vector>

#include <libxml2/libxml/xmlreader.h>

#include <qwt_plot.h>
#include <qwt_plot_curve.h>
#include <qwt_legend.h>

using namespace std;

int readSensorData(const char* filePath,vector<long int> &timestamps,vector<float> &temperatures,float &sensitivityAcc,float &sensitivityGyro,
                   vector<int> &rawAccelerationsX,vector<int> &rawAccelerationsY,vector<int> &rawAccelerationsZ,
                   vector<int> &rawAngularRatesX,vector<int> &rawAngularRatesY,vector<int> &rawAngularRatesZ)
{
    xmlTextReaderPtr reader;
    int ret;
    reader = xmlReaderForFile(filePath, NULL, 0);
    if(reader != NULL)
    {
        ret = xmlTextReaderRead(reader);
        while (ret == 1)
        {
            int depth = xmlTextReaderDepth(reader);
            int type = xmlTextReaderNodeType(reader);
            if(depth==0 && type==XML_READER_TYPE_ELEMENT)
            {
                sensitivityAcc = atof((const char*)xmlTextReaderGetAttribute(reader, (const xmlChar*)"sensitivityAcc"));
                sensitivityGyro = atof((const char*)xmlTextReaderGetAttribute(reader, (const xmlChar*)"sensitivityGyro"));
            }
            if(depth==1 && type==XML_READER_TYPE_ELEMENT)
            {
                char time[18];
                strcpy(time, (const char*)xmlTextReaderGetAttributeNo(reader,0));
                strcat(time, (const char*)xmlTextReaderGetAttributeNo(reader,1));
                timestamps.push_back(atol(time));
            }
            else if(depth==2 && type==XML_READER_TYPE_ELEMENT)
            {
                const char* name = (const char*)xmlTextReaderConstName(reader);
                ret = xmlTextReaderRead(reader);

                if(strcmp(name,"temperature")==0)
                {
                    temperatures.push_back(atof((const char*)xmlTextReaderConstValue(reader)));
                }
                else
                {
                    int value = atoi((const char*)xmlTextReaderConstValue(reader));
                    if(strcmp(name,"accelerationX")==0)
                        rawAccelerationsX.push_back(value);
                    else if(strcmp(name,"accelerationY")==0)
                        rawAccelerationsY.push_back(value);
                    else if(strcmp(name,"accelerationZ")==0)
                        rawAccelerationsZ.push_back(value);
                    else if(strcmp(name,"angularRateX")==0)
                        rawAngularRatesX.push_back(value);
                    else if(strcmp(name,"angularRateY")==0)
                        rawAngularRatesY.push_back(value);
                    else if(strcmp(name,"angularRateZ")==0)
                        rawAngularRatesZ.push_back(value);
                }
            }
            ret = xmlTextReaderRead(reader);
        }
        xmlFreeTextReader(reader);
        if(ret!=0)
        {
            fprintf(stderr, "%s : failed to parse\n", filePath);
            return -1;
        }
        return 0;
    }
    else
    {
        fprintf(stderr, "Unable to open %s\n", filePath);
        return -1;
    }
}

void convertRawData(vector<long int> timestamps,float sensitivityAcc,float sensitivityGyro,
                    vector<int> rawAccelerationsX,vector<int> rawAccelerationsY,vector<int> rawAccelerationsZ,
                    vector<int> rawAngularRatesX,vector<int> rawAngularRatesY,vector<int> rawAngularRatesZ,
                    vector<double> &scaledTimestamps,vector<float> &accelerationsX,vector<float> &accelerationsY,vector<float> &accelerationsZ,
                    vector<float> &angularRatesX,vector<float> &angularRatesY,vector<float> &angularRatesZ)
{
    long int number = timestamps.back()-timestamps.front();
    int digits = 0; do { number /= 10; digits++; } while (number != 0);
    for(unsigned int i=0; i<timestamps.size(); ++i)
    {
        //cout << timestamps.at(i+1)-timestamps.at(i) << endl;
        scaledTimestamps.push_back((double)(timestamps.at(i)%(long int)pow(10,digits+1))/1000000000);
        accelerationsX.push_back(rawAccelerationsX.at(i)/sensitivityAcc);
        accelerationsY.push_back(rawAccelerationsY.at(i)/sensitivityAcc);
        accelerationsZ.push_back(rawAccelerationsZ.at(i)/sensitivityAcc);
        angularRatesX.push_back(rawAngularRatesX.at(i)/sensitivityGyro);
        angularRatesY.push_back(rawAngularRatesY.at(i)/sensitivityGyro);
        angularRatesZ.push_back(rawAngularRatesZ.at(i)/sensitivityGyro);
    }
}

void computeRollPitchYaw(vector<double> scaledTimestamps,vector<float> accelerationsX,vector<float> accelerationsY,vector<float> accelerationsZ,
                         vector<float> angularRatesX,vector<float> angularRatesY,vector<float> angularRatesZ,
                         vector<double> &roll,vector<double> &pitch,vector<double> &yaw,
                         vector<double> &rollGyro,vector<double> &pitchGyro,vector<double> &rollAcc,vector<double> &pitchAcc)
{
    for(unsigned int i=0; i<scaledTimestamps.size(); ++i)
    {
        // compute roll, pitch and yaw from gyroscope (integration)
        if(i==0)
        {
            float dt = 0.003;
            rollGyro.push_back(angularRatesX[i]*dt);
            pitchGyro.push_back(angularRatesY[i]*dt);

            roll.push_back(angularRatesX[i]*dt);
            pitch.push_back(angularRatesY[i]*dt);
            yaw.push_back(angularRatesZ[i]*dt);
        }
        else
        {
            float dt = scaledTimestamps[i]-scaledTimestamps[i-1];
            rollGyro.push_back(rollGyro[i-1]+angularRatesX[i]*dt);
            pitchGyro.push_back(pitchGyro[i-1]+angularRatesY[i]*dt);

            roll.push_back(roll[i-1]+angularRatesX[i]*dt);
            pitch.push_back(pitch[i-1]+angularRatesY[i]*dt);
            yaw.push_back(yaw[i-1]+angularRatesZ[i]*dt);
        }

        // compute roll and pitch from accelerometer (yaw not possible)
        rollAcc.push_back(atan(accelerationsY[i]/sqrt(accelerationsX[i]*accelerationsX[i]+accelerationsZ[i]*accelerationsZ[i]))*180/M_PI);
        pitchAcc.push_back(atan(-accelerationsX[i]/sqrt(accelerationsY[i]*accelerationsY[i]+accelerationsZ[i]*accelerationsZ[i]))*180/M_PI);
//        rollAcc.push_back(atan(accelerationsY[i]/accelerationsZ[i])*180/M_PI);
//        pitchAcc.push_back(atan(-accelerationsX[i]/sqrt(accelerationsY[i]*accelerationsY[i]+accelerationsZ[i]*accelerationsZ[i]))*180/M_PI);
//        rollAcc.push_back(atan2(accelerationsY[i],accelerationsZ[i])*180/M_PI);
//        pitchAcc.push_back(atan2(-accelerationsX[i],sqrt(accelerationsY[i]*accelerationsY[i]+accelerationsZ[i]*accelerationsZ[i]))*180/M_PI);

        // if acceleration magnitude near to gravity (accelerometer quite stable)
        float accMagnitude = sqrt(accelerationsX[i]*accelerationsX[i]+accelerationsY[i]*accelerationsY[i]+accelerationsZ[i]*accelerationsZ[i]);
        if(accMagnitude>0.9 && accMagnitude<1.1)
        {
            // complementary filter
            roll[i] = roll[i]*0.98 + rollAcc[i]*0.02;
            pitch[i] = pitch[i]*0.98 + pitchAcc[i]*0.02;
        }
    }
}

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    //MainWindow w;
    //w.show();

    // read raw values from xml file
    const char* filePath = "/home/maelle/Desktop/SensorTest/jitterX1000.txt";
    vector<long int> timestamps;
    vector<float> temperatures;
    float sensitivityAcc, sensitivityGyro;
    vector<int> rawAccelerationsX, rawAccelerationsY, rawAccelerationsZ;
    vector<int> rawAngularRatesX, rawAngularRatesY, rawAngularRatesZ;

    if(readSensorData(filePath,timestamps,temperatures,sensitivityAcc,sensitivityGyro,
                      rawAccelerationsX,rawAccelerationsY,rawAccelerationsZ,rawAngularRatesX,rawAngularRatesY,rawAngularRatesZ)<0)
        return -1;

    // convert raw values
    vector<double> scaledTimestamps;   // in s (and without unchanged upper scales)
    vector<float> accelerationsX, accelerationsY, accelerationsZ;
    vector<float> angularRatesX, angularRatesY, angularRatesZ;

    convertRawData(timestamps,sensitivityAcc,sensitivityGyro,
                   rawAccelerationsX,rawAccelerationsY,rawAccelerationsZ,rawAngularRatesX,rawAngularRatesY,rawAngularRatesZ,
                   scaledTimestamps,accelerationsX,accelerationsY,accelerationsZ,angularRatesX,angularRatesY,angularRatesZ);
/*
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
    vector<double> d_accelerationsX(accelerationsX.begin(),accelerationsX.end());
    curveAccX->setRawSamples(scaledTimestamps.data(),d_accelerationsX.data(),scaledTimestamps.size());
    curveAccX->attach(&plotAccelerations);
    QwtPlotCurve *curveAccY = new QwtPlotCurve();
    curveAccY->setTitle("Y");
    curveAccY->setPen(Qt::red,2);
    curveAccY->setRenderHint(QwtPlotItem::RenderAntialiased,true);
    vector<double> d_accelerationsY(accelerationsY.begin(),accelerationsY.end());
    curveAccY->setRawSamples(scaledTimestamps.data(),d_accelerationsY.data(),scaledTimestamps.size());
    curveAccY->attach(&plotAccelerations);
    QwtPlotCurve *curveAccZ = new QwtPlotCurve();
    curveAccZ->setTitle("Z");
    curveAccZ->setPen(Qt::green,2);
    curveAccZ->setRenderHint(QwtPlotItem::RenderAntialiased,true);
    vector<double> d_accelerationsZ(accelerationsZ.begin(),accelerationsZ.end());
    curveAccZ->setRawSamples(scaledTimestamps.data(),d_accelerationsZ.data(),scaledTimestamps.size());
    curveAccZ->attach(&plotAccelerations);
    plotAccelerations.resize(600,400);
    plotAccelerations.show();

    QwtPlot plotAngularRates;
    plotAngularRates.setTitle("Angular Rates");
    plotAngularRates.setCanvasBackground(Qt::white);
    plotAngularRates.insertLegend(new QwtLegend());
    plotAngularRates.setAxisTitle(QwtPlot::yLeft,"Angular Rate (°/s)");
    plotAngularRates.setAxisTitle(QwtPlot::xBottom,"Timestamp (s)");
    QwtPlotCurve *curveAngX = new QwtPlotCurve();
    curveAngX->setTitle("X");
    curveAngX->setPen(Qt::blue,2);
    curveAngX->setRenderHint(QwtPlotItem::RenderAntialiased,true);
    vector<double> d_angularRatesX(angularRatesX.begin(),angularRatesX.end());
    curveAngX->setRawSamples(scaledTimestamps.data(),d_angularRatesX.data(),scaledTimestamps.size());
    curveAngX->attach(&plotAngularRates);
    QwtPlotCurve *curveAngY = new QwtPlotCurve();
    curveAngY->setTitle("Y");
    curveAngY->setPen(Qt::red,2);
    curveAngY->setRenderHint(QwtPlotItem::RenderAntialiased,true);
    vector<double> d_angularRatesY(angularRatesY.begin(),angularRatesY.end());
    curveAngY->setRawSamples(scaledTimestamps.data(),d_angularRatesY.data(),scaledTimestamps.size());
    curveAngY->attach(&plotAngularRates);
    QwtPlotCurve *curveAngZ = new QwtPlotCurve();
    curveAngZ->setTitle("Z");
    curveAngZ->setPen(Qt::green,2);
    curveAngZ->setRenderHint(QwtPlotItem::RenderAntialiased,true);
    vector<double> d_angularRatesZ(angularRatesZ.begin(),angularRatesZ.end());
    curveAngZ->setRawSamples(scaledTimestamps.data(),d_angularRatesZ.data(),scaledTimestamps.size());
    curveAngZ->attach(&plotAngularRates);
    plotAngularRates.resize(600,400);
    plotAngularRates.show();
*/
    // compute roll, pitch and yaw
    vector<double> roll,pitch,yaw;
    vector<double> rollGyro,pitchGyro;
    vector<double> rollAcc,pitchAcc;

    computeRollPitchYaw(scaledTimestamps,accelerationsX,accelerationsY,accelerationsZ,angularRatesX,angularRatesY,angularRatesZ,
                        roll,pitch,yaw,rollGyro,pitchGyro,rollAcc,pitchAcc);
/*
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
*/
    return a.exec();
}
