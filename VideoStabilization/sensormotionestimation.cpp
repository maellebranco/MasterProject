#include "sensormotionestimation.h"

#include <iostream>
#include <string.h>
#include <vector>

#include <libxml2/libxml/xmlreader.h>

#include <qwt_plot.h>
#include <qwt_plot_curve.h>
#include <qwt_legend.h>

using namespace std;

// read raw data from xml file
int SensorMotionEstimation::readSensorData(const char* filePath,vector<long int> &timestamps,vector<float> &temperatures,float &sensitivityAcc,float &sensitivityGyro,
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

// convert raw data to scaled values
void SensorMotionEstimation::convertRawData(vector<long int> timestamps,float sensitivityAcc,float sensitivityGyro,
                                            vector<int> rawAccelerationsX,vector<int> rawAccelerationsY,vector<int> rawAccelerationsZ,
                                            vector<int> rawAngularRatesX,vector<int> rawAngularRatesY,vector<int> rawAngularRatesZ,
                                            vector<double> &scaledTimestamps,vector<double> &accelerationsX,vector<double> &accelerationsY,vector<double> &accelerationsZ,
                                            vector<double> &angularRatesX,vector<double> &angularRatesY,vector<double> &angularRatesZ)
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

// compute roll, pitch and yaw using complementary filter
void SensorMotionEstimation::computeRollPitchYaw(vector<double> scaledTimestamps, vector<double> accelerationsX, vector<double> accelerationsY, vector<double> accelerationsZ,
                                                 vector<double> angularRatesX, vector<double> angularRatesY, vector<double> angularRatesZ,
                                                 vector<double> &roll, vector<double> &pitch, vector<double> &yaw,
                                                 vector<double> &rollGyro, vector<double> &pitchGyro, vector<double> &rollAcc, vector<double> &pitchAcc)
{
    for(unsigned int i=0; i<scaledTimestamps.size(); ++i)
    {
        // compute roll, pitch and yaw from gyroscope (integration)
        if(i==0)
        {
            double dt = 0.003;
            rollGyro.push_back(angularRatesX[i]*dt);
            pitchGyro.push_back(angularRatesY[i]*dt);

            roll.push_back(angularRatesX[i]*dt);
            pitch.push_back(angularRatesY[i]*dt);
            yaw.push_back(angularRatesZ[i]*dt);
        }
        else
        {
            double dt = scaledTimestamps[i]-scaledTimestamps[i-1];
            rollGyro.push_back(rollGyro[i-1]+angularRatesX[i]*dt);
            pitchGyro.push_back(pitchGyro[i-1]+angularRatesY[i]*dt);

            roll.push_back(roll[i-1]+angularRatesX[i]*dt);
            pitch.push_back(pitch[i-1]+angularRatesY[i]*dt);
            yaw.push_back(yaw[i-1]+angularRatesZ[i]*dt);
        }

        // compute roll and pitch from accelerometer (yaw not possible)
        rollAcc.push_back(atan(accelerationsY[i]/sqrt(accelerationsX[i]*accelerationsX[i]+accelerationsZ[i]*accelerationsZ[i]))*180/M_PI);
        pitchAcc.push_back(atan(-accelerationsX[i]/sqrt(accelerationsY[i]*accelerationsY[i]+accelerationsZ[i]*accelerationsZ[i]))*180/M_PI);

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
