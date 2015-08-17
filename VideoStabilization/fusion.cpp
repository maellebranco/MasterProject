#include "fusion.h"

#include <iostream>
#include <string.h>
#include <vector>
#include <math.h>

#include <libxml2/libxml/xmlreader.h>

using namespace std;

// read video stimestamps from xml file
int Fusion::videoTimestamps(const char* filePath, vector<long int> &videoTimestamps)
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
            if(depth==1 && type==XML_READER_TYPE_ELEMENT)
            {
                char time[18];
                strcpy(time, (const char*)xmlTextReaderGetAttributeNo(reader,0));
                strcat(time, (const char*)xmlTextReaderGetAttributeNo(reader,1));
                videoTimestamps.push_back(atol(time)-100000000);
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

// print frame capture and sensor measure in timestamp order
void Fusion::checkSynchro(vector<long int> timestamps, vector<long int> videoTimestamps)
{
    unsigned int i=0;
    for(vector<long int>::iterator it=timestamps.begin(); it!=timestamps.end(); ++it)
    {
        if(i<videoTimestamps.size() && videoTimestamps.at(i)<(*it))
        {
            cout << "frame " << i << ":   " << videoTimestamps.at(i) << endl;
            ++i;
        }
        cout << "sensor " << distance(timestamps.begin(),it) << ": " << (*it) << endl;
    }
}

// build relation map between video frames and sensor data for synchronisation (Map<indexSensorData;indexFrame>)
void Fusion::buildSynchroMap(vector<long int> timestamps, vector<long int> videoTimestamps, map<int,int> &synchroMap)
{
    unsigned int i,j;
    for(i=0, j=0; j<timestamps.size(); ++j)
    {
        if(i+1<videoTimestamps.size() && videoTimestamps.at(i)<timestamps.at(j))
        {
            if(videoTimestamps.at(i+1)<timestamps.at(j)) ++i;
            synchroMap.insert(pair<int,int>(j,i));
        }
        else if(i+1==videoTimestamps.size() && timestamps.at(j)<(videoTimestamps.at(i)+33000000))
        {
            synchroMap.insert(pair<int,int>(j,i));
        }
        else
            synchroMap.insert(pair<int,int>(j,-1));
    }
    //for(map<int,int>::const_iterator it=synchroMap.begin(); it!=synchroMap.end(); ++it) cout << it->first << " " << it->second << endl;
}

// creation of vectors for sensor data per frame (average of sensor data during frame capture)
void Fusion::fusionSensorData(map<int,int> synchroMap, vector<double> angularRatesX, vector<double> angularRatesY, vector<double> angularRatesZ,
                              vector<double> &syncAngularRatesX, vector<double> &syncAngularRatesY, vector<double> &syncAngularRatesZ)
{
    int frame = 0;
    int nb = 1;
    double angularRateX = 0;
    double angularRateY = 0;
    double angularRateZ = 0;
    syncAngularRatesX.push_back(angularRateX);  // sensor data during frame i for video motion compute frame i+1
    syncAngularRatesY.push_back(angularRateY);
    syncAngularRatesZ.push_back(angularRateZ);
    for(map<int,int>::const_iterator it=synchroMap.begin(); it!=synchroMap.end(); ++it)
    {
        if(it->second==frame)
        {
            angularRateX += angularRatesX.at(it->first);
            angularRateY += angularRatesY.at(it->first);
            angularRateZ += angularRatesZ.at(it->first);
            ++nb;
        }
        else if((it->second-frame)==1)
        {
            syncAngularRatesX.push_back(angularRateX/nb);
            syncAngularRatesY.push_back(angularRateY/nb);
            syncAngularRatesZ.push_back(angularRateZ/nb);
            angularRateX = angularRatesX.at(it->first);
            angularRateY = angularRatesY.at(it->first);
            angularRateZ = angularRatesZ.at(it->first);
            nb = 1;
            ++frame;
        }
        /*else if((it->second-frame)!=-1)
        {
            syncAngularRatesX.push_back(angularRateX/nb);
            syncAngularRatesY.push_back(angularRateY/nb);
            syncAngularRatesZ.push_back(angularRateZ/nb);
            frame = 0;
        }*/
    }
}

// no motion filter (if gyroscope still, correct translation to zero)
void Fusion::noMotionDetection(vector<double> translationsX, vector<double> translationsY,
                              vector<double> syncAngularRatesX, vector<double> syncAngularRatesY, vector<double> syncAngularRatesZ,
                              vector<double> &corrTranslationsX, vector<double> &corrTranslationsY)
{
    float limit=0.5;
    for(unsigned int i=0; i<syncAngularRatesX.size(); ++i)
    {
        if(syncAngularRatesZ.at(i)>limit || syncAngularRatesZ.at(i)<-limit)
            corrTranslationsX.push_back(translationsX.at(i));
        else // no horizontal motion
            corrTranslationsX.push_back(0);

        if(syncAngularRatesY.at(i)>limit || syncAngularRatesY.at(i)<-limit)
            corrTranslationsY.push_back(translationsY.at(i));
        else // no vertical motion
            corrTranslationsY.push_back(0);
    }
}

// no motion filter (if gyroscope still, correct translation to zero)
void Fusion::noMotionDetection(vector<double> affinesA, vector<double> affinesB, vector<double> affinesC, vector<double> affinesD,
                               vector<double> affinesTx, vector<double> affinesTy, vector<double> affinesSkew, vector<double> affinesRatio,
                               vector<double> syncAngularRatesX, vector<double> syncAngularRatesY, vector<double> syncAngularRatesZ,
                               vector<double> &corrAffinesA, vector<double> &corrAffinesB, vector<double> &corrAffinesC, vector<double> &corrAffinesD,
                               vector<double> &corrAffinesTx, vector<double> &corrAffinesTy, vector<double> &corrAffinesSkew, vector<double> &corrAffinesRatio)
{
    float limit=2;
    for(unsigned int i=0; i<syncAngularRatesX.size(); ++i)
    {
        if(syncAngularRatesX.at(i)>limit || syncAngularRatesX.at(i)<-limit
                || syncAngularRatesY.at(i)>limit || syncAngularRatesY.at(i)<-limit
                        || syncAngularRatesZ.at(i)>limit || syncAngularRatesZ.at(i)<-limit)
        {
            corrAffinesA.push_back(affinesA.at(i));
            corrAffinesB.push_back(affinesB.at(i));
            corrAffinesC.push_back(affinesC.at(i));
            corrAffinesD.push_back(affinesD.at(i));
        }
        else // no motion
        {
            corrAffinesA.push_back(0);
            corrAffinesB.push_back(0);
            corrAffinesC.push_back(0);
            corrAffinesD.push_back(0);
        }

        if(syncAngularRatesY.at(i)>limit || syncAngularRatesY.at(i)<-limit)
        {
            corrAffinesTy.push_back(affinesTy.at(i));
            corrAffinesRatio.push_back(affinesRatio.at(i));
        }
        else // no vertical motion
        {
            corrAffinesTy.push_back(0);
            corrAffinesRatio.push_back(0);
        }

        if(syncAngularRatesZ.at(i)>limit || syncAngularRatesZ.at(i)<-limit)
        {
            corrAffinesTx.push_back(affinesTx.at(i));
            corrAffinesSkew.push_back(affinesSkew.at(i));
        }
        else // no horizontal motion
        {
            corrAffinesTx.push_back(0);
            corrAffinesSkew.push_back(0);
        }
    }
}

// squared error calculation between video processing and sensor motion estimation
void Fusion::errorMotions(vector<double> translationsX, vector<double> translationsY, vector<double> syncAngularRatesY, vector<double> syncAngularRatesZ,
                          vector<double> &errorsH, vector<double> &errorsV, double &mseH, double &mseV)
{
    for(unsigned int i=0; i<translationsX.size(); ++i)
    {
        double errorH = translationsX.at(i)-syncAngularRatesZ.at(i);
        double errorV = -translationsY.at(i)-syncAngularRatesY.at(i);
        errorsH.push_back(errorH);
        errorsV.push_back(errorV);
        mseH += errorH*errorH;
        mseV += errorV*errorV;
    }
    mseH /= errorsH.size();
    mseV /= errorsV.size();
    cout << mseH << " " << mseV << endl;
}

// correction of video processing motion estimation using error from gyroscope data
void Fusion::correctionMotions(vector<double> translationsX, vector<double> translationsY, vector<double> errorsH, vector<double> errorsV, double mseH, double mseV,
                               vector<double> &corrTranslationsX, vector<double> &corrTranslationsY)
{
    int margin = 42;
    for(unsigned int i=0; i<translationsX.size(); ++i)
    {
        double valueH = 0, valueV = 0;
        if(errorsH.at(i)>sqrt(mseH+margin))
            valueH = errorsH.at(i)-sqrt(mseH+margin);
        else if(-errorsH.at(i)>sqrt(mseH+margin))
            valueH = errorsH.at(i)+sqrt(mseH+margin);
        corrTranslationsX.push_back(translationsX.at(i)-valueH);

        if(errorsV.at(i)>sqrt(mseV+margin))
            valueV = errorsV.at(i)-sqrt(mseV+margin);
        else if(-errorsV.at(i)>sqrt(mseV+margin))
            valueV = errorsV.at(i)+sqrt(mseV+margin);
        corrTranslationsY.push_back(translationsY.at(i)+valueV);
    }
}
