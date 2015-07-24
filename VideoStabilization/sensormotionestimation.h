#ifndef SENSORMOTIONESTIMATION_H
#define SENSORMOTIONESTIMATION_H

#include <vector>

using::std::vector;

class SensorMotionEstimation
{
public:
    // read raw data from xml file
    static int readSensorData(const char* filePath,vector<long int> &timestamps,vector<float> &temperatures,float &sensitivityAcc,float &sensitivityGyro,
                       vector<int> &rawAccelerationsX,vector<int> &rawAccelerationsY,vector<int> &rawAccelerationsZ,
                       vector<int> &rawAngularRatesX,vector<int> &rawAngularRatesY,vector<int> &rawAngularRatesZ);

    // convert raw data to scaled values
    static void convertRawData(vector<long int> timestamps, float sensitivityAcc, float sensitivityGyro,
                        vector<int> rawAccelerationsX, vector<int> rawAccelerationsY, vector<int> rawAccelerationsZ,
                        vector<int> rawAngularRatesX, vector<int> rawAngularRatesY, vector<int> rawAngularRatesZ,
                        vector<double> &scaledTimestamps, vector<double> &accelerationsX, vector<double> &accelerationsY, vector<double> &accelerationsZ,
                        vector<double> &angularRatesX, vector<double> &angularRatesY, vector<double> &angularRatesZ);

    // compute roll, pitch and yaw using complementary filter
    static void computeRollPitchYaw(vector<double> scaledTimestamps, vector<double> accelerationsX, vector<double> accelerationsY, vector<double> accelerationsZ,
                             vector<double> angularRatesX, vector<double> angularRatesY, vector<double> angularRatesZ,
                             vector<double> &roll, vector<double> &pitch, vector<double> &yaw,
                             vector<double> &rollGyro, vector<double> &pitchGyro, vector<double> &rollAcc, vector<double> &pitchAcc);
};

#endif // SENSORMOTIONESTIMATION_H
