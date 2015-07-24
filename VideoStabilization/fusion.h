#ifndef FUSION_H
#define FUSION_H

//#include <string>
#include <vector>
#include <map>
//#include <opencv2/core.hpp>

//using std::string;
using std::vector;
using std::map;
//using cv::Mat;

class Fusion
{
public:
    // read video stimestamps from xml file
    static int videoTimestamps(const char *filePath, vector<long int> &videoTimestamps);

    // print frame capture and sensor measure in timestamp order
    static void checkSynchro(vector<long int> timestamps, vector<long int> videoTimestamps);

    // build relation map between video frames and sensor data for synchronisation (Map<indexSensorData;indexFrame>)
    static void buildSynchroMap(vector<long int> timestamps, vector<long int> videoTimestamps, map<int,int> &synchroMap);

    // creation of vectors for sensor data per frame
    static void fusionSensorData(map<int,int> synchroMap, vector<double> angularRatesX, vector<double> angularRatesY, vector<double> angularRatesZ,
                                 vector<double> &syncAngularRatesX, vector<double> &syncAngularRatesY, vector<double> &syncAngularRatesZ);

    // no motion filter (if gyroscope still, correct translation to zero) > only for translations
    static void noMotionDetection(vector<double> translationsX, vector<double> translationsY,
                                  vector<double> syncAngularRatesX, vector<double> syncAngularRatesY, vector<double> syncAngularRatesZ,
                                  vector<double> &corrTranslationsX, vector<double> &corrTranslationsY);

    // no motion filter (if gyroscope still, correct translation to zero)
    static void noMotionDetection(vector<double> affinesA, vector<double> affinesB, vector<double> affinesC, vector<double> affinesD,
                                  vector<double> affinesTx, vector<double> affinesTy, vector<double> affinesSkew, vector<double> affinesRatio,
                                  vector<double> syncAngularRatesX, vector<double> syncAngularRatesY, vector<double> syncAngularRatesZ,
                                  vector<double> &corrAffinesA, vector<double> &corrAffinesB, vector<double> &corrAffinesC, vector<double> &corrAffinesD,
                                  vector<double> &corrAffinesTx, vector<double> &corrAffinesTy, vector<double> &corrAffinesSkew, vector<double> &corrAffinesRatio);

    // error calculation between video processing and sensor motion estimation
    static void errorMotions(vector<double> translationsX, vector<double> translationsY, vector<double> syncAngularRatesY, vector<double> syncAngularRatesZ,
                             vector<double> &errorsH, vector<double> &errorsV, double &mseH, double &mseV);

    // correction of video processing motion estimation using error from gyroscope data
    static void correctionMotions(vector<double> translationsX, vector<double> translationsY, vector<double> errorsH, vector<double> errorsV, double mseH, double mseV,
                                   vector<double> &corrTranslationsX, vector<double> &corrTranslationsY);
};

#endif // FUSION_H
