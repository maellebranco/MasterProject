#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    int nbFrames = 500; // parameter: number of frames acquired

    VideoCapture cap(0);    // open the video camera

    if (!cap.isOpened())    // if not success, exit program
    {
        cout << "ERROR: Cannot open the video camera" << endl;
        return -1;
    }

    /*
    cap.set(CV_CAP_PROP_FRAME_WIDTH,1920);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
    */

    double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);   // get the width of frames of the video
    double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); // get the width of frames of the video
    double dFourcc = cap.get(CAP_PROP_FOURCC);          // get the format code of the video
    double fps = cap.get(CAP_PROP_FPS);                 // get framerate of the video

    Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));
    int fourcc = static_cast<int>(dFourcc);

    cout << "Camera parameters: " << frameSize << ", " << fourcc << ", " << fps << endl;

    VideoWriter oVideoWriter ("/Videos/Video.yuv", fourcc, fps, frameSize, true);  // initialize the VideoWriter object

    if (!oVideoWriter.isOpened()) // if not success, exit program
    {
        cout << "ERROR: Failed to write the video" << endl;
        return -1;
    }

    for(int i=0; i<nbFrames; i++)    // non stop
    {
        Mat frame;

        bool bSuccess = cap.read(frame);    // read a new frame from camera

        if (!bSuccess)  // if not success, break loop
        {
            cout << "ERROR: Cannot read a frame from camera" << endl;
            break;
        }

        oVideoWriter.write(frame);  // write the frame into the file
    }

    return 0;
}
