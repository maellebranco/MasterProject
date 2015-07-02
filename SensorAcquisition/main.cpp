#include "mpu6050.h"

#include <iostream>
#include <unistd.h> // for usleep
#include <time.h>   // for timestamp

#include <libxml2/libxml/xmlwriter.h>

using namespace std;

struct timespec timestamp;

int main(int argc, char *argv[])
{
    cout << "Start sensor connection" << endl;
    MPU6050 sensor(1,0x68);

    sensor.initialize();
    sensor.setDigitalLowPassFilter(MPU6050::LP_98HZ);

    sensor.offsetEstimation(125); //    about 1s

    xmlTextWriterPtr writer = xmlNewTextWriterFilename("test.txt", 0); // create a new XmlWriter, with no compression
    if(writer == NULL)
    {
        printf("Error creating the xml writer\n");
        return -1;
    }
    // start the document with xml default version, UTF-8 encoding and default standalone declaration
    if(xmlTextWriterStartDocument(writer, NULL, "UTF-8", NULL)<0)
    {
        printf("Error at xmlTextWriterStartDocument\n");
        return -1;
    }
    xmlTextWriterStartElement(writer, (xmlChar*) "acquisition");    // start the root element
    xmlTextWriterWriteFormatAttribute(writer, (xmlChar*) "sensitivityAcc","%d", sensor.getSensitivityAcc());
    xmlTextWriterWriteFormatAttribute(writer, (xmlChar*) "sensitivityGyro","%.1f", sensor.getSensitivityGyro());

    int nbSample=0;
    if(argc>0) nbSample = atoi(argv[1]);
    for(int i=0; i<nbSample; ++i)
    {
        clock_gettime(CLOCK_REALTIME, &timestamp);

        sensor.readAll();
        //sensor.displayAll();

        xmlTextWriterStartElement(writer, (xmlChar*) "measure");
        xmlTextWriterWriteFormatAttribute(writer, (xmlChar*) "timestamp1","%ld", timestamp.tv_sec);
        xmlTextWriterWriteFormatAttribute(writer, (xmlChar*) "timestamp2","%09ld", timestamp.tv_nsec);
        xmlTextWriterWriteFormatElement(writer, (xmlChar*) "accelerationX","%hd", sensor.getAccelerationX());
        xmlTextWriterWriteFormatElement(writer, (xmlChar*) "accelerationY","%hd", sensor.getAccelerationY());
        xmlTextWriterWriteFormatElement(writer, (xmlChar*) "accelerationZ","%hd", sensor.getAccelerationZ());
        xmlTextWriterWriteFormatElement(writer, (xmlChar*) "angularRateX","%hd", sensor.getAngularRateX());
        xmlTextWriterWriteFormatElement(writer, (xmlChar*) "angularRateY","%hd", sensor.getAngularRateY());
        xmlTextWriterWriteFormatElement(writer, (xmlChar*) "angularRateZ","%hd", sensor.getAngularRateZ());
        xmlTextWriterWriteFormatElement(writer, (xmlChar*) "temperature","%.2f", sensor.getTemperatureCelcius());
        xmlTextWriterEndElement(writer);

        usleep(5*1000); // for sample time around 8ms
    }

    if(xmlTextWriterEndDocument(writer)<0)    // close all remaining elements and end the document
    {
        printf("Error at xmlTextWriterEndDocument\n");
        return -1;
    }

    xmlFreeTextWriter(writer);

    return 0;
}
