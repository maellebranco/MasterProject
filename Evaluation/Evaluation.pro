#-------------------------------------------------
#
# Project created by QtCreator 2015-07-18T15:16:04
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Evaluation
TEMPLATE = app

SOURCES += main.cpp

INCLUDEPATH += /usr/local/include/opencv
LIBS += -L/usr/local/lib -lopencv_shape -lopencv_stitching -lopencv_objdetect -lopencv_superres -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_core -lopencv_hal -lopencv_xfeatures2d

include(/usr/local/qwt-6.1.2/features/qwt.prf)
