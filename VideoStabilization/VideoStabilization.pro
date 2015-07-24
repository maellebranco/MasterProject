#-------------------------------------------------
#
# Project created by QtCreator 2015-06-19T12:06:46
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = VideoStabilization
TEMPLATE = app


SOURCES += main.cpp \
        videomotionestimation.cpp \
        trajectory.cpp \
        sensormotionestimation.cpp \
        stabilization.cpp \
        fusion.cpp

HEADERS += videomotionestimation.h \
        trajectory.h \
        sensormotionestimation.h \
        stabilization.h \
        fusion.h

CONFIG += c++11


INCLUDEPATH += /usr/local/include/opencv
LIBS += -L/usr/local/lib -lopencv_shape -lopencv_stitching -lopencv_objdetect -lopencv_superres -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_core -lopencv_hal -lopencv_xfeatures2d


INCLUDEPATH += /home/maelle/coin-Clp/include/coin
LIBS += -L/home/maelle/coin-Clp/lib -lCoinUtils -lz -lm -lClp -lClpSolver


INCLUDEPATH += /usr/include/libxml2
LIBS += -lxml2


include(/usr/local/qwt-6.1.2/features/qwt.prf)
