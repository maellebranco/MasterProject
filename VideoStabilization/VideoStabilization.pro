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
LIBS += -L/usr/local/lib -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_core -lopencv_xfeatures2d


INCLUDEPATH += /home/maelle/coin-Clp/include/coin
LIBS += -L/home/maelle/coin-Clp/lib -lCoinUtils -lz -lm -lClp -lClpSolver


INCLUDEPATH += /usr/include/libxml2
LIBS += -lxml2


include(/usr/local/qwt-6.1.2/features/qwt.prf)
