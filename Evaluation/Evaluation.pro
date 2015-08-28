QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Evaluation
TEMPLATE = app

SOURCES += main.cpp

INCLUDEPATH += /usr/local/include/opencv
LIBS += -L/usr/local/lib -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgproc -lopencv_core

include(/usr/local/qwt-6.1.2/features/qwt.prf)
