#-------------------------------------------------
#
# Project created by QtCreator 2015-06-27T18:06:41
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = SensorMotionEstimation
TEMPLATE = app


SOURCES += main.cpp


INCLUDEPATH += /usr/include/libxml2

LIBS += -lxml2

include(/usr/local/qwt-6.1.2/features/qwt.prf)
