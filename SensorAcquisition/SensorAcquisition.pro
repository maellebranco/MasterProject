TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    mpu6050.cpp

HEADERS += \
    mpu6050.h

INCLUDEPATH += /usr/include/libxml2

LIBS += -lxml2
