#include "mpu6050.h"

#include <iostream>
#include <sstream>
#include <fcntl.h>
#include <iomanip>
#include <stdio.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>

#include <math.h>

using namespace std;

// general methods

MPU6050::MPU6050(unsigned int bus, unsigned int address)
{
    this->bus = bus;
    this->address = address;
    this->file=-1;
    this->openI2C();

    this->registers = NULL;

    this->accelerationX = 0;
    this->accelerationY = 0;
    this->accelerationZ = 0;
    this->temperature = 0;
    this->temperatureCelcius = 0;
    this->angularRateX = 0;
    this->angularRateY = 0;
    this->angularRateZ = 0;

    this->offsetX = 118;
    this->offsetY = -15;
    this->offsetZ = 115;

    this->rangeAcc = PLUSMINUS_2_G;
    this->sensitivityAcc = FS_2G;
    this->rangeGyro = PLUSMINUS_250_DPS;
    this->sensitivityGyro = FS_250;
}

int MPU6050::openI2C()
{
   string name;
   if(this->bus==0) name = "/dev/i2c-0";
   else name = "/dev/i2c-1";

   if((this->file=open(name.c_str(),O_RDWR))<0)
   {
      perror("I2C: failed to open the bus\n");
      return 1;
   }
   if(ioctl(this->file,I2C_SLAVE,this->address)<0)
   {
      perror("I2C: Failed to connect to the device\n");
      return 1;
   }
   return 0;
}

int MPU6050::writeRegister(unsigned int registerAddress, unsigned char value)
{
   unsigned char buffer[2];
   buffer[0] = registerAddress;
   buffer[1] = value;
   if(write(this->file,buffer,2)!=2)
   {
      perror("I2C: Failed write to the device\n");
      return 1;
   }
   return 0;
}

int MPU6050::writeAddress(unsigned char addressValue)
{
   unsigned char buffer[1];
   buffer[0] = addressValue;
   if(write(this->file,buffer,1)!=1)
   {
      perror("I2C: Failed to write to the device\n");
      return 1;
   }
   return 0;
}

unsigned char MPU6050::readRegister(unsigned int registerAddress)
{
   this->writeAddress(registerAddress);
   unsigned char buffer[1];
   if(read(this->file,buffer,1)!=1)
   {
      perror("I2C: Failed to read in the value.\n");
      return 1;
   }
   return buffer[0];
}

unsigned char* MPU6050::readRegisters(unsigned int number, unsigned int fromAddress)
{
    this->writeAddress(fromAddress);
    unsigned char* data = new unsigned char[number];
    if(read(this->file,data,number)!=(int)number)
    {
       perror("IC2: Failed to read in the full buffer.\n");
       return NULL;
    }
    return data;
}

void MPU6050::closeI2C()
{
    close(this->file);
    this->file = -1;
}

MPU6050::~MPU6050()
{
    if(file!=-1) this->closeI2C();
}

// specific methods

void MPU6050::initialize()
{
    // sleep mode off (bit 6 to 0) and cloked by X gyro (bits [2:0])
    this->writeRegister(REG_PWR_MGMT_1,0b00000001);
    // set gyroscope full range to +/- 250°/s (bits [4:3])
    this->writeRegister(REG_GYRO_CONFIG,PLUSMINUS_250_DPS);
    this->rangeGyro = PLUSMINUS_250_DPS;
    this->sensitivityGyro = FS_250;
    // set accelerometer full range to +/- 4g (bits [4:3]) and High Pass Filter disabled (bits [2:0])
    this->writeRegister(REG_ACCEL_CONFIG,PLUSMINUS_4_G);
    this->rangeAcc = PLUSMINUS_4_G;
    this->sensitivityAcc = FS_4G;
    usleep(30*1000); // wait 30ms for sensors start-up
}

void MPU6050::setDigitalLowPassFilter(LPFILTER filterValue)
{
    this->writeRegister(REG_CONFIG,filterValue);
}

bool MPU6050::testConnection()
{
    return (this->readRegister(REG_WHO_AM_I) == this->address);
}

int MPU6050::readFullSensorState()
{
    this->registers = this->readRegisters(BUFFER_SIZE, 0x00);
    //if(this->registers[REG_WHO_AM_I]!=this->address)  // false each time
    //{
    //    cout << "MPU6050: Failure Condition - Sensor ID not Verified" << endl;
    //    return 1;
    //}
    return 0;
}

float MPU6050::readSampleRate()
{
    unsigned char divider = this->readRegister(REG_SMPLRT_DIV);
    unsigned char conf_DLPF = this->readRegister(REG_CONFIG);
    conf_DLPF = conf_DLPF & 0b00000111;
    float sampleRate;
    if(conf_DLPF==0 || conf_DLPF==7)
        sampleRate = 8.0/(1.0+(float)divider);
    else
        sampleRate = 1.0/(1.0+(float)divider);
    return sampleRate;
}

short MPU6050::readAccelerationX()
{
    this->registers = this->readRegisters(2,REG_ACCEL_XOUT);
    short acc = (((short)registers[0])<<8) | registers[1];
    //acc = ~acc + 1; // 2's complement
    return acc;

}

short MPU6050::readAccelerationY()
{
    this->registers = this->readRegisters(2,REG_ACCEL_YOUT);
    short acc = (((short)registers[0])<<8) | registers[1];
    //acc = ~acc + 1; // 2's complement
    return acc;

}

short MPU6050::readAccelerationZ()
{
    this->registers = this->readRegisters(2,REG_ACCEL_ZOUT);
    short acc = (((short)registers[0])<<8) | registers[1];
    //acc = ~acc + 1; // 2's complement
    return acc;

}

short MPU6050::readAngularRateX()
{
    this->registers = this->readRegisters(2,REG_GYRO_XOUT);
    short ang = (((short)registers[0])<<8) | registers[1];
    //ang = ~ang + 1; // 2's complement
    return ang;
}

short MPU6050::readAngularRateY()
{
    this->registers = this->readRegisters(2,REG_GYRO_YOUT);
    short ang = (((short)registers[0])<<8) | registers[1];
    //ang = ~ang + 1; // 2's complement
    return ang;
}

short MPU6050::readAngularRateZ()
{
    this->registers = this->readRegisters(2,REG_GYRO_ZOUT);
    short ang = (((short)registers[0])<<8) | registers[1];
    //ang = ~ang + 1; // 2's complement
    return ang;
}

int MPU6050::readAccelerations()
{
    this->registers = this->readRegisters(6,REG_ACCEL_XOUT);
    this->accelerationX = (((short)registers[0])<<8) | registers[1];
    this->accelerationY = (((short)registers[2])<<8) | registers[3];
    this->accelerationZ = (((short)registers[4])<<8) | registers[5];
    //this-> accelerationX = ~accelerationX + 1; // 2's complement
    //this->accelerationY = ~accelerationY + 1;
    //this->accelerationZ = ~accelerationZ + 1;
    if(!this->testConnection())
    {
        cout << "MPU6050: Failure Condition - Sensor ID not Verified" << endl;
        return 1;
    }
    return 0;
}

int MPU6050::readAngularRates()
{
    this->registers = this->readRegisters(6,REG_GYRO_XOUT);
    this->angularRateX = (((short)registers[0])<<8) | registers[1];
    this->angularRateY = (((short)registers[2])<<8) | registers[3];
    this->angularRateZ = (((short)registers[4])<<8) | registers[5];
    //this->angularRateX = ~angularRateX + 1; // 2's complement
    //this->angularRateY = ~angularRateY + 1;
    //this->angularRateZ = ~angularRateZ + 1;
    if(!this->testConnection())
    {
        cout << "MPU6050: Failure Condition - Sensor ID not Verified" << endl;
        return 1;
    }
    return 0;
}

int MPU6050::readTemperature()
{
    this->registers = this->readRegisters(2,REG_TEMP_OUT);
    this->temperature = (((short)registers[0])<<8) | registers[1];
    this->temperatureCelcius = temperature/340 + 36.53;
    if(!this->testConnection())
    {
        cout << "MPU6050: Failure Condition - Sensor ID not Verified" << endl;
        return 1;
    }
    return 0;
}

int MPU6050::readAll()
{
    this->registers = this->readRegisters(14,REG_ACCEL_XOUT);
    this->accelerationX = (((short)registers[0])<<8) | registers[1];
    this->accelerationY = (((short)registers[2])<<8) | registers[3];
    this->accelerationZ = (((short)registers[4])<<8) | registers[5];
    this->temperature = (((short)registers[6])<<8) | registers[7];
    this->angularRateX = (((short)registers[8])<<8) | registers[9];
    this->angularRateY = (((short)registers[10])<<8) | registers[11];
    this->angularRateZ = (((short)registers[12])<<8) | registers[13];
    //this-> accelerationX = ~accelerationX + 1; // 2's complement
    //this->accelerationY = ~accelerationY + 1;
    //this->accelerationZ = ~accelerationZ + 1;
    //this->angularRateX = ~angularRateX + 1;
    //this->angularRateY = ~angularRateY + 1;
    //this->angularRateZ = ~angularRateZ + 1;
    this->temperatureCelcius = temperature/340 + 36.53;
    if(!this->testConnection())
    {
        cout << "MPU6050: Failure Condition - Sensor ID not Verified" << endl;
        return 1;
    }
    return 0;
}

int MPU6050::offsetEstimation(int nbSample)
{
    float totalX=0, totalY=0, totalZ=0;
    for(int i=0; i<nbSample; ++i)
    {
        if(this->readAngularRates()) return 1;
        else
        {
            totalX += this->angularRateX;
            totalY += this->angularRateY;
            totalZ += this->angularRateZ;
            usleep(8*1000);
        }
    }
    this->offsetX = totalX/nbSample;
    this->offsetY = totalY/nbSample;
    this->offsetZ = totalZ/nbSample;
    cout << "Offset X: " << offsetX << endl << "Offset Y: " << offsetY << endl << "Offset Z: " << offsetZ << endl;
    return 0;
}

void MPU6050::displayAccelerations()
{
    float sensitivity = (float)this->sensitivityAcc;
    cout << "Accelerations: " << endl;
    cout << "   X: " << (this->accelerationX)/sensitivity << " g (" << this->accelerationX << ")" << endl;
    cout << "   Y: " << (this->accelerationY)/sensitivity << " g (" << this->accelerationY << ")" << endl;
    cout << "   Z: " << (this->accelerationZ)/sensitivity << " g (" << this->accelerationZ << ")" << endl;
}

void MPU6050::displayAngularRates()
{
    float sensitivity = ((float)this->sensitivityGyro)/10;
    cout << "Angular Rates: " << endl;
    cout << "   X: " << ((this->angularRateX)-(this->offsetX))/sensitivity << " °/s (" << ((this->angularRateX)-(this->offsetX)) << ")" << endl;
    cout << "   Y: " << ((this->angularRateY)-(this->offsetY))/sensitivity << " °/s (" << ((this->angularRateY)-(this->offsetY)) << ")" << endl;
    cout << "   Z: " << ((this->angularRateZ)-(this->offsetZ))/sensitivity << " °/s (" << ((this->angularRateZ)-(this->offsetZ)) << ")" << endl;
}

void MPU6050::displayTemperature()
{
    cout << "Temperature: " << this->temperatureCelcius << " °C (" << this->temperature << ")" << endl;
}

void MPU6050::displayAll()
{
    this->displayAccelerations();
    this->displayAngularRates();
    this->displayTemperature();
}

void MPU6050::displayRegisters(int nbRegisters)
{
    cout << "Dumping Registers for Debug Purposes:" << endl;
    for(int i=0; i<nbRegisters; ++i)
    {
        cout << setw(2) << setfill('0') << hex << (int)(*(registers+i)) << " ";
        if (i%16==15) cout << endl;
    }
    cout << dec << endl;
}
