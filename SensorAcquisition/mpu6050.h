#ifndef MPU6050_H
#define MPU6050_H

#define BUFFER_SIZE 0x75

#define REG_SMPLRT_DIV      0x19    // R/W  [7:0]
#define REG_CONFIG          0x1A    // R/W  [5:3] EXT_SYNC_SET  [2:0] DLPF_CFG
#define REG_GYRO_CONFIG     0x1B    // R/W  [7:5] (X/Y/Z)G_SET  [4:3] FS_SET
#define REG_ACCEL_CONFIG    0x1C    // R/W  [7:5] (X/Y/Z)A_SET  [4:3] AFS_SET   [2:0] ACCEL_HPH
#define REG_ACCEL_XOUT      0x3B    // R    [15:0] (2x[7:0])
#define REG_ACCEL_YOUT      0x3D    // R    [15:0] (2x[7:0])
#define REG_ACCEL_ZOUT      0x3F    // R    [15:0] (2x[7:0])
#define REG_TEMP_OUT        0x41    // R    [15:0] (2x[7:0])
#define REG_GYRO_XOUT       0x43    // R    [15:0] (2x[7:0])
#define REG_GYRO_YOUT       0x45    // R    [15:0] (2x[7:0])
#define REG_GYRO_ZOUT       0x47    // R    [15:0] (2x[7:0])
#define REG_PWR_MGMT_1      0x6B    // R/W  [7] DEVICE_RESET    [6] SLEEP   [5] CYCLE   [3] TEMP_DIS    [2:0] CLK_SEL
#define REG_WHO_AM_I        0x75    // R    [6:1]

class MPU6050
{
    enum RANGE_ACC  // value of accel_config register (HPF disabled)
    {
        PLUSMINUS_2_G    = 0b00000000,
        PLUSMINUS_4_G    = 0b00001000,
        PLUSMINUS_8_G    = 0b00010000,
        PLUSMINUS_16_G   = 0b00011000,
    };

    enum SENSITIVITY_ACC    // LSB/mg
    {
        FS_2G   = 16384,
        FS_4G   = 8192,
        FS_8G   = 4096,
        FS_16G  = 2048,
    };

    enum RANGE_GYRO  // value of gyro_config register
    {
        PLUSMINUS_250_DPS   = 0b00000000,
        PLUSMINUS_500_DPS   = 0b00001000,
        PLUSMINUS_1000_DPS  = 0b00010000,
        PLUSMINUS_2000_DPS  = 0b00011000,
    };

    enum SENSITIVITY_GYRO   // 10*LSB/°/s
    {
        FS_250  = 1310,
        FS_500  = 655,
        FS_1000 = 328,
        FS_2000 = 164,
    };

public:
    enum LPFILTER   // value of config register for gyro bandwidth value (with FSYNC disabled)
    {
        LP_256HZ    = 0b00000000,    // acc 260Hz
        LP_188HZ 	= 0b00000001,    // acc 184Hz
        LP_98HZ 	= 0b00000010,    // acc 94Hz
        LP_42HZ 	= 0b00000011,    // acc 44Hz
        LP_20HZ 	= 0b00000100,    // acc 21Hz
        LP_10HZ 	= 0b00000101,    // acc 10Hz
        LP_5HZ  	= 0b00000110,    // acc 5Hz
    };

private:
    unsigned int bus;       // the I2C bus number
    unsigned int address;   // the device address on the I2C bus
    int file;               // the file handle to the device
    unsigned char *registers;

    short accelerationX, accelerationY, accelerationZ;
    short temperature;
    float temperatureCelcius;
    short angularRateX, angularRateY, angularRateZ;
    short offsetX, offsetY, offsetZ;

    RANGE_ACC rangeAcc;
    SENSITIVITY_ACC sensitivityAcc;
    RANGE_GYRO rangeGyro;
    SENSITIVITY_GYRO sensitivityGyro;

public:
    MPU6050(unsigned int bus=1, unsigned int address=0x68);
    int openI2C();
    int writeRegister(unsigned int registerAddress, unsigned char value);
    int writeAddress(unsigned char addressValue);
    unsigned char readRegister(unsigned int registerAddress);
    unsigned char* readRegisters(unsigned int number, unsigned int fromAddress=0);
    void closeI2C();
    ~MPU6050();

    void initialize();
    void setDigitalLowPassFilter(LPFILTER filterValue);
    bool testConnection();
    int readFullSensorState();  // some errors
    float readSampleRate();
    short readAccelerationX();
    short readAccelerationY();
    short readAccelerationZ();
    short readAngularRateX();
    short readAngularRateY();
    short readAngularRateZ();
    int readAccelerations();    // update private states (return 1 if sync loss)
    int readAngularRates();     // ""
    int readTemperature();      // ""
    int readAll();              // ""
    int offsetEstimation(int nbSample); // ""

    short getAccelerationX() { return this->accelerationX; }
    short getAccelerationY() { return this->accelerationY; }
    short getAccelerationZ() { return this->accelerationZ; }
    float getTemperatureCelcius() { return this->temperatureCelcius; }
    short getAngularRateX() { return ((this->angularRateX)-(this->offsetX)); }
    short getAngularRateY() { return ((this->angularRateY)-(this->offsetY)); }
    short getAngularRateZ() { return ((this->angularRateZ)-(this->offsetZ)); }
    int getSensitivityAcc() { return this->sensitivityAcc; }
    float getSensitivityGyro() { return (float)this->sensitivityGyro/10; }

    void displayAccelerations();
    void displayAngularRates();
    void displayTemperature();
    void displayAll();
    void displayRegisters(int nbRegisters=BUFFER_SIZE);
};

#endif // MPU6050_H
