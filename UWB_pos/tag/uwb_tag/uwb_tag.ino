/*

For ESP32 UWB or ESP32 UWB Pro

*/
#include "datastructures.h"
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>
#include "DW1000Ranging.h"
// DW1000 makerfrabs wiring:
#define SPI_SCK 18
#define SPI_MISO 19
#define SPI_MOSI 23
#define DW_CS 4

const int SEND_FREQ = 100;

// connection pins
const uint8_t PIN_RST = 27;  // reset pin
const uint8_t PIN_IRQ = 34;  // irq pin
const uint8_t PIN_SS = 4;    // spi select pin
const uint8_t PIN_SDA = 21;
const uint8_t PIN_SCL = 22;

Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);



// DWM1000 wiring:
// #define SPI_SCK 8
// #define SPI_MISO 9
// #define SPI_MOSI 10
// #define DW_CS 2

// const uint8_t PIN_RST = 5;
// const uint8_t PIN_IRQ = 3;
// const uint8_t PIN_SS = 2;

Message curData;
u_int32_t lastSent = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Wire.begin(PIN_SDA,PIN_SCL);
  accel.begin();
  accel.setRange(ADXL345_RANGE_16_G);
  // Serial.println("haiiii");
  //init the configuration
  SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);
  DW1000Ranging.initCommunication(PIN_RST, PIN_SS, PIN_IRQ);  //Reset, CS, IRQ pin
  //define the sketch as anchor. It will be great to dynamically change the type of module
  DW1000Ranging.attachNewRange(newRange);
  DW1000Ranging.attachNewDevice(newDevice);
  DW1000Ranging.attachInactiveDevice(inactiveDevice);
  //Enable the filter to smooth the distance
  //DW1000Ranging.useRangeFilter(true);

  //we start the module as a tag
  DW1000Ranging.startAsTag("7D:00:22:EA:82:60:3B:9C", DW1000.MODE_LONGDATA_RANGE_LOWPOWER);
}

void loop() {
  sensors_event_t event;
  accel.getEvent(&event);
  DW1000Ranging.loop();

  if ((millis() - lastSent) > SEND_FREQ) {
    Serial.print(millis());
    Serial.print(',');
    Serial.print(curData.distances[0]);
    Serial.print(',');
    Serial.print(curData.distances[1]);
    Serial.print(',');
    Serial.print(curData.distances[2]);
    Serial.print(',');
    Serial.print(curData.distances[3]);
    Serial.print(',');
    Serial.print(event.acceleration.x);
    Serial.print(',');
    Serial.print(event.acceleration.y);
    Serial.print(',');
    Serial.print(event.acceleration.z);
    Serial.println();
    lastSent = millis();
  }
  // delay(20);
}

void newRange() {
  u_int16_t address = DW1000Ranging.getDistantDevice()->getShortAddress();
  switch (address) {
    case 0x1786:
      curData.distances[0] = DW1000Ranging.getDistantDevice()->getRange();
      curData.RSSI[0] = DW1000Ranging.getDistantDevice()->getRXPower();
      break;

    case 0x2222:
      curData.distances[1] = DW1000Ranging.getDistantDevice()->getRange();
      curData.RSSI[1] = DW1000Ranging.getDistantDevice()->getRXPower();
      break;

    case 0x2897:
      curData.distances[2] = DW1000Ranging.getDistantDevice()->getRange();
      curData.RSSI[2] = DW1000Ranging.getDistantDevice()->getRXPower();
      break;
    
    case 0x4444:
      curData.distances[3] = DW1000Ranging.getDistantDevice()->getRange();
      curData.RSSI[3] = DW1000Ranging.getDistantDevice()->getRXPower();
      break;




    default:
      Serial.print("from: ");
      Serial.print(DW1000Ranging.getDistantDevice()->getShortAddress());
      Serial.print("\t Range: ");
      Serial.print(DW1000Ranging.getDistantDevice()->getRange());
      Serial.print(" m");
      Serial.print("\t RX power: ");
      Serial.print(DW1000Ranging.getDistantDevice()->getRXPower());
      Serial.println(" dBm");
  }
}

void newDevice(DW1000Device *device) {
  Serial.print("ranging init; 1 device added ! -> ");
  Serial.print(" short:");
  Serial.println(device->getShortAddress(), HEX);
}

void inactiveDevice(DW1000Device *device) {
  Serial.print("delete inactive device: ");
  Serial.println(device->getShortAddress(), HEX);
}
