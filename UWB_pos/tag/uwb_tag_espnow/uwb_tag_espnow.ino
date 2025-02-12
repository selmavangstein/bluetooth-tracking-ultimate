/*

For ESP32 UWB or ESP32 UWB Pro

*/
#include "datastructures.h"
#include <esp_now.h>
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

const uint8_t broadcastAddress[] = { 0x64, 0xE8, 0x33, 0x50, 0xC3, 0xf8 };
esp_now_peer_info_t peerInfo;

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

void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  // Debugging:
  // Serial.print("\r\nLast Packet Send Status:\t");
  // Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Delivery Success" : "Delivery Fail");
  return;
}



void setup() {
  Serial.begin(115200);
  delay(1000);
  Wire.begin(PIN_SDA, PIN_SCL);
  accel.begin();
  accel.setRange(ADXL345_RANGE_16_G);
  // Serial.println("haiiii");
  //init the configuration
  SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);

  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }
  esp_now_register_send_cb(OnDataSent);

  // Register peer
  memcpy(peerInfo.peer_addr, broadcastAddress, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  // Add peer
  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    // Debug
    // Serial.println("Failed to add peer");
    return;
  }

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
    // Serial.print(millis());
    // Serial.print(',');
    // Serial.print(curData.distances[0]);
    // Serial.print(',');
    // Serial.print(curData.distances[1]);
    // Serial.print(',');
    // Serial.print(curData.distances[2]);
    // Serial.print(',');
    // Serial.print(curData.distances[3]);
    // Serial.print(',');
    // Serial.print(event.acceleration.x);
    // Serial.print(',');
    // Serial.print(event.acceleration.y);
    // Serial.print(',');
    // Serial.print(event.acceleration.z);
    // Serial.println();
    curData.timestamp = millis();
    esp_err_t result = esp_now_send(broadcastAddress, (uint8_t *)&curMessage, sizeof(curMessage));  // sending our message to the recorder beacon via espnow
    delay(30);
    result = esp_now_send(broadcastAddress, (uint8_t *)&curMessage, sizeof(curMessage));  // redundancy message. won't actually be printed, as only novel timestamps are printed
    lastSent = millis();
  }
  // delay(20);
}

void newRange() {
  u_int16_t address = DW1000Ranging.getDistantDevice()->getShortAddress();
  switch (address) {
    case 0x1786:
      curData.CollectedBeaconData[0].FTMdist = (int)(DW1000Ranging.getDistantDevice()->getRange() * 100);
      curData.CollectedBeaconData[0].rssi = (int)(DW1000Ranging.getDistantDevice()->getRXPower() * 100);
      break;

    case 0x2222:
      curData.CollectedBeaconData[1].FTMdist = (int)(DW1000Ranging.getDistantDevice()->getRange() * 100);
      curData.CollectedBeaconData[1].rssi = (int)(DW1000Ranging.getDistantDevice()->getRXPower() * 100);
      break;

    case 0x2897:
      curData.CollectedBeaconData[2].FTMdist = (int)(DW1000Ranging.getDistantDevice()->getRange() * 100);
      curData.CollectedBeaconData[2].rssi = (int)(DW1000Ranging.getDistantDevice()->getRXPower() * 100);
      break;

    case 0x4444:
      curData.CollectedBeaconData[3].FTMdist = (int)(DW1000Ranging.getDistantDevice()->getRange() * 100);
      curData.CollectedBeaconData[3].rssi = (int)(DW1000Ranging.getDistantDevice()->getRXPower() * 100);
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
