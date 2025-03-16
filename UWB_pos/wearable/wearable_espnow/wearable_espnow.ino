/*

To be flashed onto makerfabs UWB (dw1000) development kits. 
Please look at the readme for more info.
*/
#include "WiFi.h"
#include "datastructures.h"
#include <esp_now.h>
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>
#include "DW1000Ranging.h"
// DW1000 makerfabs wiring:
#define SPI_SCK 18
#define SPI_MISO 19
#define SPI_MOSI 23
#define DW_CS 4

const int SEND_FREQ = 100; //defines how often a new message is sent.

// connection pins
const uint8_t PIN_RST = 27;  // reset pin
const uint8_t PIN_IRQ = 34;  // irq pin
const uint8_t PIN_SS = 4;    // spi select pin
const uint8_t PIN_SDA = 21;
const uint8_t PIN_SCL = 22;

Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);


// NOTE: change the broadcastAddress to the mac address of the recorder beacon
// to find the recorder address, uncomment the readMacAddress() function in setup(), then insert it here in the following format
// const uint8_t broadcastAddress[] = { 0x64, 0xE8, 0x33, 0x50, 0xC3, 0xf8 }; //prev recorder beacon address
const uint8_t broadcastAddress[] = { 0xf0, 0x9e, 0x9e, 0x3b, 0xe5, 0xd8 };
esp_now_peer_info_t peerInfo;

Message curData;
u_int32_t lastSent = 0;

void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  // Uncomment for USB debugging
  // Serial.print("\r\nLast Packet Send Status:\t");
  // Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Delivery Success" : "Delivery Fail");
  return;
}



void setup() {
  curData.playerID = 'b'; //change this depending on which player this wearable represents. Must be char.
  Serial.begin(115200);
  delay(1000);
  // Debugging with no accel
  Wire.begin(PIN_SDA, PIN_SCL);
  accel.begin();
  accel.setRange(ADXL345_RANGE_16_G);

  //init the configuration
  SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);
  WiFi.mode(WIFI_STA);
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
    Serial.println("Failed to add peer");
    return;
  }

  DW1000Ranging.initCommunication(PIN_RST, PIN_SS, PIN_IRQ);  //Reset, CS, IRQ pin
  DW1000Ranging.attachNewRange(newRange);
  DW1000Ranging.attachNewDevice(newDevice);
  DW1000Ranging.attachInactiveDevice(inactiveDevice);
  //Set as tag
  DW1000Ranging.startAsTag("7D:00:22:EA:82:60:3B:9C", DW1000.MODE_LONGDATA_RANGE_LOWPOWER); //tag1
  // DW1000Ranging.startAsTag("99:99:88:88:66:11:25:32", DW1000.MODE_LONGDATA_RANGE_LOWPOWER);  //tag2
}

void loop() {

  DW1000Ranging.loop();

  if ((millis() - lastSent) > SEND_FREQ) {
    //   // Uncomment to debug results over USB
    //   Serial.print(millis());
    //   Serial.print(',');
    //   Serial.print(curData.CollectedBeaconData[0].dist);
    //   Serial.print(',');
    //   Serial.print(curData.CollectedBeaconData[1].dist);
    //   Serial.print(',');
    //   Serial.print(curData.CollectedBeaconData[2].dist);
    //   Serial.print(',');
    //   Serial.print(curData.CollectedBeaconData[3].dist);
    //   Serial.print(',');
    //   Serial.print(event.acceleration.x);
    //   Serial.print(',');
    //   Serial.print(event.acceleration.y);
    //   Serial.print(',');
    //   Serial.print(event.acceleration.z);
    //   Serial.println();
    sensors_event_t event;
    accel.getEvent(&event);
    curData.xaccel = (int)(event.acceleration.x * 100.0);
    curData.yaccel = (int)(event.acceleration.y * 100.0);
    curData.zaccel = (int)(event.acceleration.z * 100.0);
    curData.timestamp = millis();
    esp_err_t result = esp_now_send(broadcastAddress, (uint8_t *)&curData, sizeof(curData));  // sending our message to the recorder beacon via espnow
    lastSent = millis();
  }
}

void newRange() {
  u_int16_t address = DW1000Ranging.getDistantDevice()->getShortAddress();
  switch (address) {
    case 0x1786:
      curData.CollectedBeaconData[0].dist = (int)(DW1000Ranging.getDistantDevice()->getRange() * 100);
      curData.CollectedBeaconData[0].rssi = (int)(DW1000Ranging.getDistantDevice()->getRXPower() * 100);
      break;

    case 0x2222:
      curData.CollectedBeaconData[1].dist = (int)(DW1000Ranging.getDistantDevice()->getRange() * 100);
      curData.CollectedBeaconData[1].rssi = (int)(DW1000Ranging.getDistantDevice()->getRXPower() * 100);
      break;

    case 0x2897:
      curData.CollectedBeaconData[2].dist = (int)(DW1000Ranging.getDistantDevice()->getRange() * 100);
      curData.CollectedBeaconData[2].rssi = (int)(DW1000Ranging.getDistantDevice()->getRXPower() * 100);
      break;

    case 0x4444:
      curData.CollectedBeaconData[3].dist = (int)(DW1000Ranging.getDistantDevice()->getRange() * 100);
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
