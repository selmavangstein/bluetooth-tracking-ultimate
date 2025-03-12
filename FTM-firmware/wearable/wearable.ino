/* Based on Wi-Fi FTM Initiator Arduino Example
 *
 *   This code is to be flashed on one of our wearable devices, which is made up of 
 *   a xiao esp32s3 and a ADXL345 accelerometer. Note that if the accelerometer is 
 *   disconnected, or otherwise nonfunctional, you may experience unpredictable 
 *   behavior
 * 
 *   NOTE: ensure that you change the MAC address of the recorder beacon
 */
#include "WiFi.h"
#include <esp_now.h>
#include <Wire.h>
#include <Adafruit_Sensor.h> //should be installed as a dependency for the ADXL345 library
#include <Adafruit_ADXL345_U.h>
#include "dataStructures.h"

Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);  // ensure board target is set to xiao_esp32s3, or else the I2C pins may be incorrect

// Beacon number and names -- assumes no password to connect
int numBeacons = 4;
const char *beacons[4] = { "beacon1", "beacon2", "beacon3", "beacon4" };


// FTM settings
// Number of FTM frames requested in terms of 4 or 8 bursts (allowed values - 0 (No pref), 16, 24, 32, 64)
const uint8_t FTM_FRAME_COUNT = 8;  //default 16
// Requested time period between consecutive FTM bursts in 100’s of milliseconds (allowed values - 0 (No pref) or 2-255)
const uint16_t FTM_BURST_PERIOD = 2;

// Semaphore to signal when FTM Report has been received
SemaphoreHandle_t ftmSemaphore;
// Status of the received FTM Report
bool ftmSuccess = true;
// bool received = false;

unsigned long startConnectTime;
int connectTimeOut = 150;  //time in ms that we will try connecting for.

BeaconData curValues;
Message curMessage;

const uint8_t broadcastAddress[] = { 0x64, 0xE8, 0x33, 0x50, 0xC3, 0xf8 }; //TODO: CHANGE THIS to the address of your recorder beacon!

esp_now_peer_info_t peerInfo;

void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  // Debugging:
  // Serial.print("\r\nLast Packet Send Status:\t");
  // Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Delivery Success" : "Delivery Fail");
  return;
}

// FTM report handler with the calculated data from the round trip
// WARNING: This function is called from a separate FreeRTOS task (thread)!
void onFtmReport(arduino_event_t *event) {
  const char *status_str[5] = { "SUCCESS", "UNSUPPORTED", "CONF_REJECTED", "NO_RESPONSE", "FAIL" };
  wifi_event_ftm_report_t *report = &event->event_info.wifi_ftm_report;
  // Set the global report status
  ftmSuccess = report->status == FTM_STATUS_SUCCESS;
  if (ftmSuccess) {
    // Serial.printf("[%.2f,%lu,%d]", (float)report->dist_est / 100.0, report->rtt_est,WiFi.RSSI());
    curValues.FTMdist = report->dist_est;
    curValues.rssi = WiFi.RSSI();
    free(report->ftm_report_data);
    WiFi.disconnect();  //TODO: evaluate importance of these disconnect calls within callback; don't think they're necessary
  } else {
    WiFi.disconnect(); 
  }
  // Signal that report is received
  xSemaphoreGive(ftmSemaphore);
}

// Initiate FTM Session and wait for FTM Report
bool getFtmReport() {
  if (!WiFi.initiateFTM(FTM_FRAME_COUNT, FTM_BURST_PERIOD)) {
    Serial.print("\"FTM Error: Initiate Session Failed\"");
    return false;
  }
  // Wait for signal that report is received and return true if status was success
  return xSemaphoreTake(ftmSemaphore, portMAX_DELAY) == pdPASS && ftmSuccess;
}

void setup() {
  Serial.begin(115200);
  accel.begin();
  accel.setRange(ADXL345_RANGE_16_G);

  WiFi.mode(WIFI_STA);
  // Create binary semaphore (initialized taken and can be taken/given from any thread/ISR)
  ftmSemaphore = xSemaphoreCreateBinary();

  // Will call onFtmReport() from another thread with FTM Report events.
  WiFi.onEvent(onFtmReport, ARDUINO_EVENT_WIFI_FTM_REPORT);

  if (esp_now_init() != ESP_OK) {
    // Debug
    // Serial.println("Error initializing ESP-NOW");
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
}

void loop() {
  for (int i = 0; i < numBeacons; i++) {
    sensors_event_t event;
    accel.getEvent(&event);
    startConnectTime = millis();
    // TODO: add method that sets all curValues to default, then call it here. This way, failed readings can be deciphered.
    curMessage.xaccel = (int16_t)(event.acceleration.x * 100);
    curMessage.yaccel = (int16_t)(event.acceleration.y * 100);
    curMessage.zaccel = (int16_t)(event.acceleration.z * 100);
    WiFi.begin(beacons[i], NULL);
    while ((WiFi.status() != WL_CONNECTED) && (millis() - startConnectTime) < connectTimeOut) {
      delay(10);
    }
    if (WiFi.status() == WL_CONNECTED) {
      getFtmReport();
      //Now, we update our message. Specifically, we update the data corresponding to the beacon.
      curMessage.CollectedBeaconData[i].dist = curValues.dist;
      WiFi.disconnect();
    } else {
      WiFi.disconnect();
    }
  }
  curMessage.timestamp = millis();  
  esp_err_t result = esp_now_send(broadcastAddress, (uint8_t *)&curMessage, sizeof(curMessage));  // sending our message to the recorder beacon via espnow
  delay(30);
  result = esp_now_send(broadcastAddress, (uint8_t *)&curMessage, sizeof(curMessage));  // redundancy message. won't actually be printed, as only novel timestamps are printed
}
