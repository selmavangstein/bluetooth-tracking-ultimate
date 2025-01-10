/* Based on Wi-Fi FTM Initiator Arduino Example
 *
 *   This example code is in the Public Domain (or CC0 licensed, at your option.)
 *
 *   Unless required by applicable law or agreed to in writing, this
 *   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 *   CONDITIONS OF ANY KIND, either express or implied.
 */
#include "WiFi.h"
#include <esp_now.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADXL345_U.h>
#include "dataStructures.h"

Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);

/*
 *   THIS FEATURE IS SUPPORTED ONLY BY ESP32-S2 AND ESP32-C3
 */

// Beacon number and names -- assumes no password to connect
int numBeacons = 3;
const char *beacons[3] = { "beacon1","beacon2", "beacon3" };  //"beacon1",


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

uint8_t broadcastAddress[] = {0x84, 0xFC, 0xE6, 0x7B, 0xC7, 0xd0};

esp_now_peer_info_t peerInfo;

void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
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
        // The estimated distance in meters may vary depending on some factors (see README file)
        // Serial.printf("[%.2f,%lu,%d]", (float)report->dist_est / 100.0, report->rtt_est,WiFi.RSSI());
        curValues.FTMdist = (u_int16_t) report->dist_est;
        // curValues.rssi = WiFi.RSSI();
        // Pointer to FTM Report with multiple entries, should be freed after use
        free(report->ftm_report_data);
        WiFi.disconnect();  //might be evil!!!
        // received = true;
    } else {
        // Serial.print("[FTMErr");
        // Serial.print(status_str[report->status]);
        // Serial.print(']');
        WiFi.disconnect();  //might be evil!!!
    }
    // Signal that report is `ed
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
      Serial.println("Error initializing ESP-NOW");
      return;
    }
    esp_now_register_send_cb(OnDataSent);
  
  // Register peer
    memcpy(peerInfo.peer_addr, broadcastAddress, 6);
    peerInfo.channel = 0;  
    peerInfo.encrypt = false;
  
  // Add peer        
    if (esp_now_add_peer(&peerInfo) != ESP_OK){
      Serial.println("Failed to add peer");
      return;
    }
  
}

void loop() {
    for (int i = 0; i < numBeacons; i++) {
        startConnectTime = millis();
        sensors_event_t event; 
        accel.getEvent(&event);
        // TODO: add method that sets all curValues to default, then call it here. This way, failed readings can be deciphered.
        curValues.xaccel = (int16_t) event.acceleration.x;
        curValues.yaccel = (int16_t) event.acceleration.y;
        curValues.zaccel = (int16_t) event.acceleration.z;
        WiFi.begin(beacons[i], NULL);
        while ((WiFi.status() != WL_CONNECTED) && (millis() - startConnectTime) < connectTimeOut) {
            delay(10);
        }
        if (WiFi.status() == WL_CONNECTED) {
            getFtmReport();
            // curValues.timestamp = millis();
            
            //Now, we update our message. Specifically, we update the data corresponding to the beacon.
            curMessage.CollectedBeaconData[i].FTMdist = curValues.FTMdist;
            curMessage.CollectedBeaconData[i].xaccel = curValues.xaccel;
            curMessage.CollectedBeaconData[i].yaccel = curValues.yaccel;
            curMessage.CollectedBeaconData[i].zaccel = curValues.zaccel;
            // curMessage.bundles[i].rssi = curValues.rssi;



            // Serial.printf("%d",curValues.FTMdist);
            WiFi.disconnect();
        } else {
            WiFi.disconnect();
            // Serial.print("\"conERR\"");
        }
    }
    // sending our message to the recorder beacon via espnow
    esp_err_t result = esp_now_send(broadcastAddress, (uint8_t *) &curMessage, sizeof(curMessage));

}
