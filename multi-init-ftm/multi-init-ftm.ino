/* Based on Wi-Fi FTM Initiator Arduino Example

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/
#include "WiFi.h"

/*
   THIS FEATURE IS SUPPORTED ONLY BY ESP32-S2 AND ESP32-C3
*/

// Beacon number and names -- assumes no password to connect
int numBeacons = 2;
const char *beacons[2] = { "beacon2", "beacon3" };  //"beacon1",


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
int connectTimeOut = 250;  //time in ms that we will try connecting for.

// FTM report handler with the calculated data from the round trip
// WARNING: This function is called from a separate FreeRTOS task (thread)!
void onFtmReport(arduino_event_t *event) {
  const char *status_str[5] = { "SUCCESS", "UNSUPPORTED", "CONF_REJECTED", "NO_RESPONSE", "FAIL" };
  wifi_event_ftm_report_t *report = &event->event_info.wifi_ftm_report;
  // Set the global report status
  ftmSuccess = report->status == FTM_STATUS_SUCCESS;
  if (ftmSuccess) {
    // The estimated distance in meters may vary depending on some factors (see README file)
    Serial.printf("%.2f|%lu|%d", (float)report->dist_est / 100.0, report->rtt_est, WiFi.RSSI());
    // Pointer to FTM Report with multiple entries, should be freed after use
    free(report->ftm_report_data);
    WiFi.disconnect();  //might be evil!!!
    // received = true;
  } else {
    Serial.print("FTMErr:");
    Serial.print(status_str[report->status]);
    WiFi.disconnect();  //might be evil!!!
  }
  // Signal that report is received
  xSemaphoreGive(ftmSemaphore);
}

// Initiate FTM Session and wait for FTM Report
bool getFtmReport() {
  if (!WiFi.initiateFTM(FTM_FRAME_COUNT, FTM_BURST_PERIOD)) {
    Serial.print("FTM Error: Initiate Session Failed");
    return false;
  }
  // Wait for signal that report is received and return true if status was success
  return xSemaphoreTake(ftmSemaphore, portMAX_DELAY) == pdPASS && ftmSuccess;
}

void setup() {
  Serial.begin(115200);

  // Create binary semaphore (initialized taken and can be taken/given from any thread/ISR)
  ftmSemaphore = xSemaphoreCreateBinary();

  // Will call onFtmReport() from another thread with FTM Report events.
  WiFi.onEvent(onFtmReport, ARDUINO_EVENT_WIFI_FTM_REPORT);
}

void loop() {
  Serial.print("{");
  for (int i = 0; i < numBeacons; i++) {
    startConnectTime = millis();
    if (i != 0){
      Serial.print(",");
    }
      Serial.printf("%s:", beacons[i]);
    WiFi.begin(beacons[i], NULL);
    while ((WiFi.status() != WL_CONNECTED) && (millis() - startConnectTime) < connectTimeOut) { 
      delay(10);
    }
    if (WiFi.status() == WL_CONNECTED) {
      getFtmReport();
      WiFi.disconnect();
    } else {
      WiFi.disconnect();
      Serial.print("conERR");
    }
  }
  Serial.print("}");
  Serial.println();
}
