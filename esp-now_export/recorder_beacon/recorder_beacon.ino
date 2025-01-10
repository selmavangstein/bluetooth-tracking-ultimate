/*
  the recorder is heavily based off of code from Rui Santos & Sara Santos - Random Nerd Tutorials
  Complete project details at https://RandomNerdTutorials.com/get-change-esp32-esp8266-mac-address-arduino/
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files.  
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

// ESP Board MAC Address: 84:fc:e6:7b:c7:d0
#include "dataStructures.h"
#include <esp_now.h>
#include <WiFi.h>

// Create a struct_message called myData
Message myData;




// callback function that will be executed when data is received
void OnDataRecv(const uint8_t * mac_addr, const uint8_t *incomingData, int len) {
  memcpy(&myData, incomingData, sizeof(myData));

  // Update the structures with the new incoming data
  Serial.printf("ts:%lu b1:%d b2:%d b3:%d", myData.timestamp,myData.bundles[0].FTMdist,myData.bundles[1].FTMdist,myData.bundles[2].FTMdist);
  Serial.println();
}
 
void setup() {
  //Initialize Serial Monitor
  Serial.begin(115200);
  
  //Set device as a Wi-Fi Station
  WiFi.mode(WIFI_STA);

  //Init ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }
  
  // Once ESPNow is successfully Init, we will register for recv CB to
  // get recv packer info
  esp_now_register_recv_cb(esp_now_recv_cb_t(OnDataRecv));
}
 
void loop() {


  delay(10);  
}