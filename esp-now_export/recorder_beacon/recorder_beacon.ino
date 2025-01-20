/*
  the recorder is heavily based off of code from Rui Santos & Sara Santos - Random Nerd Tutorials
  Complete project details at https://RandomNerdTutorials.com/get-change-esp32-esp8266-mac-address-arduino/
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files.  
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
*/

// upstream ESP Board MAC Address: 84:fc:e6:7b:c7:d0
// xiao recorder: 64:e8:33:50:c3:f8
#include "dataStructures.h"
#include <esp_now.h>
#include <WiFi.h>

// Create a struct_message called myData
Message myData;
/*
To reduce the odds of competing for a shared resource, we are only doing copying in our 
rec callback function. Thus, printing happens on a set interval. 
*/
u_int32_t last_reported {0}; //when multiple players are introduced, this will become a array of timestamps, as each player will have a unique timestamp to indicate message uniqueness




// callback function that will be executed when data is received
void OnDataRecv(const uint8_t * mac_addr, const uint8_t *incomingData, int len) {
  memcpy(&myData, incomingData, sizeof(myData));// Update the structures with the new incoming data
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
  if (myData.timestamp > last_reported) {
    last_reported = myData.timestamp;
    // Serial.printf("ts:%lu-b1:{d:%d,ax:%d,ay:%d,az:%d} b2:{d:%d,ax:%d,ay:%d,az:%d} b3:{d:%d,ax:%d,ay:%d,az:%d}",myData.timestamp,
    //             myData.CollectedBeaconData[0].FTMdist,myData.CollectedBeaconData[0].xaccel,myData.CollectedBeaconData[0].yaccel,myData.CollectedBeaconData[0].zaccel,
    //             myData.CollectedBeaconData[1].FTMdist,myData.CollectedBeaconData[1].xaccel,myData.CollectedBeaconData[1].yaccel,myData.CollectedBeaconData[1].zaccel,
    //             myData.CollectedBeaconData[2].FTMdist,myData.CollectedBeaconData[2].xaccel,myData.CollectedBeaconData[2].yaccel,myData.CollectedBeaconData[2].zaccel);
    Serial.printf("%lu,%d,%d,%d,%d,%d,%d,%d",myData.timestamp,myData.CollectedBeaconData[0].FTMdist,myData.CollectedBeaconData[1].FTMdist,myData.CollectedBeaconData[2].FTMdist,myData.CollectedBeaconData[3].FTMdist,
                  myData.xaccel,myData.yaccel,myData.zaccel); //note: currently, not reporting RSSI, but this could be added in the future
    Serial.println();
  }
  
  delay(50);  
}