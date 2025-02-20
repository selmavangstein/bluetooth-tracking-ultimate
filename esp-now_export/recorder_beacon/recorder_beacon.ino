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
#include <esp_wifi.h>
#include <WiFi.h>

const int NumPlayers = 2; // remember to change how we init last_reported depending on number of players!

// Create a struct_message called myData
Message myData;
/*
To reduce the odds of competing for a shared resource, we are only doing copying in our 
rec callback function. Thus, printing happens on a set interval. 
*/
u_int32_t last_reported[NumPlayers] {0,0}; //when multiple players are introduced, this will become a array of timestamps, as each player will have a unique timestamp to indicate message uniqueness




// callback function that will be executed when data is received
void OnDataRecv(const uint8_t * mac_addr, const uint8_t *incomingData, int len) {
  memcpy(&myData, incomingData, sizeof(myData));// Update the structures with the new incoming data
  // if (myData.playerID == 'a'){
  //   last_reported
  // }
  // Serial.print("haiii");
}

void readMacAddress(){
  uint8_t baseMac[6];
  esp_err_t ret = esp_wifi_get_mac(WIFI_IF_STA, baseMac);
  if (ret == ESP_OK) {
    Serial.printf("%02x:%02x:%02x:%02x:%02x:%02x\n",
                  baseMac[0], baseMac[1], baseMac[2],
                  baseMac[3], baseMac[4], baseMac[5]);
  } else {
    Serial.println("Failed to read MAC address");
  }
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

  readMacAddress();
  
  // Once ESPNow is successfully Init, we will register for recv CB to
  // get recv packer info
  esp_now_register_recv_cb(esp_now_recv_cb_t(OnDataRecv));
}
 
void loop() {
  if (myData.playerID == 'a' and myData.timestamp > last_reported[0]) {
    last_reported[0] = myData.timestamp;
    Serial.printf("%c,%lu,%f,%f,%f,%f,%f,%f,%f",myData.playerID,myData.timestamp,(float)myData.CollectedBeaconData[0].dist/100.0,(float)myData.CollectedBeaconData[1].dist/100.0,(float)myData.CollectedBeaconData[2].dist/100.0,(float)myData.CollectedBeaconData[3].dist/100.0,
                  (float)myData.xaccel/100.0,(float)myData.yaccel/100.0,(float)myData.zaccel/100.0); //note: currently, not reporting RSSI, but this could be added in the future
    Serial.println();
  }else if (myData.playerID == 'b' and myData.timestamp > last_reported[1]){
    last_reported[1] = myData.timestamp;
    Serial.printf("%c,%lu,%f,%f,%f,%f,%f,%f,%f",myData.playerID,myData.timestamp,(float)myData.CollectedBeaconData[0].dist/100.0,(float)myData.CollectedBeaconData[1].dist/100.0,(float)myData.CollectedBeaconData[2].dist/100.0,(float)myData.CollectedBeaconData[3].dist/100.0,
                  ((float)myData.xaccel)/100.0,((float)myData.yaccel)/100.0,(float)myData.zaccel/100.0); //note: currently, not reporting RSSI, but this could be added in the future
    Serial.println();
  }
  
  delay(50);  
  // Serial.println("hello");
}