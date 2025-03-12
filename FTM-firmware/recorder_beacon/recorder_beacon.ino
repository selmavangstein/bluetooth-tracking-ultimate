/*
 * the recorder is heavily based off of code from Rui Santos & Sara Santos - Random Nerd Tutorials
 * Complete project details at https://RandomNerdTutorials.com/get-change-esp32-esp8266-mac-address-arduino/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files.
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * provided is the code for our recorder beacons. It will receive message objects (see "dataStructures.h")
 * and print new message instances. These messages are received via espnow. For further information about
 * espnow, please consult the above tutorial or expressif's documentation
 *
 * NOTE: before actually using this firmware, you'll need to determine the MAC address
 * of the recorder beacon. To do this, uncomment the readMacAddress function in the setup
 * of the program. You can also sometimes see this in the flashing menu of the Arduino IDE.
 * Note this down, and modify the address in the wearables firmware accordingly.
 */

#include "dataStructures.h"
#include <esp_now.h>
#include <esp_wifi.h>
#include <WiFi.h>

const int NumPlayers = 2;  // remember to change how we init last_reported depending on number of players!

//TODO: potentially try hashmap so NumPlayers is not hardcoded
Message msgBuffer;  //temporary buffer which holds data, before copying it to the appropriate message buffer
Message playerMessage[NumPlayers];

/*
 To reduce the odds of competing for a shared resource, we are only doing copying in our
 rec callback function. Thus, printing happens on a set interval, in our main loop().
 */
u_int32_t last_reported[NumPlayers]{ 0, 0 };  //when multiple players are introduced, this will become a array of timestamps, as each player will have a unique timestamp to indicate message uniqueness


// callback function that will be executed when data is received
// for now this is only set up for 2 players, although it should be relatively easy to extend
void OnDataRecv(const uint8_t *mac_addr, const uint8_t *incomingData, int len) {
  memcpy(&msgBuffer, incomingData, sizeof(msgBuffer));  // Update the structures with the new incoming data
  if (msgBuffer.playerID == 'a') {
    memcpy(&playerMessage[0], &msgBuffer, sizeof(msgBuffer));
  } else if (msgBuffer.playerID == 'b') {
    memcpy(&playerMessage[1], &msgBuffer, sizeof(msgBuffer));
  } else {
    Serial.println("WARNING: unknown player");
  }
}

void readMacAddress() {
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
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);

  //Init ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }
  // readMacAddress(); //enable to ensure correct MAC address of receiver beacon

  esp_now_register_recv_cb(esp_now_recv_cb_t(OnDataRecv));
}

void loop() {
  // Multiplayer logic currently disabled, due to bugs
  // for (int i = 0; i < NumPlayers; i++) {
  //   if (playerMessage[i].timestamp > last_reported[i]) {
  //     Serial.printf("%c,%lu,%f,%f,%f,%f,%f,%f,%f", playerMessage[i].playerID, playerMessage[i].timestamp, (float)playerMessage[i].CollectedBeaconData[0].dist / 100.0, (float)playerMessage[i].CollectedBeaconData[1].dist / 100.0, (float)playerMessage[i].CollectedBeaconData[2].dist / 100.0, (float)playerMessage[i].CollectedBeaconData[3].dist / 100.0,
  //                   (float)playerMessage[i].xaccel / 100.0, (float)playerMessage[i].yaccel / 100.0, (float)playerMessage[i].zaccel / 100.0);  //note: currently, not reporting RSSI, but this could be added in the future
  //     Serial.println();
  //     last_reported[i] = playerMessage[i].timestamp;
  //   }
  // }
  Serial.printf("%c,%lu,%f,%f,%f,%f,%f,%f,%f", playerMessage[0].playerID, playerMessage[0].timestamp, (float)playerMessage[0].CollectedBeaconData[0].dist / 100.0, (float)playerMessage[0].CollectedBeaconData[1].dist / 100.0, (float)playerMessage[0].CollectedBeaconData[2].dist / 100.0, (float)playerMessage[0].CollectedBeaconData[3].dist / 100.0,
                (float)playerMessage[0].xaccel / 100.0, (float)playerMessage[0].yaccel / 100.0, (float)playerMessage[0].zaccel / 100.0);  //note: currently, not reporting RSSI, but this could be added in the future
  Serial.println();

  // Serial.printf("%c,%lu,%f,%f,%f,%f,%f,%f,%f", playerMessage[1].playerID, playerMessage[1].timestamp, (float)playerMessage[1].CollectedBeaconData[0].dist / 100.0, (float)playerMessage[1].CollectedBeaconData[1].dist / 100.0, (float)playerMessage[1].CollectedBeaconData[2].dist / 100.0, (float)playerMessage[1].CollectedBeaconData[3].dist / 100.0,
  //               (float)playerMessage[1].xaccel / 100.0, (float)playerMessage[1].yaccel / 100.0, (float)playerMessage[1].zaccel / 100.0);  //note: currently, not reporting RSSI, but this could be added in the future
  // Serial.println();

  delay(50); //fresh values will be printed every 50ms

}
