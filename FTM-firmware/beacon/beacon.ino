/* Wi-Fi FTM Responder - heavily based off of Arduino example
 *
 * This code is to be flashed on FTM responder beacons.
 * It's only purpose is to respond to FTM messages
 * It will not output anything via serial by default.
*/
#include "WiFi.h"
// Change the SSID and PASSWORD here if needed
const char *WIFI_FTM_SSID = "beacon4"; //change to "beacon1","beacon2","beacon3" as needed
// const char *WIFI_FTM_PASS = "optional"; //for prototyping, we do unencrypted communication



void setup() {
  // Enable AP with FTM support (last argument is 'true')
  WiFi.softAP(WIFI_FTM_SSID, NULL, 1, 0, 4, true);
}

void loop() {
  delay(1000);
}
