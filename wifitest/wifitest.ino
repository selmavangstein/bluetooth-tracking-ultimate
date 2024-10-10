#include <WiFi.h> //using WiFi by Arduino; libe 1.2.7

#define RECONNECT_THRESHOLD 15

const char* ssid = "Marshall device";
const char* pass = "beepboopbeep";
int num_dropped_signals = 0;
long cur_RSSI = 0;

void setup() {
  // put your setup code here, to run once:
  // https://www.upesy.com/blogs/tutorials/how-to-connect-wifi-acces-point-with-esp32#
  // credit where credit is due
  Serial.begin(115200);
  delay(2000);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid,pass);

  while(WiFi.status() != WL_CONNECTED){
        Serial.print(".");
        delay(100);
    }
  Serial.println("We're online!");
}

void loop() {
  // put your main code here, to run repeatedly:
  delay(1000);
  cur_RSSI = WiFi.RSSI();
  Serial.print("current: ");
  Serial.println(cur_RSSI);
  if (cur_RSSI == 0){
    num_dropped_signals++;
    if(num_dropped_signals > RECONNECT_THRESHOLD){
      setup();
      num_dropped_signals = 0;
    }
  }
}
