/*
Data pertaining to an individual beacon. Each message contains 6 instances.
Acceleration data is recorded once per beacon, as required by Kalman filter
*/
typedef struct BeaconData {
  int dist{ 0 };  //note: this is in cm!
  int8_t rssi {127}; //not required for current postprocessing, however will be valuable in the future
} BeaconData;  //Total size: 3bytes

/*
The actual message to be sent to the reporter beacon
*/
typedef struct Message {
  u_int32_t timestamp{ 0 };  // potentially useful if caching messages
  char playerID{ 'a' };
  BeaconData CollectedBeaconData[6];
  int16_t xaccel{ 0 };
  int16_t yaccel{ 0 };
  int16_t zaccel{ 0 };

} Message;  //Total size: 31bytes