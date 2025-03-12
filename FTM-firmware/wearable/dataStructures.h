/*
Data pertaining to an individual beacon. Each message contains 4 instances.
*/
typedef struct BeaconData {
  int dist{ 0 };  //note: this is in cm!
  int8_t rssi {127}; //not required for current postprocessing, however will be valuable in the future
} BeaconData;  

/*
The actual message to be sent to the reporter beacon
*/
typedef struct Message {
  u_int32_t timestamp{ 0 };  // potentially useful if caching messages
  char playerID{ 'a' };
  BeaconData CollectedBeaconData[4];
  int16_t xaccel{ 0 };
  int16_t yaccel{ 0 };
  int16_t zaccel{ 0 };

} Message;