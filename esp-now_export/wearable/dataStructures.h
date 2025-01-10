/*
Struct containing data values from a sample. Initialized with default values. 
Will need to clarify that if a packet is received and contains only the default vals,
it constitutes a failure.

TODO: want one databundle per beacon, then another struct for the entire message to be communicated
  So this will have playerID, timestamp, and arr DataBundle[6]
*/
typedef struct DataBundle {
    // u_int32_t timestamp {0};
    // int16_t xaccel {0}; //Think we actually just want one of these
    // int16_t yaccel {0};
    // int16_t zaccel {0};
    u_int16_t FTMdist {0}; //note: this is in cm!
    char beaconID {'a'};
    // char playerID {'a'};
    // char d {}; // notably, can encode another char without affecting size of struct--thx packing! Not sure what to use this for, but leaving as an option
    int8_t rssi {127};

    void printContents(){
      Serial.println("-------------------");
      // Serial.printf("TS: %lu\n",timestamp);
      // Serial.printf("xaccel: %d\n",xaccel);
      // Serial.printf("yaccel: %d\n",yaccel);
      // Serial.printf("zaccel: %d\n",zaccel);
      Serial.printf("dist: %d\n", FTMdist);
      Serial.printf("beaconID: %s\n", beaconID);
      // Serial.printf("playerID: %s\n", playerID);
      Serial.printf("RSSI: %d\n", rssi);
       Serial.println("-------------------");
    }
} DataBundle;


/*
The actual message to be sent to the reporter beacon :D
*/
typedef struct Message {
  u_int32_t timestamp {0};
  int16_t xaccel {0}; //Think we actually just want one of these
  int16_t yaccel {0};
  int16_t zaccel {0};
  char playerID {'a'};
  DataBundle bundles[6];

} Message;