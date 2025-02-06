typedef struct Message{
  char playerID {'a'};
  float distances[4] = {0.0,0.0,0.0,0.0};
  float RSSI[4];
  u_int32_t timestamp {0};
}Message;