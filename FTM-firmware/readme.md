# FTM software
FTM was one of the first positioning technologies we seriously considered. This setup consists of beacons, a wearable, and a recorder beacon. The wearable device iterates through each of the beacons, connecting and taking an FTM measurement, before sending this and acceleration data via espnow to be recorded. 

## Required materials
For our implementation of FTM positioning, we're making use of 6x xiao esp32s3 boards. 4 will act as beacons, 1 will act as a wearable, and 1 will act as a "recorder beacon". This firmware is likely compatible with any esp32 that supports FTM ranging, but this has not been confirmed.

## Flashing boards
Before getting started, make sure you install the necessary dependencies via the Arduino IDE. You'll need Wire, adafruit Sensor, and adafruit ADXL345_U.

Start by flashing the recorder beacon board; you'll need to make sure that you acquire the MAC address for this board, or else the wearable will not be able to communicate with it. This device will send data it receives via a serial-usb connection.

Your wearable board will simply need to be flashed with the provided sketch, after modifying the espNow receiver address. See the recorder_beacon sketch for more details on the mac address retrieval process. Please also ensure that the ADXL345 accelerometer is wired correctly, as a failure to do so can cause unpredictable issues.

To flash beacons, make sure that you change the beacon's address to one of the 4 provided ANCHOD_ADD values (such that there is one of each). Note that if you do change the address of an anchor, you'll also need to modify the switch found in the wearable.

## Common Issues
Make sure that you've enabled USE-CDC-ONBOOT (in the arduino ide) for any device you intend to print over serial. 

Make sure that you have the correct board selected, otherwise pin assignment/other issues might occur.

Ensure that your accelerometer is producing values; try using one of Adafruit's example sketches. Failure to do so can cause issues down the line.

Make sure that you have the correct MAC address of your recorder beacon, and have correctly input this into the wearable's firmware. Otherwise, the connection will fail and you won't be able to collect data.

Make sure each beacon has a unique network name, and that any changes you make to your beacon names are also reflected in your wearable firmware. 