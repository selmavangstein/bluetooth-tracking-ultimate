# UWB software
For its accuracy and sampling rate, UWB represented one of the best options we found thus far. For our experiments, we utilized dw1000 boards. By default, this software also utilizes espnow to extract data. The recorder beacon software is identical for FTM, thus the sketch found in the FTM-firmware can be used.

## Required materials
For our implementation of UWB positioning, we're making use of 5xMakerfabs UWB-esp32 boards and another that supports espNow. 4 will act as beacons(to use UWB terminology, anchor), 1 will act as a wearable (tag), and 1 will act as a recorder beacon. You will also need an ADXL345 accelerometer, connected to the default i2c pins of your board.

## Flashing boards
Before getting started, make sure you install the necessary dependencies via the Arduino IDE. Notably, we built using makerfabs provided driver (based off the original, Thotro driver)

Your wearable board will simply need to be flashed with the provided sketch, after modifying the espNow receiver address. See the recorder_beacon sketch in FTM-firmware for more details on the mac address retrieval process. Please also ensure that the ADXL345 accelerometer is wired correctly, as a failure to do so can cause unpredictable issues.

To flash beacons, make sure that you change the beacon's address to one of the 4 provided ANCHOD_ADD values (such that there is one of each). Note that if you do change the address of an anchor, you'll also need to modify the switch found in the wearable.

## Common issues
Make sure that you've enabled USE-CDC-ONBOOT (in the arduino ide) for any device you intend to print over serial. 

Make sure that you have the correct board selected, otherwise pin assignment/other issues might occur. The makerfabs boards just work with the "ESP32 Dev Module" setting.

Ensure that your accelerometer is producing values; try using one of Adafruit's example sketches. Failure to do so can cause issues down the line.

Make sure that you have the correct MAC address of your recorder beacon, and have correctly input this into the wearable's firmware. Otherwise, the connection will fail and you won't be able to collect data.

