/* Our anchor code used for UWB ranging, based off of library example. For more details, check the readme
 *
 * For ESP32 UWB or ESP32 UWB Pro by makerfabs
 *
 */

#include <SPI.h>
#include "DW1000Ranging.h"


// NOTE: anchors must have a different address. As such, ensure to uncomment a different line per beacon
// The short address is easily accessible in the wearable.
// #define ANCHOR_ADD "86:17:5B:D5:A9:9A:E2:9C" //Anchor 1 => thus short address = 1786
// #define ANCHOR_ADD "22:22:AB:89:23:45:C8:A2" //Anchor 2 => thus short address = 2222
// #define ANCHOR_ADD "97:28:6C:E6:BA:AB:F3:AD" //Anchor 3 => thus short address = 2897
#define ANCHOR_ADD "44:44:A1:B2:C3:D5:11:21" //Anchor 4 => thus short address = 4444

#define SPI_SCK 18
#define SPI_MISO 19
#define SPI_MOSI 23
#define DW_CS 4

// connection pins
const uint8_t PIN_RST = 27; // reset pin
const uint8_t PIN_IRQ = 34; // irq pin
const uint8_t PIN_SS = 4;   // spi select pin

void setup()
{
    Serial.begin(115200);
    delay(1000);
    //init the configuration
    SPI.begin(SPI_SCK, SPI_MISO, SPI_MOSI);
    DW1000Ranging.initCommunication(PIN_RST, PIN_SS, PIN_IRQ); //Reset, CS, IRQ pin
    //define the sketch as anchor. It will be great to dynamically change the type of module
    DW1000Ranging.attachNewRange(newRange);
    DW1000Ranging.attachBlinkDevice(newBlink);
    DW1000Ranging.attachInactiveDevice(inactiveDevice);

    DW1000Ranging.startAsAnchor(ANCHOR_ADD, DW1000.MODE_LONGDATA_RANGE_LOWPOWER, false);
    // DW1000Ranging.startAsAnchor(ANCHOR_ADD, DW1000.MODE_SHORTDATA_FAST_LOWPOWER);
    // DW1000Ranging.startAsAnchor(ANCHOR_ADD, DW1000.MODE_LONGDATA_FAST_LOWPOWER);
    // DW1000Ranging.startAsAnchor(ANCHOR_ADD, DW1000.MODE_SHORTDATA_FAST_ACCURACY);
    // DW1000Ranging.startAsAnchor(ANCHOR_ADD, DW1000.MODE_LONGDATA_FAST_ACCURACY);
    // DW1000Ranging.startAsAnchor(ANCHOR_ADD, DW1000.MODE_LONGDATA_RANGE_ACCURACY);
}

void loop()
{
    DW1000Ranging.loop();
}

void newRange()
{
    Serial.print("from: ");
    Serial.print(DW1000Ranging.getDistantDevice()->getShortAddress(), HEX);
    Serial.print("\t Range: ");
    Serial.print(DW1000Ranging.getDistantDevice()->getRange());
    Serial.print(",");
    Serial.print(DW1000Ranging.getDistantDevice()->getRXPower());
    Serial.println();
}

void newBlink(DW1000Device *device)
{
    Serial.print("blink; 1 device added ! -> ");
    Serial.print(" short:");
    Serial.println(device->getShortAddress(), HEX);
}

void inactiveDevice(DW1000Device *device)
{
    Serial.print("delete inactive device: ");
    Serial.println(device->getShortAddress(), HEX);
}
