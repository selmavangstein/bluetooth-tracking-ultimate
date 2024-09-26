#include <Adafruit_NeoPixel.h>
#ifdef __AVR__
  #include <avr/power.h>
#endif
#define PIN        3
#define NUMPIXELS 1
#define BUTTON 0

int buttonState = 0;
int lastButtonState = 0;
int colorIndex = 0;

Adafruit_NeoPixel pixels(NUMPIXELS, PIN, NEO_GRB + NEO_KHZ800);
#define DELAYVAL 500

uint32_t colorList[] = {
  pixels.Color(150, 0, 0),   // Red
  pixels.Color(0, 150, 0),   // Green
  pixels.Color(0, 0, 150),   // Blue
  pixels.Color(150, 150, 0), // Yellow
  pixels.Color(150, 0, 150)  // Purple
};
#define NUM_COLORS 5

void setup() {
#if defined(__AVR_ATtiny85__) && (F_CPU == 16000000)
  clock_prescale_set(clock_div_1);
#endif

  pixels.begin();
  pinMode(BUTTON, INPUT);
}

void loop() {
  pixels.clear();
  buttonState = digitalRead(BUTTON);

  if (buttonState == HIGH && lastButtonState == LOW) {
    colorIndex = (colorIndex + 1) % NUM_COLORS;
    pixels.setPixelColor(0, colorList[colorIndex]);
    pixels.show();
    delay(DELAYVAL);
  } 
  
  lastButtonState = buttonState;
}