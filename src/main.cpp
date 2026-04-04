#include <AD9833.h>
#include <SPI.h>

// Pin Definitions for ESP32-S3
#define FNC_PIN 10
#define DAT_PIN 11
#define CLK_PIN 12

// Initialize the generator
AD9833 gen(FNC_PIN);

//sweep settings
const float startFreq = 22000.0; // 22 kHz
const float endFreq = 25000.0; // 25 kHz
const float freqstep = 10.0; // 10 Hz step
const uint32_t delayTime = 1; // 1 ms delay between steps

void setup() {
  Serial.begin(115200);
  delay(1000); // Give the S3 a moment to stabilize
  
  Serial.println("--- ESP32-S3 AD9833 Controller ---");

  // On the S3, we explicitly start the SPI bus to ensure correct pins
  SPI.begin(CLK_PIN, -1, DAT_PIN, FNC_PIN); // SCK, MISO (not used), MOSI, SS

  gen.begin();
  gen.reset();
  gen.setWave(AD9833_SINE);
 // gen.setFrequency(25000.0,0);

}

void loop() {
  // Sweep from 22 kHz to 25 kHz and back
  for (float f = startFreq; f <= endFreq; f += freqstep) {
    gen.setFrequency(f, 0);
    delay(delayTime);
  }

  for (float f = endFreq; f >= startFreq; f -= freqstep) {
    gen.setFrequency(f, 0);
    delay(delayTime);
  } 

}
