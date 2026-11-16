// Smart Grow - main controller sketch
// Reads soil moisture + light (BH1750), receives "s,l,n,p,k" values from the
// Python GUI over serial, and drives relays for the water / N / P / K pumps
// and an L298 motor driver for the shade.

#include <Wire.h>
#include <BH1750.h>

BH1750 lightSensor;

// Sensor readings
float s = 0;   // soil moisture
float l = 0;   // light intensity
float n = 0;   // nitrogen (to be received from Python)
float p = 0;   // phosphorus (to be received from Python)
float k = 0;   // potassium (to be received from Python)

void setup() {
  Wire.begin();
  lightSensor.begin();

  Serial.begin(9600);

  // Pin setup (relays are active LOW)
  pinMode(2, OUTPUT); // L298 IN1 = close shade
  pinMode(3, OUTPUT); // L298 IN2 = open shade
  pinMode(4, OUTPUT); // Pump N
  pinMode(5, OUTPUT); // Pump P
  pinMode(6, OUTPUT); // Pump K
  pinMode(7, OUTPUT); // Water pump (soil moisture)

  digitalWrite(2, HIGH);
  digitalWrite(3, HIGH);
  digitalWrite(4, HIGH);
  digitalWrite(5, HIGH);
  digitalWrite(6, HIGH);
  digitalWrite(7, HIGH);
}

void loop() {
  // Read Soil Moisture
  s = analogRead(A0);
  s = s * 100;
  s = s / 1023;

  // Read Light Intensity from BH1750
  l = lightSensor.readLightLevel();

  // Print the values (Soil and Light only)
  Serial.print("Soil: ");
  Serial.print(s);
  Serial.print(", Light: ");
  Serial.println(l);

  // Wait for values from Python
  while (Serial.available() < 3) {  // Ensure we receive data
    delay(100);
  }

  // Receive and store values from Python
  s = Serial.parseFloat();
  l = Serial.parseFloat();
  n = Serial.parseFloat();
  p = Serial.parseFloat();
  k = Serial.parseFloat();

  // === Soil Moisture ===
  if (s > 500) {
    digitalWrite(7, LOW);
    delay(2000);
    digitalWrite(7, HIGH);
    delay(50);
  }
  if (s < 500) {
    digitalWrite(2, LOW);
    digitalWrite(3, HIGH);
    delay(10000);
    digitalWrite(2, HIGH);
    digitalWrite(3, LOW);
    delay(100);
  }

  // === Light ===
  if (l > 0) {
    digitalWrite(2, LOW);
    digitalWrite(3, HIGH);
    delay(2000);
    digitalWrite(2, HIGH);
    digitalWrite(3, HIGH);
    delay(100);
  }
  if (l < 0) {
    digitalWrite(2, HIGH);
    digitalWrite(3, LOW);
    delay(2000);
    digitalWrite(2, HIGH);
    digitalWrite(3, HIGH);
    delay(100);
  }
  if (l == 0) {
    digitalWrite(2, HIGH);
    digitalWrite(3, HIGH);
    delay(100);
  }

  // Execute conditions based on received values
  // === N ===
  if (n > 0) {
    digitalWrite(4, LOW);
    delay(2000);
    digitalWrite(4, HIGH);
    delay(50);
  }
  if (n == 0) {
    digitalWrite(4, HIGH);
    delay(50);
  }

  // === P ===
  if (p > 0) {
    digitalWrite(6, LOW);
    delay(2000);
    digitalWrite(6, HIGH);
    delay(50);
  }
  if (p == 0) {
    digitalWrite(6, HIGH);
    delay(50);
  }

  // === K ===
  if (k > 0) {
    digitalWrite(5, LOW);
    delay(2000);
    digitalWrite(5, HIGH);
    delay(50);
  }
  if (k == 0) {
    digitalWrite(5, HIGH);
    delay(50);
  }

  delay(1000000);  // Wait before next cycle
}
