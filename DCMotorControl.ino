// Pin usage with the Motorshield
// ---------------------------------------
// Analog pins: not used at all
//     A0 ... A5 are still available
//     They all can also be used as digital pins.
//     Also I2C (A4=SDA and A5=SCL) can be used.
//     These pins have a breadboard area on the shield.
// Digital pins: used: 3,4,5,6,7,8,9,10,11,12
//     Pin 9 and 10 are only used for the servo motors.
//     Already in use: 0 (RX) and 1 (TX).
//     Unused: 2,13
//     Pin 2 has an soldering hole on the board, 
//           easy to connect a wire.
//     Pin 13 is also connected to the system led.
// I2C is possible, but SPI is not possible since 
// those pins are used.
//

// 8-bit bus after the 74HC595 shift register 
// (not Arduino pins)
// These are used to set the direction of the bridge driver.

#include <Servo.h>


// Arduino pins for the shift register
#define MOTORLATCH 12
#define MOTORCLK 4
#define MOTORENABLE 7
#define MOTORDATA 8

#define MOTOR1_A 2
#define MOTOR1_B 3
#define MOTOR2_A 1
#define MOTOR2_B 4
#define MOTOR3_A 5
#define MOTOR3_B 7
#define MOTOR4_A 0
#define MOTOR4_B 6

// Arduino pins for the PWM signals.
#define MOTOR1_PWM 11
#define MOTOR2_PWM 3
#define MOTOR3_PWM 6
#define MOTOR4_PWM 5
#define SERVO1_PWM 10
#define SERVO2_PWM 9

// Codes for the motor function.
#define FORWARD 1
#define BACKWARD 2
#define BRAKE 3
#define RELEASE 4

#define WHEEL_1 1
#define WHEEL_2 2
#define WHEEL_3 3
#define WHEEL_4 4

void setup() {
  // put your setup code here, to run once:
  pinMode(MOTOR1_PWM,OUTPUT) ;    //we have to set PWM pin as output
  pinMode(MOTOR1_B,OUTPUT) ;  //Logic pins are also set as output
  pinMode(MOTOR1_A,OUTPUT) ;
  pinMode(MOTOR2_PWM,OUTPUT) ;    //we have to set PWM pin as output
  pinMode(MOTOR2_A,OUTPUT) ;  //Logic pins are also set as output
  pinMode(MOTOR2_B,OUTPUT) ;
  pinMode(MOTOR3_A, OUTPUT);
  pinMode(MOTOR3_B, OUTPUT);
  pinMode(MOTOR3_PWM,OUTPUT) ;
  pinMode(MOTOR4_A, OUTPUT);
  pinMode(MOTOR4_B, OUTPUT); 
  pinMode(MOTOR3_PWM,OUTPUT) ; 
  pinMode(MOTORLATCH, OUTPUT);
  pinMode(MOTORENABLE, OUTPUT);
  pinMode(MOTORDATA, OUTPUT);
  pinMode(MOTORCLK, OUTPUT);  
  
  digitalWrite(MOTOR1_B, LOW); 
  digitalWrite(MOTOR1_A, LOW);
  digitalWrite(MOTOR2_B, LOW); 
  digitalWrite(MOTOR2_A, LOW);
  digitalWrite(MOTOR3_B, LOW); 
  digitalWrite(MOTOR3_A, LOW);
  digitalWrite(MOTOR4_B, LOW); 
  digitalWrite(MOTOR4_A, LOW);
  
  Serial.begin(9600);
}

void goForward(int spd, int wheel){
  int fwdPin;
  int revPin;
  int speedPin;
  fwdPin = getWheelFwdPin(wheel);
  revPin = getWheelRevPin(wheel);
  speedPin = getWheelSpeedPin(wheel);
  if (fwdPin == -1 || revPin == -1 || speedPin == -1){
    return; //Invalid Wheel Specified
  }  
  digitalWrite(revPin, LOW); //For Safety make sure directRev is set LOW before setting directFwd HIGH
  digitalWrite(fwdPin , HIGH);
  digitalWrite(speedPin,spd) ;
  //analogWrite(speedPin,spd) ;
  Serial.println("Go Forward");
  Serial.println(fwdPin);
  Serial.println(revPin); 
}
void goBackward(int spd, int wheel){
  int fwdPin;
  int revPin;
  int speedPin;
  fwdPin = getWheelFwdPin(wheel);
  revPin = getWheelRevPin(wheel);
  speedPin = getWheelSpeedPin(wheel);
  if (fwdPin == -1 || revPin == -1 || speedPin == -1){
    return; //Invalid Wheel Specified
  }  
 Serial.println("Go Back");
 Serial.println(fwdPin);
 Serial.println(revPin);
 digitalWrite(fwdPin, LOW);  //For Safety make sure directFwd is set LOW before setting directFwd HIGH
 digitalWrite(revPin, HIGH);
 analogWrite(speedPin,spd) ;
// digitalWrite(MOTOR1_A, LOW);  //For Safety make sure directFwd is set LOW before setting directFwd HIGH
// digitalWrite(MOTOR1_B, HIGH);
// analogWrite(MOTOR1_PWM,spd) ;  
}

int getWheelFwdPin(int wheel){
  if (wheel == WHEEL_1){
    return MOTOR1_A;      
  }
  else if (wheel == WHEEL_2){
    return MOTOR2_A;           
  }
   else if (wheel == WHEEL_3){
    return MOTOR3_A;           
  }
   else if (wheel == WHEEL_4){
    return MOTOR4_A;           
  }
  else{
     return -1;
  }  
}
int getWheelRevPin(int wheel){
  if (wheel == WHEEL_1){
    return MOTOR1_B;      
  }
  else if (wheel == WHEEL_2){
    return MOTOR2_B;           
  }
  else if (wheel == WHEEL_3){
    return MOTOR3_B;           
  }
  else if (wheel == WHEEL_4){
    return MOTOR4_B;           
  }
  else{
     return -1;
  }  
}
int getWheelSpeedPin(int wheel){
    if (wheel == WHEEL_1){
    return MOTOR1_PWM;      
  }
  else if (wheel == WHEEL_2){
    return MOTOR2_PWM;           
  }
  else if (wheel == WHEEL_3){
    return MOTOR3_PWM;           
  }
  else if (wheel == WHEEL_4){
    return MOTOR4_PWM;           
  }
  else{
     return -1;
  }  
}
void haltMotor(int wheel){
  int fwdPin;
  int revPin;
  int speedPin;
  fwdPin = getWheelFwdPin(wheel);
  revPin = getWheelRevPin(wheel);
  speedPin = getWheelSpeedPin(wheel);
  if (fwdPin == -1 || revPin == -1 || speedPin == -1){
    return; //Invalid Wheel Specified
  }
  digitalWrite(MOTOR1_A, LOW);  
  digitalWrite(MOTOR1_B, LOW);  
  analogWrite(speedPin,0) ; 
  
}
void turnRight(){
  goForward(255, WHEEL_1);
  goForward(80, WHEEL_2);
}
void loop() {
  // put your main code here, to run repeatedly:
  //goForward(255, WHEEL_1);
 //  goBackward(255, WHEEL_2);
  goForward(255,WHEEL_3);
  goForward(255,WHEEL_2);
  delay(5000);
  haltMotor(WHEEL_1);
  haltMotor(WHEEL_2);
  delay(1000);
//  goForward(255, WHEEL_2);
//  goBackward(255, WHEEL_1);
//  delay(5000);
//  haltMotor(WHEEL_1);
//  haltMotor(WHEEL_2);
//  delay(1000);
}
