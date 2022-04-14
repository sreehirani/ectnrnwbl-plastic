//Connection setup between Arduino and the controller
// Controller pin DIR - sets the director of the motor
//  LOW=Motor A=Forward HIGH=Motor B=Backward
// For this program, DIR is connected to port 2 of Arduino

// Controller pin PWM - set it to HIGH to turn on the motor; LOW to turn off
// For this program, PWM is connected to Arduino port 3

// Controller pin GND - connect it to Arduino pin GND

// Set delay timer: 1000=1 second

#define pinPMW 3
#define pinDIR 2
void setup() 
{
  pinMode(pinPMW,OUTPUT);
  pinMode(pinDIR, OUTPUT);
}

void loop()// infinite loop
{                   
  digitalWrite(pinPMW,HIGH);//Turn on power
  digitalWrite(pinDIR,LOW);
  delay(25000); //Forward for 25 seconds
  
  digitalWrite(pinDIR,HIGH);
  delay(30000); //Backward for 25 second and 5 second rest

  digitalWrite(pinPMW,LOW);//Turn off power
  delay(600000);//Delay for 10 minutes
}
