#define pinPUL 8
#define pinDIR 9
#define pinENA 10

void setup() 
{
  pinMode(pinPUL,OUTPUT);
  pinMode(pinDIR, OUTPUT);
  pinMode(pinENA, OUTPUT);
}

void loop() {
  
  forward(100);
  reverse(100);
  delay(60000)
}

void forward(int steps) {
  int i;                  
  digitalWrite(pinENA,LOW); // LOW - Enable motor
  digitalWrite(pinDIR,LOW); // To set direction
  for (i=0;i<steps;i++) {
    digitalWrite(pinPUL,!digitalRead(pinPUL));
    delay(10);
  }
  digitalWrite(pinENA,HIGH); // HIGH - Disable motor
}

void reverse(int steps) {
  int i;                  
  digitalWrite(pinENA,LOW); // LOW - Enable motor
  digitalWrite(pinDIR,HIGH); // To set direction
  for (i=0;i<steps;i++) {
    digitalWrite(pinPUL,!digitalRead(pinPUL));
    delay(10);
  }
  digitalWrite(pinENA,HIGH); // HIGH - Disable motor
}
