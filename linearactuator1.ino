#define pinSTP 3
#define pinDIR 2
#define pinENA 5
void setup() 
{
  pinMode(pinSTP,OUTPUT);
  pinMode(pinDIR, OUTPUT);
  pinMode(pinENA, OUTPUT);

  pinMode(LED_BUILTIN, OUTPUT);
  
  digitalWrite(pinSTP,HIGH);
  digitalWrite(pinDIR,LOW);
  digitalWrite(pinENA,255);
  delay(2000);
}

void loop() 
{
  digitalWrite(LED_BUILTIN, HIGH);   
  delay(1000);                       
  digitalWrite(LED_BUILTIN, LOW);                           
  
  
  digitalWrite(pinSTP,LOW);
  digitalWrite(pinDIR,LOW);
  delay(1000); 
}
