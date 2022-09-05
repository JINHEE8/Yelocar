#define encoder 2

unsigned int rpm;
volatile byte pulses;
unsigned long TIME;
unsigned int pulse_per_turn = 20; 
//depends on the number of slots on the slotted disc

void count(){
  // counting the number of pulses for calculation of rpm
  pulses++;
}
void setup(){
  //reset all to 0
  rpm = 0;
  pulses = 0;
  TIME = 0;

  Serial.begin(9600);
  pinMode(encoder, INPUT);// setting up encoder pin as input
  //triggering count function everytime the encoder turns from HIGH to LOW
  attachInterrupt(digitalPinToInterrupt(encoder), count, FALLING);
}

void loop(){
  if (millis() - TIME >= 100){ // updating every 0.1 second
    detachInterrupt(digitalPinToInterrupt(encoder)); // turn off trigger
    //calcuate for rpm 
    rpm = (60 *100 / pulse_per_turn)/ (millis() - TIME) * pulses;
    TIME = millis();
    pulses = 0;
    //print output 
    Serial.println(rpm);
    //trigger count function everytime the encoder turns from HIGH to LOW
    attachInterrupt(digitalPinToInterrupt(encoder), count, FALLING);
  }
}
