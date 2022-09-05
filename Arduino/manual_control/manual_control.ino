int steer = 7;
int throttle = 8;
unsigned long steer_duration;
unsigned long steer_overall_duration;
unsigned long throttle_duration;
unsigned long throttle_overall_duration;

void setup() {
  Serial.begin(9600);
  pinMode(steer, INPUT);
  pinMode(throttle, INPUT);
}

void loop() {
  steer_duration = pulseInLong(steer, HIGH);
  steer_overall_duration = pulseIn(steer, LOW);
  steer_overall_duration += steer_duration;
  throttle_duration = pulseInLong(throttle, HIGH);
  throttle_overall_duration = pulseIn(throttle, LOW);
  throttle_overall_duration += throttle_duration;
  Serial.print ("steer_duration : ");
  Serial.print (steer_duration);
  Serial.print ("/");
  Serial.print (steer_overall_duration);
  Serial.print (" throttle_duration : ");
  Serial.print (throttle_duration);
  Serial.print ("/");
  Serial.println (throttle_overall_duration);
}
