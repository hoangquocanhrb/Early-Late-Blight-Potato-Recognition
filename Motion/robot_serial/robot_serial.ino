
#define IN7 7 //new one 
#define IN6 6 //new one
#define IN5 5 //old one
#define IN4 4 //old one
#define IN3 3 //old two
#define IN2 2 //old two
#define MAX_SPEED 255 //từ 0-255
#define MIN_SPEED 0

void setup() {
pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(IN5, OUTPUT);
  pinMode(IN6, OUTPUT);
  pinMode(IN7, OUTPUT);
  Serial.begin(9600);  
}

String control = "stop";
String Signal = "stop";
int speed = 190;

// the loop function runs over and over again forever
void loop() {
     if (Serial.available() > 0) {
        Signal = Serial.readString();
        // split string to get action, speed
        // action-speed
        int action_id = Signal.indexOf('-');
        
        control = Signal.substring(0, action_id);
        String str_speed = Signal.substring(action_id+1, Signal.length());
        speed = str_speed.toInt();
        // Serial.print("Received: ");
        Serial.print(control);
        Serial.print(' ');
        Serial.println(speed);
    }    
    Serial.println(control);
    if(control == "trai_tien") {
      trai_tien(speed);
    }
    else if (control == "trai_lui"){
      trai_lui(speed);
    }
    else if (control == "trai_dung"){
      trai_dung();
    }
    else if (control == "phai_tien"){
      phai_tien(speed);
    }
    else if (control == "phai_lui"){
      phai_lui(speed);
    }
    else if (control == "phai_dung"){
      phai_dung();
    }
    else if (control == "stop"){
      stop_motion();
    }
    else if (control == "quay_trai"){
      quay_trai(speed);
    }
    else if (control == "quay_phai"){
      quay_phai(speed);
    }
    else if (control == "di_lui"){
      di_lui(speed);
    }
    else if (control == "di_thang"){
      di_thang(speed);
    }
    delay(500);
             
}


void trai_lui(int speed)
{
speed = constrain(speed, MIN_SPEED, MAX_SPEED);
digitalWrite(IN3, LOW);
analogWrite(IN2, speed);
}

void trai_tien(int speed){
  speed = constrain(speed, MIN_SPEED, MAX_SPEED);
analogWrite(IN3, speed);
digitalWrite(IN2, LOW);
}

void trai_dung(){
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
}

void phai_tien(int speed){
  speed = constrain(speed, MIN_SPEED, MAX_SPEED);
  digitalWrite(IN4, LOW);
  analogWrite(IN5, speed);
  digitalWrite(IN7, LOW);
  analogWrite(IN6, int(speed/8));
}

void phai_lui(int speed){
  speed = constrain(speed, MIN_SPEED, MAX_SPEED);
  analogWrite(IN4, speed);
  digitalWrite(IN5, LOW);
  analogWrite(IN7, int(speed/8));
  digitalWrite(IN6, LOW);
}

void phai_dung(){
  digitalWrite(IN4, LOW);
  digitalWrite(IN5, LOW);
  digitalWrite(IN6, LOW);
  digitalWrite(IN7, LOW);
}

void stop_motion(){
  trai_dung();
  phai_dung();
}

void quay_trai(int speed){
  speed = constrain(speed, MIN_SPEED, MAX_SPEED);
  trai_lui(speed);
  phai_tien(speed);
}
void quay_phai(int speed){
  speed = constrain(speed, MIN_SPEED, MAX_SPEED);
  trai_tien(speed);
  phai_lui(speed);
}
void di_lui(int speed){
  speed = constrain(speed, MIN_SPEED, MAX_SPEED);
  trai_lui(speed);
  phai_lui(speed);
}
void di_thang(int speed){
  speed = constrain(speed, MIN_SPEED, MAX_SPEED);
  trai_tien(speed);
  phai_tien(speed);
}
