# Yelocar

<img src = "https://user-images.githubusercontent.com/84055592/188574698-191c8a9a-71bf-47b4-b862-b13b37f09c2a.png" width="70%" height="70%"/>


#### This project presents a hybrid control of autonomous robot vehicle using behavior cloning algorithm with PID control for enhanced driving performance. The behavior cloning algorithm based on convolutional neural network (CNN) mimics human driving behaviors for autonomous driving by obtaining and learning manual (human) control data. Raspberry Pi camera v2 and Jestson nano are used in this project. With a CNN based behavior cloning model, a PID control has been combined as a lateral controller to propose hybrid control architecture which improves the performance of trajectory tracking in various environments.
#### Driving video: <https://youtu.be/fZ85c0ost3g>

## Vehicle Configuration


<img src = "https://user-images.githubusercontent.com/84055592/188573338-a36622cb-4f64-4ebf-866f-52eb01eafb0f.png" width="70%" height="70%"/>

#### The tested robot vehicle is equipped with RaspberryPi camera v2 and Jetson Nano. PCA9685 is used to send PWM (Pulse Width Modulation) signals from Jetson Nano to lower-level controllers by I2C (Inter-Integrated Circuit). The lower-level controller consists of brushed DC motor and servo motor which are operated by PWM signals. The Arduino UNO encodes PWM signals of throttle and steering, which is input as a string data, into byte. An encoder has been attached to acquire a speed of the robot vehicle.

## Hybrid Control


* Architecture of Hybrid Control
<img src = "https://user-images.githubusercontent.com/84055592/188573621-55d72c83-f32c-4dfd-896f-b92a5b54d69f.png" width="70%" height="70%"/>    

* The Example of Input Steering Control Ratio
<img src = "https://user-images.githubusercontent.com/84055592/188576524-79c9030e-3d85-4f0d-928a-89cd44962f71.png" width="60%" height="60%"/>
