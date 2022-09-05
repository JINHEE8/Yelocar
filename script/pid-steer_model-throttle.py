#!/usr/bin/env python

import time
import matplotlib.pyplot as plt
from simple_pid import PID
import cv2
import busio
from adafruit_pca9685 import PCA9685
from board import SCL, SDA
import tensorflow as tf
import numpy as np
import argparse
import time
import math
import serial
import datetime

hsv_lower = np.array([20, 90, 123])
hsv_upper = np.array([50, 255, 255])

def gstreamer_pipeline(
    capture_width=64,
    capture_height=48,
    display_width=64,
    display_height=48,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
class Yelocar:

    def __init__(self):
        self.theta_cam = 0
        self.steer1 = 0

    # Map using look up table
    # input : theta, output : steering_value
    def update(self, delta):
        if delta >= 0 and delta <= (17*math.pi/180):
            self.steer1 = 0.4779-(delta*(0.4779-0.2)/(17*math.pi/180))
    
        elif delta > (17*math.pi/180):
            self.steer1 = 0.2

        elif delta < 0 and delta >= ((-21)*math.pi/180):
            self.steer1 = 0.4779+(abs(delta)*(0.8-0.4779)/(21*math.pi/180))

        elif delta < ((-21)*math.pi/180):
            self.steer1 = 0.8    
            
        else:
            self.steer1 = int(0x2380)

        return self.steer1



    # Control with pid
    @staticmethod
    def pid_vehicle(theta_cam, last_time,vel, model):

        masked = cv2.bitwise_and(frame, frame, mask=mask)
        masked = np.reshape(masked, (1, 48, 64, 3))
        masked = np.float32(masked)
        masked = np.true_divide(masked, 255.)

        output = model(masked)[0]

        theta_dot = pid(theta_cam)

        if vel > 0:
            theta = theta_dot*0.3/vel

        else:
            theta = 0
            
        print('theta_cam : {:.4f}, theta_dot : {:.4f}, theta : {:.4f}, velocity : {:.4f}'.format(theta_cam, theta_dot, theta,vel))
        f1.write('theta_cam : {:.4f}, theta_dot : {:.4f}, theta : {:.4f}, velocity : {:.4f}\n'.format(theta_cam, theta_dot, theta,vel))

        steer = yelocar.update(theta)
        throttle = output[1]

        return steer, throttle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trained model test code")
    parser.add_argument('model_path', metavar='model_path', type=str, nargs=1, help="path for model directory")
    args = parser.parse_args()

    i2c_bus = busio.I2C(SCL, SDA)

    # Create a simple PCA9685 class instance.
    pca = PCA9685(i2c_bus)

    # Set the PWM frequency to 60hz.
    pca.frequency = 90

    left = 0x1B30
    center = 0x2380
    right = 0x2E60
    forward = 0x2D40
    stop = 0x2140
    backward = 0x1A70

    pca.channels[0].duty_cycle = center
    pca.channels[1].duty_cycle = stop

    width = 640
    height = 480
    fps = 30

    model = tf.saved_model.load(args.model_path[0])

    yelocar = Yelocar()
    theta_cam = yelocar.theta_cam

    pid = PID(1.5, 0.2, 0.2, setpoint=0)    # 0
    pid.output_limits = (-1, 1)
    pid._max_output = 1

    start_time = time.time()
    last_time = start_time

    # Keep track of values for plotting
    setpoint, y, x = [], [], []

    # Take out .. from control_vehicle function
    now = time.localtime()
    log1 = '%02d-%02d-%02d-%02d-pid-steer_model-throttle' % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min) + '.log'

    f1 = open('./pid-steer_model-throttle/' + log1, 'w')

    cap = cv2.VideoCapture(gstreamer_pipeline(capture_width=640, capture_height=480, display_width=640, display_height=480, framerate=30), cv2.CAP_GSTREAMER)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    # Initialize theta = 0
    #i = 1
    count = 1
    r = 0.05 #radius[m]
    with serial.Serial('/dev/ttyUSB0', 9600, timeout=10) as ser:	# open serial port
        ser.write(bytes('YES\n', 'utf-8'))

        if cap.isOpened():
            window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
            while cv2.getWindowProperty("CSI Camera", 0) >= 0:

                ser_in = ser.readline().decode('utf-8', 'ignore')

                rpm = int(ser_in.split(' ')[0])
                velocity = r*2*math.pi*rpm/60 #rpm contver to m/s

                ret_val, img = cap.read()
                cv2.imshow("CSI Camera", img)

                frame = np.array(cv2.resize(img, (64, 48)))
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
                flag = 0
                start = 0
                end = 0

                # Calculate theta
                # Scan from (40, 0) to (40, 60)

                for x in range(0, 64) :

                    #Yellow
                    if (20 <= hsv[28, x][0] <=50) and (90 <= hsv[28, x][1] <=255) and (123 <= hsv[28, x][2] <=255) :
                        if (flag != 1):
                            #first x coordinate in yellow lane
                            print('start x : ',x, ', hsv : ', hsv[40, x])
                            start = x

                        if (x == 63) :
                            print('63 end x : ',x, ', hsv : ', hsv[40, x-1])
                            end = 63
                            break

                        flag = 1

                    elif (flag == 1) :
                        #last x coordinate in yellow lane
                        print('end x : ',x-1, ', hsv : ', hsv[40, x-1])
                        end = x-1
                        break

                mid = int((start+end)/2)
                hsv[40, mid] = [0, 99, 100]
        

                mid_flag = 0
                if (mid == 0) :
                    mid_flag = 1    #detected yellow lane nowhere in image
                    mid = 31

                theta_cam = (math.atan((mid-31)/20))
                print("theta_cam : ", theta_cam)

                current_time = time.time()
                print(current_time - start_time)
                print(current_time - last_time)
                print("count = ", count)


                steer, throttle = yelocar.pid_vehicle(theta_cam, last_time,velocity, model)

                if(mid_flag == 1) :
                    mid = 'null'
                    theta_cam = 0.0

                if (mid == 31) :
                    theta_cam = 0.0


                now = time.localtime()
                now_time = '%02d:%02d:%02d' % (now.tm_hour, now.tm_min, now.tm_sec)


                f1.write('pid >> time : {}, steer : {:.4f}, throttle : {:.4f}, theta_cam : {:.4f}, mid_x : {}, '.format(now_time, steer, throttle, theta_cam, mid))
                print('pid >> time : {}, steer : {:.4f}, throttle : {:.4f}, theta_cam : {:.4f}, mid_x : {}, '.format(now_time, steer, throttle, theta_cam, mid))


                steer = int(steer * (1888 - 1108) + 1108)
                throttle = int(throttle * (1840 - 1352) + 1352)

                if int(steer) <= 1000:
                    steer = int(0x2380)
                elif int(steer) <= 1444:
                    steer = int(((0x2380 - 0x1B30) / (1444 - 1108)) * (steer - 1444) + 0x2380)
                elif int(steer <= 2000):
                    steer = int(((0x2E60 - 0x2380) / (1888 - 1444)) * (steer - 1444) + 0x2380)
                else:
                    steer = int(center)

                if int(throttle) <= 1000:
                    throttle = int(0x2140)
                elif int(throttle) >= 1352:
                    throttle = int(((0x2D40 - 0x2140) / (1840 - 1352)) * (throttle - 1352) + 0x2140)
                else:
                    throttle = int(((0x2140 - 0x1A70) / (1352 - 1076)) * (throttle - 1352) + 0x2140)

                pca.channels[0].duty_cycle = steer
                pca.channels[1].duty_cycle = throttle

                print("finished pca control")
                # i = i+1
                count = count+1
                print("count = ", count)

                last_time = time.time()

    
        else:
            print("Unable to open camera")
            pass
    f1.close()

    cap.release()
    cv2.destroyAllWindows()
