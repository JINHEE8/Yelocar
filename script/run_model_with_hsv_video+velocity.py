import cv2
import busio
from adafruit_pca9685 import PCA9685
from board import SCL, SDA
import tensorflow as tf
import numpy as np
import argparse
import time
import datetime
import math
import serial


# gstreamer_pipe    line returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen

hsv_lower = np.array([20, 90, 123])
hsv_upper = np.array([50, 255, 255])

def gstreamer_pipeline(
    capture_width=64,
    capture_height=48,
    display_width=64,
    display_height=48,
    framerate=21,
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

def control_vehicle(model_path):
    # Create the I2C bus interface.
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
    # pca.channels[1].duty_cycle = stop
    pca.channels[1].duty_cycle = 0x21D0

    width = 640
    height = 480
    fps = 30

    model = tf.saved_model.load(model_path)
    model_name = model_path.split('/')[-1]
    now = time.localtime()

    log_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + model_name  + '.log'

    f_model = open('newModelLog/Model/' + log_name, 'w')

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    # print(gstreamer_pipeline(capture_width=width, capture_height=height, display_width=width, display_height=height, framerate=fps))
    
    cap = cv2.VideoCapture(gstreamer_pipeline(capture_width=width, capture_height=height, display_width=width, display_height=height, framerate=fps), cv2.CAP_GSTREAMER)
    # cap = cv2.VideoCapture(gstreamer_pipeline(capture_width=1640, capture_height=1232, display_width=640, display_height=480, framerate=30), cv2.CAP_GSTREAMER)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('newModelLog/Model/'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + model_name + '.avi', fourcc, float(10), (width, height))

    # Initialize theta = 0
    #i = 1
    count = 1
    start_time = time.time()
    velocity = 0.0

    r = 0.05 #radius[m]
    with serial.Serial('/dev/ttyUSB0', 9600, timeout=10) as ser:	# open serial port
        ser.write(bytes('YES\n', 'utf-8'))
        while (time.time() - start_time) < 360:

            if cap.isOpened():
                window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
                # Window
                while cv2.getWindowProperty("CSI Camera", 0) >= 0:

                    ser_in = ser.readline().decode('utf-8', 'ignore')

                    rpm = int(ser_in.split(' ')[0])
                    velocity = r*2*math.pi*rpm/60 #rpm contver to m/s

                    ret_val, img = cap.read()
                    cv2.imshow("CSI Camera", img)

                    frame = np.array(cv2.resize(img, (64, 48)))
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
                    masked = cv2.bitwise_and(frame, frame, mask=mask)

                    masked = np.reshape(masked, (1, 48, 64, 3))
                    masked = np.float32(masked)
                    masked = np.true_divide(masked, 255.)

                    #model_prediction_time = time.time()

                    output = model(masked)[0]

                    steer = output[0]
                    throttle = output[1] + 0.08


                    steer_model = steer
                    throttle_model = throttle


                    steer = int(steer * (1888 - 1108) + 1108)
                    throttle = int(throttle * (1840 - 1352) + 1352)

                    if int(steer) <= 1000:
                        steer = int(0x2380)
                    elif int(steer) <= 1444:    #1444
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
                                #print('start x : ',x, ', hsv : ', hsv[40, x])
                                start = x

                            if (x == 63) :
                                #print('63 end x : ',x, ', hsv : ', hsv[40, x-1])
                                end = 63
                                break

                            flag = 1

                        elif (flag == 1) :
                            #last x coordinate in yellow lane
                            #print('end x : ',x-1, ', hsv : ', hsv[40, x-1])
                            end = x-1
                            break

                    mid = int((start+end)/2)


                    theta_cam = (math.atan((mid-31)/20))

                    now = time.localtime()
                    now_time = '%02d:%02d:%02d' % (now.tm_hour, now.tm_min, now.tm_sec)

                    f_model.write('{: 4d}   {}   {:.4f}   {:.4f}   {:.4f}   {: 5.4f}   {}\n'.format(count, now_time, steer_model, throttle_model, velocity, theta_cam, mid))
                    print('{: 4d}   {}   {:.4f}   {:.4f}   {:.4f}   {: 5.4f}   {}\n'.format(count, now_time, steer_model, throttle_model, velocity, theta_cam, mid))

                    out.write(img)

                    # This also acts as
                    keyCode = cv2.waitKey(1)

                    # Stop the program on the ESC or q key
                    if keyCode == 27 or keyCode == 113:
                        # 27: ESC key, 113: 'q' key based on ASCII code
                        break

                    count = count+1

                    #last_time = time.time()

                cap.release()
                out.release()
                cv2.destroyAllWindows()
            else:
                print("Unable to open camera")
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trained model test code")
    parser.add_argument('model_path', metavar='model_path', type=str, nargs=1, help="path for model directory")

    args = parser.parse_args()
    control_vehicle(args.model_path[0])
