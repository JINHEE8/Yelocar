import cv2

import serial
import datetime
import time
import busio
from board import SCL, SDA
from multiprocessing import Process

# Import the PCA9685 module.
from adafruit_pca9685 import PCA9685

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen

def gstreamer_pipeline(
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
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

def show_camera():
    width = 640
    height = 480
    fps = 30

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    # print(gstreamer_pipeline(capture_width=width, capture_height=height, display_width=width, display_height=height, framerate=fps))
    cap = cv2.VideoCapture(gstreamer_pipeline(capture_width=width, capture_height=height, display_width=width, display_height=height, framerate=fps), cv2.CAP_GSTREAMER)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('Data/'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.avi', fourcc, float(30), (width, height))

    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, img = cap.read()
            cv2.imshow("CSI Camera", img)
            out.write(img)
            # This also acts as
            keyCode = cv2.waitKey(30) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    else:
        # print("Unable to open camera")
        pass

def control_logging():
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
    pca.channels[1].duty_cycle = stop

    f = open('Data/'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.log', 'a')
    f.write("date time steeringValue throttleValue \n")

    with serial.Serial('/dev/ttyUSB0', 9600, timeout=10) as ser:	# open serial port
        ser.write(bytes('YES\n', 'utf-8'))
        while True:
            ser_in = ser.readline().decode('utf-8', 'ignore')
            
            steer = int(ser_in.split(' ')[2].split('/')[0])
            throttle = int(ser_in.split(' ')[5].split('/')[0])

            f.write("{} {} {}\n".format(datetime.datetime.now(), steer, throttle))
            print("{} {}".format(datetime.datetime.now(), ser_in), end='')

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
    f.close()

if __name__ == "__main__":
    # show_camera()
    camera_proc = Process(target=show_camera, args=())
    control_proc = Process(target=control_logging, args=())

    camera_proc.start()
    control_proc.start()

    camera_proc.join()
    control_proc.join()
