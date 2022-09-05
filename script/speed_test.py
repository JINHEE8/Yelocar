import serial
import datetime
import time
import busio
from board import SCL, SDA
from multiprocessing import Process

# Import the PCA9685 module.
from adafruit_pca9685 import PCA9685

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
pca.channels[1].duty_cycle = 0x21D0