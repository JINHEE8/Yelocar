import serial
import datetime
import time
import busio
from board import SCL, SDA
import argparse

# Import the PCA9685 module.
from adafruit_pca9685 import PCA9685

# Create the I2C bus interface.
i2c_bus = busio.I2C(SCL, SDA)

# Create a simple PCA9685 class instance.
pca = PCA9685(i2c_bus)

# Set the PWM frequency to 60hz.
pca.frequency = 90

# Set the PWM duty cycle for channel zero to 50%. duty_cycle is 16 bits to match other PWM objects
# but the PCA9685 will only actually give 12 bits of resolution.
# pca.channels[0].duty_cycle = 0x1B30	# servo left, 1108/10280
# pca.channels[0].duty_cycle = 0x2380	# servo center, 1444/10280
# pca.channels[0].duty_cycle = 0x2E60	# servo right, 1888/10280

# pca.channels[1].duty_cycle = 0x2D40	# throttle forward, 1840/10280(10400)
# pca.channels[1].duty_cycle = 0x2140	# throttle stop, 1352/10280(10400)
# pca.channels[1].duty_cycle = 0x1A70	# throttle backward, 1076/10280(10400)

left = 0x1B30
center = 0x2380
right = 0x2E60
forward = 0x2D40
stop = 0x2140
backward = 0x1A70

print('center, stop')
pca.channels[0].duty_cycle = center
pca.channels[1].duty_cycle = stop
time.sleep(5)

print('center, forward')
pca.channels[0].duty_cycle = center
pca.channels[1].duty_cycle = forward
time.sleep(5)

print('center, stop')
pca.channels[0].duty_cycle = center
pca.channels[1].duty_cycle = stop
time.sleep(5)

print('center, backward')
pca.channels[0].duty_cycle = center
pca.channels[1].duty_cycle = backward
time.sleep(5)

print('center, stop')
pca.channels[0].duty_cycle = center
pca.channels[1].duty_cycle = stop