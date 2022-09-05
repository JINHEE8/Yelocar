import cv2
import tensorflow as tf
import numpy as np
import time
import argparse
# from tensorflow.python.compiler.tensorrt import trt_convert as trt

# conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
# conversion_params = conversion_params._replace(
#     max_workspace_size_bytes=(1<<32))
# conversion_params = conversion_params._replace(precision_mode="FP16")
# conversion_params = conversion_params._replace(
#     maximum_cached_engines=100)

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen

window_size = 10

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

def control_vehicle(model_path):
    # # Create the I2C bus interface.
    # i2c_bus = busio.I2C(SCL, SDA)

    # # Create a simple PCA9685 class instance.
    # pca = PCA9685(i2c_bus)

    # # Set the PWM frequency to 60hz.
    # pca.frequency = 90
    
    # left = 0x1B30
    # center = 0x2380
    # right = 0x2E60
    # forward = 0x2D40
    # stop = 0x2140
    # backward = 0x1A70

    # pca.channels[0].duty_cycle = center
    # pca.channels[1].duty_cycle = stop

    width = 64
    height = 48
    fps = 30

    model = tf.saved_model.load(model_path)
    # saved_model_loaded = tf.saved_model.load(
    #     model_path)
    # graph_func = saved_model_loaded.signatures
    # frozen_func = convert_to_constants.convert_variables_to_constants_v2(
    #     graph_func)
    # model = frozen_func

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    # print(gstreamer_pipeline(capture_width=width, capture_height=height, display_width=width, display_height=height, framerate=fps))
    cap = cv2.VideoCapture(gstreamer_pipeline(capture_width=width, capture_height=height, display_width=width, display_height=height, framerate=fps), cv2.CAP_GSTREAMER)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    frames = []
    np_frames = None

    if cap.isOpened():
        # window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        # while cv2.getWindowProperty("CSI Camera", 0) >= 0:
        while True:
            camera_reading_time = time.time()

            ret_val, img = cap.read()
            cv2.imshow("CSI Camera", img)

            print('camera_reading_time: {}'.format(time.time() - camera_reading_time))
            
            image_preprocessing = time.time()

            frame = np.array(cv2.resize(img, (64, 48)))

            frames.append(list(frame))
            if len(frames) < window_size:
                cv2.waitKey(1)
                continue
            else:
                frames = frames[-window_size:]
                np_frames = np.array(frames)

            np_frames = np.reshape(np_frames, (1, window_size, 48, 64, 3))
            np_frames = np.true_divide(np_frames, 255.)
            np_frames = np.float32(np_frames)

            print('image_preprocessing: {}'.format(time.time() - image_preprocessing))

            model_prediction_time = time.time()

            output = model(np_frames)[0]

            print('model_prediction_time: {}'.format(time.time() - model_prediction_time))

            print('steer : {:.4f}, throttle : {:.4f}'.format(output[0], output[1]))

            # steer = int(steer * (1888 - 1108) + 1108)
            # throttle = int(throttle * ((1840 - 1352) + 1352)

            # if int(steer) <= 1000:
            #     steer = int(0x2380)
            # elif int(steer) <= 1444:
            #     steer = int(((0x2380 - 0x1B30) / (1444 - 1108)) * (steer - 1444) + 0x2380)
            # elif int(steer <= 2000):
            #     steer = int(((0x2E60 - 0x2380) / (1888 - 1444)) * (steer - 1444) + 0x2380)
            # else:
            #     steer = int(center)

            # if int(throttle) <= 1000:
            #     throttle = int(0x2140)
            # elif int(throttle) >= 1352:
            #     throttle = int(((0x2D40 - 0x2140) / (1840 - 1352)) * (throttle - 1352) + 0x2140)
            # else:
            #     throttle = int(((0x2140 - 0x1A70) / (1352 - 1076)) * (throttle - 1352) + 0x2140)

            # pca.channels[0].duty_cycle = steer
            # pca.channels[1].duty_cycle = throttle

            # This also acts as
            keyCode = cv2.waitKey(1) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 'q':
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trained model test code")
    parser.add_argument('model_path', metavar='p', type=str, nargs=1, help="path for model directory")

    args = parser.parse_args()
    control_vehicle(args.model_path[0])
