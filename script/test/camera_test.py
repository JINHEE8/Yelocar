import cv2
import numpy as np

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

def camera_test():
    width = 640
    height = 480
    fps = 30

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    # print(gstreamer_pipeline(capture_width=width, capture_height=height, display_width=width, display_height=height, framerate=fps))
    cap = cv2.VideoCapture(gstreamer_pipeline(capture_width=width, capture_height=height, display_width=width, display_height=height, framerate=fps), cv2.CAP_GSTREAMER)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        print("cap is opened\n")
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            print("cv2.getWindowProperty('CSI Camera', 0) >= 0\n")
            ret_val, img = cap.read()
            cv2.imshow("CSI Camera", img)

            frame = np.array(cv2.resize(img, (64, 48)))
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
            masked = cv2.bitwise_and(frame, frame, mask=mask)            

            masked = np.reshape(masked, (1, 48, 64, 3))
            masked = np.float32(masked)
            masked = np.true_divide(masked, 255.)
            # This also acts as
            keyCode = cv2.waitKey(1)
            # Stop the program on the ESC or q key
            if keyCode == 27 or keyCode == 113:
                # 27: ESC key, 113: 'q' key based on ASCII code
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")
        pass
    #f.close()

if __name__ == "__main__":
    camera_test()