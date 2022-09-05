import os
import cv2
import datetime
import numpy as np
import h5py

image_list = []
action_list = []

hsv_lower = np.array([20, 90, 123])
hsv_upper = np.array([50, 255, 255])

def make_dataset(path, filename):
    print(filename)
    image_list = []
    action_list = []

    logpath = path+filename+'.log'
    avipath = path+filename+'.mp4'

    logfile = open(logpath, 'r')
    avifile = cv2.VideoCapture(avipath)

    loglines = logfile.readlines()

    current_avi_time = 0
    start_time = datetime.datetime(int(filename[0:4]), int(filename[4:6]), int(filename[6:8]), int(filename[9:11]), int(filename[11:13]), int(filename[13:15]))

    i = 0

    for line in loglines:
        i += 1
        logdata = line.split(' ')
        date = logdata[0].split('-')
        time = logdata[1].split(':')
        log_time = datetime.datetime(int(date[0]), int(date[1]), int(date[2]), int(time[0]), int(time[1]), int(time[2].split('.')[0]), int(time[2].split('.')[1]))

        delt_time = (log_time-start_time).total_seconds()

        image = None
        steer = (int(logdata[4].split('/')[0]) - 1108) / (1888 - 1108)
        throttle = (int(logdata[7].split('/')[0]) - 1352) / (1840 - 1352)

        while(avifile.isOpened() and current_avi_time < delt_time):
            ret, frame = avifile.read()
            if ret == True:
                current_avi_time += 1/30
                if current_avi_time > delt_time:
                    image = cv2.resize(frame, (64, 48))

                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
                    masked = cv2.bitwise_and(image, image, mask=mask)
                    # cv2.imshow('frame',masked)
                    # cv2.waitKey(1)

                    image_list.append(masked)
                    action_list.append([steer, throttle])
            else:
                break

        if i % 1000 == 0:
            print(i)
    print('read finished')

    avifile.release()
    cv2.destroyAllWindows()
    logfile.close()

    print('make temp dataset')
    image_train = np.array(image_list[:int(len(image_list)*0.8)])
    action_train = np.array(action_list[:int(len(image_list)*0.8)])
    image_val = np.array(image_list[int(len(image_list)*0.8):])
    action_val = np.array(action_list[int(len(image_list)*0.8):])

    train_len = len(image_train)
    val_len = len(image_val)

    with h5py.File('./'+filename+'_low_temp.hdf5', 'w') as hf:
        dset_x_train = hf.create_dataset('x_train', (train_len, 48, 64, 3), chunks=True, compression="gzip")
        dset_y_train = hf.create_dataset('y_train', (train_len, 2), chunks=True, compression="gzip")
        dset_x_val = hf.create_dataset('x_val', (val_len, 48, 64, 3), chunks=True, compression="gzip")
        dset_y_val = hf.create_dataset('y_val', (val_len, 2), chunks=True, compression="gzip")

        dset_x_train[:, :, :, :] = image_train
        dset_y_train[:, :] = action_train
        dset_x_val[:, :, :, :] = image_val
        dset_y_val[:, :] = action_val

    print(filename + ' finished')


if __name__ == '__main__':
    path = './Data/'

    files = os.listdir(path)

    files.remove('.gitignore')

    for i in range(0, len(files), 2):
        make_dataset(path, files[i].split('.')[0])

    image_train = None
    action_train = None
    image_val = None
    action_val = None

    for i in range(0, len(files), 2):
        filename = files[i].split('.')[0]
        with h5py.File('./'+filename+'_low_temp.hdf5', 'r') as hf:
            print(filename)
            if image_train is None:
                image_train = np.array(hf['x_train'][:, :, :, :])
                action_train = np.array(hf['y_train'][:, :])
                image_val = np.array(hf['x_val'][:, :, :, :])
                action_val = np.array(hf['y_val'][:, :])
            else:
                image_train = np.concatenate((image_train, hf['x_train']), axis=0)
                action_train = np.concatenate((action_train, hf['y_train']), axis=0)
                image_val = np.concatenate((image_val, hf['x_val']), axis=0)
                action_val = np.concatenate((action_val, hf['y_val']), axis=0)

    train_len = len(image_train)
    val_len = len(image_val)

    with h5py.File('./Model/dataset_hsv_00.hdf5', 'w') as hf:
        dset_x_train = hf.create_dataset('x_train', (train_len, 48, 64, 3), chunks=True, compression="gzip")
        dset_y_train = hf.create_dataset('y_train', (train_len, 2), chunks=True, compression="gzip")
        dset_x_val = hf.create_dataset('x_val', (val_len, 48, 64, 3), chunks=True, compression="gzip")
        dset_y_val = hf.create_dataset('y_val', (val_len, 2), chunks=True, compression="gzip")

        dset_x_train[:, :, :, :] = image_train
        dset_y_train[:, :] = action_train
        dset_x_val[:, :, :, :] = image_val
        dset_y_val[:, :] = action_val

    # dataset = tf.data.Dataset.from_tensors((image_feature[i], action_feature[i]) for i in range(len(image_feature)))
    # tf.data.experimental.save(dataset, path+'saved_data')