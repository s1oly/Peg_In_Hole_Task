#!/usr/bin/env python3

import cv2
import numpy as np
from threading import Thread, Lock
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
import re


def flat_field_correction(f0, img, counter, num_init_frames, KJ):

    h,w = img.shape[0], img.shape[1]

    mean = np.mean(f0, axis=0)
    J = img / mean

    # imgffc[:,:,0] = (imgffc[:,:,0] - imgffc[:,:,0].min())/ (imgffc[:,:,0].max() - imgffc[:,:,0].min())
    # imgffc[:, :, 1] = (imgffc[:, :, 1] - imgffc[:, :, 1].min()) / (imgffc[:, :, 1].max() - imgffc[:, :, 1].min())
    # imgffc[:, :, 2] = (imgffc[:, :, 2] - imgffc[:, :, 2].min()) / (imgffc[:, :, 2].max() - imgffc[:, :, 2].min())

    J_min = np.tile(np.min(np.min(J, axis=1), axis=0).reshape(1,-1),(h,w,1))
    J_max = np.tile(np.max(np.max(J, axis=1), axis=0).reshape(1,-1),(h,w,1))

    R = (J - J_min) / (J_max - J_min)
    aa = np.sum(np.sum(J, axis=1), axis=0) / np.sum(np.sum(R, axis=1), axis=0)
    K = np.tile(aa.reshape(1,-1),(h,w,1))
    J = K*R

    if counter==num_init_frames:
        KJ = np.round(np.mean(255 / np.max(np.max(J, axis=1), axis=0).reshape(1, -1)))

    J = KJ * J
    J[J<0] = 0
    J[J>255] = 255

    return J.astype('uint8'), KJ



def get_diff_img_2(img1, img2):
    return (img1 * 1.0 - img2) / 255.  +  0.5


def get_camera_id(camera_name):
    """
    Find the camera ID that has the corresponding camera name.

    :param camera_name: str; The name of the camera.
    :return: int; The camera ID.
    """
    cam_num = None
    for file in os.listdir("/sys/class/video4linux"):
        real_file = os.path.realpath("/sys/class/video4linux/" + file + "/name")
        with open(real_file, "rt") as name_file:
            name = name_file.read().rstrip()
        if camera_name in name:
            cam_num = int(re.search("\d+$", file).group(0))
            found = "FOUND!"
        else:
            found = "      "
        print("{} {} -> {}".format(found, file, name))

    return cam_num


def resize_crop(img, imgw, imgh):
    """
    Resize and crop the image to the desired size.

    :param img: np.ndarray; The image to resize and crop.
    :param imgw: int; The width of the desired image.
    :param imgh: int; The height of the desired image.
    :return: np.ndarray; The resized and cropped image.
    """
    # remove 1/7th of border from each size
    border_size_x, border_size_y = int(img.shape[0] * (1 / 7)), int(
        np.floor(img.shape[1] * (1 / 7))
    )
    cropped_imgh = img.shape[0] - 2 * border_size_x
    cropped_imgw = img.shape[1] - 2 * border_size_y
    # Extra cropping to maintain aspect ratio
    extra_border_h = 0
    extra_border_w = 0
    if cropped_imgh * imgw / imgh > cropped_imgw + 1e-8:
        extra_border_h = int(cropped_imgh - cropped_imgw * imgh / imgw)
    elif cropped_imgh * imgw / imgh < cropped_imgw - 1e-8:
        extra_border_w = int(cropped_imgw - cropped_imgh * imgw / imgh)
    # keep the ratio the same as the original image size
    img = img[
        border_size_x + extra_border_h : img.shape[0] - border_size_x,
        border_size_y + extra_border_w : img.shape[1] - border_size_y,
    ]
    # final resize for the desired image size
    img = cv2.resize(img, (imgw, imgh))
    return img


class WebcamVideoStream :
    def __init__(self, src, width = 320, height = 240) :
        print(get_camera_id('GelSight Mini'))

        self.stream = cv2.VideoCapture(get_camera_id('GelSight Mini'))
        # self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
        # self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()
        self.imgw = width
        self.imgh = height

    def start(self) :
        if self.started :
            print ("already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self) :
        while self.started :
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self) :
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self) :
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback) :
        self.stream.release()


def chop_border_resize(img):
    img = img[chop_border_size:imgh - chop_border_size, chop_border_size:imgw - chop_border_size]
    img = cv2.resize(img, (imgw, imgh))
    return img



if __name__ == '__main__':

    rospy.init_node('gsmini_node', anonymous=True)

    SAVE_VIDEO_FLAG = False
    SAVE_SINGLE_IMGS_FLAG = False
    cvbridge = CvBridge()
    chop_border_size = 0
    imgh = 240
    imgw = 320
    NUM_SENSORS = 1

    gs = {}
    gs['img'] = [0] * 2
    gs['gsmini_pub'] = [0] * 2
    gs['vs'] = [0] * 2
    gs['img_msg'] = [0] * 2

    for i in range(NUM_SENSORS):
        gs['gsmini_pub'][i] = rospy.Publisher("/gelsight_{}".format(i), Image, queue_size=1)
        gs['vs'][i] = WebcamVideoStream(src=2*i + 2).start() # make sure the id numbers of the cameras are ones recognized by the computer. Default, 2 and 4

    r = rospy.Rate(25)  # 10hz
    while not rospy.is_shutdown():

        for i in range(NUM_SENSORS):
            gs['img'][i] = gs['vs'][i].read()
            # gs['img'][i] = resize_crop(gs['img'][i], gs['vs'][i].imgw, gs['vs'][i].imgh)
            gs['img'][i] = cv2.resize(gs['img'][i], (gs['vs'][i].imgw, gs['vs'][i].imgh))

        ''' publish image to ros '''
        for i in range(NUM_SENSORS):
            gs['img_msg'][i] = cvbridge.cv2_to_imgmsg(gs['img'][i], encoding="passthrough")
            gs['img_msg'][i].header.stamp = rospy.Time.now()
            gs['img_msg'][i].header.frame_id = 'map'
            gs['gsmini_pub'][i].publish(gs['img_msg'][i])

        r.sleep()

    for i in range(NUM_SENSORS): gs['vs'][i].stop()
    cv2.destroyAllWindows()