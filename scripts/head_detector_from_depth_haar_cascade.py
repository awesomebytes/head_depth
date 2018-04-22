#!/usr/bin/env python

import rospy
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
from rospkg import RosPack
from cv_bridge import CvBridge

"""
Haar cascade based head detection via depth image.

Author: Sammy Pfeiffer <Sammy.Pfeiffer at student.uts.edu.au>
"""


class HeadDetectorFromDepth(object):
    def __init__(self, haar_cascade_file=None):
        # Load the classifier
        if haar_cascade_file:
            self.classifier = cv2.CascadeClassifier(haar_cascade_file)
        else:
            rp = RosPack()
            pkg_path = rp.get_path('head_depth')
            self.classifier = cv2.CascadeClassifier(
                pkg_path +
                "/file/haarcascades/haarcascade_range_multiview_5p_bg.xml")
        # Subscribe to the topics
        self.bridge = CvBridge()
        self.last_depth = None
        self.depth_sub = rospy.Subscriber(
            '/pepper/camera/depth/image_raw/compressedDepth',
            CompressedImage, self.depth_cb, queue_size=1)
        self.last_rgb = None
        self.rgb_sub = rospy.Subscriber(
            '/pepper/camera/front/image_raw/compressed',
            CompressedImage, self.compressed_rgb_cb, queue_size=1)

    def depth_cb(self, data):
        self.last_depth = data

    def compressed_rgb_cb(self, data):
        self.last_rgb = data

    def run(self):
        rospy.loginfo("Running!")
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.last_depth and self.last_rgb:
                # # convert to cv2, need this little hack to avoid conversion error
                # self.last_depth.encoding = "mono16"
                # img = self.bridge.imgmsg_to_cv2(self.last_depth, "mono8")
                img = self.adapt_depth_image(self.last_depth)
                heads = self.classifier.detectMultiScale(img, 1.1, 6,
                                                         (cv2.cv.CV_HAAR_DO_CANNY_PRUNING),
                                                         (20, 20))
                print("# " + str(len(heads)) + " heads.")
                # rgbimg = self.bridge.compressed_imgmsg_to_cv2(
                #     self.last_rgb, "passthrough")
                # for (x, y, w, h) in heads:
                #     cv2.rectangle(rgbimg,
                #                   (x, y),
                #                   (x + w, y + h),
                #                   (255, 0, 0), 2)
                # cv2.imshow('Detections', rgbimg)
                # cv2.waitKey(3)
            r.sleep()

    def adapt_depth_image(self, compressed_image):
        # Trick to recompose the compressedDepth image
        depth_header_size = 12
        raw_data = compressed_image.data[depth_header_size:]
        depth_img_raw = cv2.imdecode(np.frombuffer(raw_data, np.uint8),
                                     cv2.IMREAD_UNCHANGED)

        # Recompose as a ROS image (we need to, to be able to transform it later)
        mono16_image = self.bridge.cv2_to_imgmsg(depth_img_raw, "mono16")
        # OpenCV needs mono8
        return self.bridge.imgmsg_to_cv2(mono16_image, "mono8")


if __name__ == '__main__':
    rospy.init_node("head_detector_from_depth")
    hdfd = HeadDetectorFromDepth(haar_cascade_file=None)
    hdfd.run()
