'''
ECE276A WI20 HW1
Stop Sign Detector
'''

import os, cv2
from skimage.morphology import binary_dilation
from skimage.measure import label, regionprops
import classifier
import detector
import image
import numpy as np
import itertools
from image import Image


class StopSignDetector():

    def __init__(self):
        '''
		Initilize your stop sign detector with the attributes you need,
		e.g., parameters of your classifier
		'''
        self.clf = classifier.GaussianNaiveBayes()
        self.clf.load("./gnb_300000_histeq.pic")

    def segment_image(self, img):
        '''
		Obtain a segmented image using a color classifier,
		e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture,
		call other functions in this class if needed

		Inputs:
			img - original image
		Outputs:
			mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''
        if not isinstance(img, Image):
            img = Image(img)

        X = img.rgb.data.reshape(-1, 3)
        img_mask = self.clf.predict(X).reshape(img.shape[:2])
        img_mask = (img_mask == 0).astype(int)
        return img_mask

    def get_bounding_box(self, img, return_rc_coord=False):
        '''
		Find the bounding box of the stop sign
		call other functions in this class if needed

		Inputs:
			img - original image
		Outputs:
			boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2]
			where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
			is from left to right in the image.

		Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
		'''
        if not isinstance(img, Image):
            img = Image(img)
        oimg = img.copy()

        ssclf = classifier.SSBBoxDeterministic(oimg.area)
        img_mask_label = np.zeros((oimg.nr, oimg.nc))

        img = oimg
        img_mask_normal = self.segment_image(img)
        img_mask_label_normal = label(img_mask_normal, connectivity=1)
        for region in detector.Region.find(img_mask_label_normal):
            if ssclf.predict(region):
                minr, minc, maxr, maxc = region.bbox
                img_mask_label[minr:maxr, minc:maxc] += region.convex_image

        img = oimg.ycrcb
        img = img.histogram_equalize(channel_id=0, vmin=0, vmax=255)
        histeq_img = img

        img_mask_boost = self.segment_image(img)
        img_mask_label_boost = label(img_mask_boost, connectivity=1)
        for region in detector.Region.find(img_mask_label_boost):
            if ssclf.predict(region):
                minr, minc, maxr, maxc = region.bbox
                img_mask_label[minr:maxr, minc:maxc] += region.convex_image

        for sr, vr in [[1.1, 1.1], [2.3, 1.2]]:
            img = histeq_img.hsv
            img = img.mulclip(factor=sr, channel_id=1, vmin=0, vmax=255)
            img = img.mulclip(factor=vr, channel_id=2, vmin=0, vmax=255)

            img_mask_boost = self.segment_image(img)
            img_mask_label_boost = label(img_mask_boost, connectivity=1)
            for region in detector.Region.find(img_mask_label_boost):
                if ssclf.predict(region):
                    minr, minc, maxr, maxc = region.bbox
                    img_mask_label[minr:maxr, minc:maxc] += region.convex_image

        boxes = []
        img_mask_label = (img_mask_label != 0).astype(int)
        img_mask_label = label(img_mask_label, connectivity=1)
        for region in detector.Region.find(img_mask_label):
            if ssclf.predict(region):
                minr, minc, maxr, maxc = region.bbox
                minx, maxx = minc, maxc
                miny, maxy = img.shape[0] - maxr, img.shape[0] - minr

                if return_rc_coord:
                    boxes.append([minr, minc, maxr, maxc])
                else:
                    boxes.append([minx, miny, maxx, maxy])

        return boxes


import matplotlib.pyplot as plt
if __name__ == '__main__':
    my_detector = StopSignDetector()

    # folder = "trainset"
    # files = [os.path.join(folder, filename) for filename in os.listdir(folder)]
    files = ['../trainset/0.jpg']
    for filename in files:
        # read one test image
        img = cv2.imread(filename)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        mask_img = my_detector.segment_image(img)
        boxes = my_detector.get_bounding_box(img, return_rc_coord=True)

        plt.figure()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for minr, minc, maxr, maxc in boxes:
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)

            plt.plot(bx, by, '-g', linewidth=2.5)
        plt.show()
#Display results:
#(1) Segmented images
#	 mask_img = my_detector.segment_image(img)
#(2) Stop sign bounding box
#    boxes = my_detector.get_bounding_box(img)
#The autograder checks your answers to the functions segment_image() and get_bounding_box()
#Make sure your code runs as expected on the testset before submitting to Gradescope
