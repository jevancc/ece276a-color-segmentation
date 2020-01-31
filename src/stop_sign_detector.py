'''
ECE276A WI20 HW1
Stop Sign Detector
'''

import os, cv2
from skimage.measure import label, regionprops
import classifier
import detector
import image
import numpy as np

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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        X = img.reshape(-1, 3)
        img_mask = self.clf.predict(X).reshape(img.shape[:2])
        img_mask = (img_mask == 0).astype(int)
        return img_mask

    def get_bounding_box(self, img):
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
        img_area = img.shape[0] * img.shape[1]
        ssclf = classifier.SSBBoxDeterministic(img_area)

        img_mask_label = np.zeros((img.shape[0], img.shape[1]))

        img_mask_normal = self.segment_image(img)
        img_mask_label_normal = label(img_mask_normal, connectivity=1)
        for region in detector.Region.find(img_mask_label_normal):
            if ssclf.predict(region):
                minr, minc, maxr, maxc = region.bbox
                img_mask_label[minr:maxr, minc:maxc] += region.convex_image

        img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        f = image.build_histogram_equalizer(img[:, :, 0], 255)
        img[:, :, 0] = f(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
        img_mask_boost = self.segment_image(img)
        img_mask_label_boost = label(img_mask_boost, connectivity=1)
        for region in detector.Region.find(img_mask_label_boost):
            if ssclf.predict(region):
                minr, minc, maxr, maxc = region.bbox
                img_mask_label[minr:maxr, minc:maxc] += region.convex_image

        for _ in range(3):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img[:, :, 1] = np.clip((img[:, :, 1].astype(float) * 1.2),
                                   a_min=0,
                                   a_max=255).astype(np.uint8)
            img[:, :, 2] = np.clip((img[:, :, 2].astype(float) * 1.2),
                                a_min=0,
                                a_max=255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
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

                boxes.append([minx, miny, maxx, maxy])

        return boxes


if __name__ == '__main__':
    my_detector = StopSignDetector()

    # folder = "trainset"
    # files = [os.path.join(folder, filename) for filename in os.listdir(folder)]
    files = ['../trainset/1.jpg']
    for filename in files:
        # read one test image
        img = cv2.imread(filename)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        mask_img = my_detector.segment_image(img)
        boxes = my_detector.get_bounding_box(img)
        #Display results:
        #(1) Segmented images
        #	 mask_img = my_detector.segment_image(img)
        #(2) Stop sign bounding box
        #    boxes = my_detector.get_bounding_box(img)
        #The autograder checks your answers to the functions segment_image() and get_bounding_box()
        #Make sure your code runs as expected on the testset before submitting to Gradescope
