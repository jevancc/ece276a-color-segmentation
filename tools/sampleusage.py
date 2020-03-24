import numpy as np
import matplotlib.pyplot as plt
# Read labeled data from .npz file
labeled_data_5 = np.load('../trainset/5.npz')

# Get "COLOR_STOP_SIGN_RED" pixels. It will return a numpy array in shape (N_PIXELS, 3).
# All of the pixels are represented in RGB color (data[i, 0] for the R channel of ith pixel, data[i, 1] for G chanel of ith pixel)
# You can access other color with indexes as follows:
# - COLOR_STOP_SIGN_RED - Stop sign red
# - COLOR_OTHER_RED - red but not stop sign red
# - COLOR_BROWN - brown
# - COLOR_ORANGE - orange
# - COLOR_BLUE - blue
# - COLOR_OTHER - all the other colors
pixels_stopsign = labeled_data_5['COLOR_STOP_SIGN_RED']
pixels_brown = labeled_data_5['COLOR_BROWN']

# Get the mask of stop sign regions.
# It will return a binary numpy array in shape (M, N), where M, N is the height and width of the image
# The element of the mask is 1 if the corresponding pixel in the image belongs to a stop sign.
# You can use cv2 to find bounding boxes of the stop signs on the mask
img = cv2.imread('../trainset/5.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img[data['MASK_STOP_SIGN'] == 0, :] = 0
plt.imshow(img)
