import cv2
from matplotlib import pyplot as plt
import sys
import numpy as np
from util import overlay_bias_image, overlay_bias_image_addition
import blend_modes

def show_row_sum(img):
    row_sum = []

    for row in img:
        row_sum.append(sum(row) / img.shape[0])

    kernel = np.ones(3) / 3
    row_sum = np.convolve(row_sum, kernel, mode='same')

    plt.plot(row_sum)
    plt.show()

if(len(sys.argv) < 4):
    print('too few arguments')
    sys.exit(0)

path_ir = sys.argv[1]
path_rgb = sys.argv[2]
path_bias = sys.argv[3]

img_ir = cv2.imread(path_ir, cv2.IMREAD_GRAYSCALE)
img_rgb = cv2.imread(path_rgb)
img_bias = cv2.imread(path_bias, cv2.IMREAD_GRAYSCALE)

#img_ir = overlay_bias_image_addition(img_bias, img_ir, 1.0)
img_ir = overlay_bias_image(img_bias, img_ir, 0.5)

#for i in range(0, 100, 10):
#    tmp_ir = cv2.adaptiveThreshold(img_ir, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, i)
#    tmp_rgb = cv2.adaptiveThreshold(img_rgb, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, i)
#    fig, axes = plt.subplots(1, 2)
#    axes[0].imshow(tmp_ir, 'gray')
#    axes[1].imshow(tmp_rgb, 'gray')
#    plt.show()

####################################
# pretty good mask for highlighter #
####################################

img_rgb_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

#lower = np.array([0, 0, 200])
#upper = np.array([255, 50, 255])

lower = np.array([0, 0, 200])
upper = np.array([255, 70, 255])

thresh = cv2.inRange(img_rgb_hsv, lower, upper)
thresh = cv2.bitwise_not(thresh)
result = cv2.bitwise_and(img_rgb, img_rgb, mask=thresh)
#result_tmp = result

#result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
lower = np.array([0, 200, 0])
upper = np.array([255, 255, 255])
thresh = cv2.inRange(result, lower, upper)

kernel = np.ones((3, 3), np.uint8)
#thresh = cv2.dilate(thresh, kernel)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#thresh = cv2.dilate(thresh, kernel)
result = cv2.bitwise_and(img_rgb, img_rgb, mask=thresh)

# background pixels are black now, therefore we make them white
result[np.where((result == [0, 0, 0]).all(axis=2))] = [255, 255, 255]

fig, axes = plt.subplots(1, 3)
axes[0].imshow(img_rgb)
axes[1].imshow(thresh, 'gray')
#axes[2].imshow(result_tmp)
axes[2].imshow(result)
plt.show()

##############################
# remove wrinkles from paper #
# (does not work very well)  #
##############################

img_ir_rgba = img_ir
img_ir_rgba = np.bitwise_not(img_ir_rgba)
img_ir_rgba = cv2.cvtColor(img_ir_rgba, cv2.COLOR_GRAY2RGBA)
img_rgb_rgba = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
img_rgb_rgba = np.bitwise_not(img_rgb_rgba)
img_rgb_rgba = cv2.cvtColor(img_rgb_rgba, cv2.COLOR_GRAY2RGBA)

img_ir_rgba = img_ir_rgba.astype(np.float32)
img_rgb_rgba = img_rgb_rgba.astype(np.float32)

#result = blend_modes.dodge(scan, bias_inverted, 1) # awesome! white background, high contrast
#result = blend_modes.addition(scan, bias_inverted, 1) # similar to dodge, but lower contrast
#result = blend_modes.difference(scan, bias_inverted, 1) # good when bias image is not inverted (but result is inverted)
img_dewrinkle = blend_modes.difference(img_ir_rgba, img_rgb_rgba, 1.0)

img_dewrinkle = img_dewrinkle.astype(np.uint8)
img_dewrinkle = cv2.cvtColor(img_dewrinkle, cv2.COLOR_RGBA2GRAY)

fig, axes = plt.subplots(1, 3)
axes[0].imshow(img_ir, 'gray')
axes[1].imshow(img_rgb)
axes[2].imshow(img_dewrinkle, 'gray')
plt.show()

###########################################################
# overlay IR scan and RGB scan to extract pen annotations #
###########################################################

img_rgb_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

thresh_ir = cv2.adaptiveThreshold(img_ir, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 30)
thresh_rgb = cv2.adaptiveThreshold(img_rgb_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 30)

fig, axes = plt.subplots(1, 3)
thresh_ir_invert = np.bitwise_not(thresh_ir)
img_ir_dewrinkle = np.bitwise_and(img_ir, thresh_ir)
axes[0].imshow(img_ir, 'gray')
axes[1].imshow(thresh_ir, 'gray')
axes[2].imshow(img_ir_dewrinkle, 'gray')
plt.show()

#show_row_sum(thresh_ir)
#show_row_sum(thresh_ir.transpose())

img_xor = np.bitwise_xor(thresh_ir, thresh_rgb)

kernel = np.ones((3, 3), np.uint8)
result = cv2.morphologyEx(img_xor, cv2.MORPH_OPEN, kernel)

fig, axes = plt.subplots(1, 4)
axes[0].imshow(thresh_ir, 'gray')
axes[1].imshow(thresh_rgb, 'gray')
axes[2].imshow(img_xor, 'gray')
axes[3].imshow(result, 'gray')
plt.show()
