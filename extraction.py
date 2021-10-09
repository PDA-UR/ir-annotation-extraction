import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import sys

#source: https://stackoverflow.com/questions/55673060/how-to-set-white-pixels-to-transparent-using-opencv2 --> answer by Qwertford
def save_image(img, filename):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    # get the image dimensions (height, width and channels)
    h, w, c = image_rgb.shape
    # append Alpha channel -- required for BGRA (Blue, Green, Red, Alpha)
    image_rgba = np.concatenate([image_rgb, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
    # create a mask where white pixels ([255, 255, 255]) are True
    white = np.all(image_rgb == [255, 255, 255], axis=-1)
    # change the values of Alpha to 0 for all the white pixels
    image_rgba[white, -1] = 0
    # save the image
    cv2.imwrite(filename, image_rgba)

# get rid of background in RGB image by converting image to HSV
# and masking away areas with low saturation and high value (e.g. paper)
def remove_rgb_background(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # low saturation, high value areas are the background
    lower = np.array([0, 0, 200])
    upper = np.array([255, 50, 255])

    thresh = cv2.inRange(img_hsv, lower, upper)
    thresh = cv2.bitwise_not(thresh)
    result = cv2.bitwise_and(img, img, mask=thresh)

    # background pixels are black now, therefore we make them white
    result[np.where((result == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    return result

def overlay_bias_image(bias_image, scan, alpha=1.0):
    bias_inverted = cv2.bitwise_not(bias_image)
    result = cv2.addWeighted(scan, alpha, bias_inverted, 1.0 - alpha, 0.0)
    return result

def main(filename):
    DEBUG = False

    rgb_path = filename + '_RGB.png'
    ir_path = filename + '_IR.png'
    bias_path = 'bias.png'

    img_rgb = cv2.imread(rgb_path)
    img_IR = cv2.imread(ir_path)
    img_IR = img_IR[:,:,2]
    img_bias = cv2.imread(bias_path, cv2.IMREAD_GRAYSCALE) 

    #img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

    if(DEBUG):
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(img_rgb)
        axes[1].imshow(img_IR, 'gray')
        plt.show()

    img_IR_clean = overlay_bias_image(img_bias, img_IR, 0.5)

    _, img_IR_thresh = cv2.threshold(img_IR_clean, 125, 255, 0)

    inverted = cv2.bitwise_not(img_IR_thresh)
    kernel = np.ones((3, 3), np.uint8)
    img_IR_dilate = cv2.dilate(inverted, kernel, iterations=2)

    if(DEBUG):
        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(img_IR_clean, 'gray')
        axes[1].imshow(img_IR_thresh, 'gray')
        axes[2].imshow(img_IR_dilate, 'gray')
        plt.show()

    img_rgb_clean = remove_rgb_background(img_rgb)

    img_IR_dilate = cv2.cvtColor(img_IR_dilate, cv2.COLOR_GRAY2BGR)
    img_annotations = cv2.add(img_rgb_clean, img_IR_dilate)

    #_, annotation_thresh = cv2.threshold(img_annotations, 200, 255, cv2.THRESH_BINARY)
    #annotation_thresh = cv2.bitwise_not(annotation_thresh)
    #annotation_thresh = cv2.cvtColor(annotation_thresh, cv2.COLOR_BGR2GRAY)
    #annotations = cv2.bitwise_and(img_rgb, img_rgb, mask=annotation_thresh)

    if(DEBUG):
        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(img_rgb)
        axes[1].imshow(img_rgb_clean)
        axes[2].imshow(img_annotations)
        plt.show()

    save_image(img_annotations, filename + '_annotation.png')

if __name__ == "__main__":
    main(sys.argv[1])
