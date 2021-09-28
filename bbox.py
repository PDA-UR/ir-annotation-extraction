# partially based on https://datascience.stackexchange.com/questions/47302/how-can-i-detect-blocks-of-text-from-scanned-document-images

import cv2
import sys
import numpy as np
from matplotlib import pyplot as plt

def get_text_bb(img):
    # convert to grayscale if it is not already
    if(len(img.shape) > 2):
        img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_grayscale = img

    # TODO adjust all hard coded values to image size dynamically

    # low pass filter to get rid of noise and small spots
    blur = cv2.GaussianBlur(img_grayscale, (9,9), 0)
    plt.imshow(blur, 'gray')
    plt.show()

    # dilate to get rid of small spots
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilation = cv2.dilate(blur, kernel, iterations=1)
    plt.imshow(dilation, 'gray')
    plt.show()

    # huge erosion kernel to emphasize text areas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    erosion = cv2.erode(dilation, kernel, iterations=5)
    plt.imshow(erosion, 'gray')
    plt.show()

    # threshold for contour detection
    # adaptive because image might have a brightness gradient
    thresh = cv2.adaptiveThreshold(erosion, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    plt.imshow(thresh, 'gray')
    plt.show()

    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    x_values = []
    y_values = []

    # minimum size of contours
    min_w = img.shape[1] / 100
    min_h = img.shape[0] / 200
    #print(min_w, min_h)

    # find bounding boxes around contours and remember minimum and maximum coordinates
    # the final bounding box spans all detected contours (and thus the whole text)
    for c in contours:
        rect = cv2.boundingRect(c)

        # get rid of small contours to decrease false positives
        # TODO make dependent on image size
        if rect[2] < min_w or rect[3] < min_h:
            continue

        #print (cv2.contourArea(c))
        x, y, w, h = rect
        x_values.append(x)
        x_values.append(x + w)
        y_values.append(y)
        y_values.append(y + h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # final bounding box
    x = min(x_values)
    y = min(y_values)
    w = max(x_values) - x
    h = max(y_values) - y

    plt.imshow(img)
    plt.show()
    return (x, y, w, h)


if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print('too few arguments')
        sys.exit(0)

    img_path = sys.argv[1]

    img = cv2.imread(img_path)

    cv2.rectangle(img, get_text_bb(img), (255, 0, 0), 2)

    plt.imshow(img)
    plt.show()
