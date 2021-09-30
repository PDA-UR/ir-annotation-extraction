import cv2
from matplotlib import pyplot as plt
import sys
import numpy as np

def show_row_sum(img):
    row_sum = []

    for row in img:
        row_sum.append(sum(row) / img.shape[0])

    kernel = np.ones(3) / 3
    row_sum = np.convolve(row_sum, kernel, mode='same')

    plt.plot(row_sum)
    plt.show()

if(len(sys.argv) < 3):
    print('too few arguments')
    sys.exit(0)

path_ir = sys.argv[1]
path_rgb = sys.argv[2]

img_ir = cv2.imread(path_ir, cv2.IMREAD_GRAYSCALE)
img_rgb = cv2.imread(path_rgb, cv2.IMREAD_GRAYSCALE)

img_ir = cv2.adaptiveThreshold(img_ir, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 30)
img_rgb = cv2.adaptiveThreshold(img_rgb, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 30)

show_row_sum(img_ir)
show_row_sum(img_ir.transpose())

plt.imshow(img_ir, 'gray')
plt.show()
plt.imshow(img_rgb, 'gray')
plt.show()

result = np.bitwise_and(img_ir, img_rgb)

plt.imshow(result, 'gray')
plt.show()

