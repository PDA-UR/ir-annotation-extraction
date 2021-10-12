import numpy as np
import cv2
import blend_modes

# adjust black and white point to use whole dynamic range
def normalize_image(img):
    minimum = np.amin(img)
    maximum = np.amax(img)
    img = img.astype(np.float32)
    img -= minimum
    img /= maximum - minimum
    img *= 255
    img = img.astype(np.uint8)
    return img

# in case the brightness distribution of the IR scan is uneven,
# it is recommended to record an empty bias frame and use it
# to normalize the brightness distribution over the image
def overlay_bias_image(bias_image, scan, alpha=0.5):
    bias_inverted = cv2.bitwise_not(bias_image)
    result = cv2.addWeighted(scan, alpha, bias_inverted, 1.0 - alpha, 0.0)
    return result

def overlay_bias_image_addition(bias_image, scan, alpha=1.0):
    bias_inverted = cv2.bitwise_not(bias_image)

    bias_inverted = cv2.cvtColor(bias_inverted, cv2.COLOR_GRAY2RGBA)
    scan = cv2.cvtColor(scan, cv2.COLOR_GRAY2RGBA)

    bias_inverted = bias_inverted.astype(np.float32)
    scan = scan.astype(np.float32)

    #result = blend_modes.dodge(scan, bias_inverted, 1) # awesome! white background, high contrast
    #result = blend_modes.addition(scan, bias_inverted, 1) # similar to dodge, but lower contrast
    #result = blend_modes.difference(scan, bias_inverted, 1) # good when bias image is not inverted (but result is inverted)
    result = blend_modes.addition(scan, bias_inverted, alpha)

    result = result.astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_RGBA2GRAY)
    return result
