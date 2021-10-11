import cv2

# in case the brightness distribution of the IR scan is uneven,
# it is recommended to record an empty bias frame and use it
# to normalize the brightness distribution over the image
def overlay_bias_image(bias_image, scan, alpha=0.5):
    bias_inverted = cv2.bitwise_not(bias_image)
    result = cv2.addWeighted(scan, alpha, bias_inverted, 1.0 - alpha, 0.0)
    return result
