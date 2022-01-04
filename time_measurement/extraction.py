import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
from util import overlay_bias_image
from timeit import default_timer as timer

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
def remove_rgb_background(img, sat_thresh=50, val_thresh=200):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # low saturation, high value areas are the background
    lower = np.array([0, 0, val_thresh])
    upper = np.array([255, sat_thresh, 255])

    thresh = cv2.inRange(img_hsv, lower, upper)
    thresh = cv2.bitwise_not(thresh)
    result = cv2.bitwise_and(img, img, mask=thresh)

    # background pixels are black now, therefore we make them white
    result[np.where((result == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    return result

def get_rgb_colormask(img, sat_thresh=50, val_thresh=200):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # high saturation, not too dark or bright value areas are annotations
    lower = np.array([0, sat_thresh, val_thresh])
    upper = np.array([255, 255, 255 - val_thresh])

    thresh = cv2.inRange(img_hsv, lower, upper)
    thresh = cv2.bitwise_not(thresh)
    result = cv2.bitwise_and(img, img, mask=thresh)

    result[np.where((result != [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    return result


# removes red/blue fringing due to not exactly aligned color channels because of scanner movement
def remove_fringing(img):
    # work on image copies as np operations overwrite the original image
    img_blue = img.copy()
    img_red = img.copy()

    # get blue and red color channels (those are the farthest away)
    blue_channel = img_blue[:,:,0]
    red_channel = img_red[:,:,2]

    # move channels in opposite directions
    blue_channel = np.roll(blue_channel, 1, axis=0)
    red_channel = np.roll(red_channel, -1, axis=0)

    # apply changes to color channels
    img_blue[:,:,0] = blue_channel
    img_red[:,:,2] = red_channel

    # the results now have green/purple fringing in opposite directions
    # therefore, blend them together for a result with less color fringing
    img_result = cv2.addWeighted(img_blue, 0.5, img_red, 0.5, 0.0)

    return img_result

# removes black fringing at the top and bottom of text
def grow_printed_text(img):
    ## might be more performant, but did not work as well
    #kernel = np.asarray([[  0,    0, 1,    0,   0], 
    #                     [0, 1,   1, 1, 0], 
    #                     [0, 1,   1, 1, 0], 
    #                     [0, 1,   1, 1, 0], 
    #                     [  0,    0, 1,    0,   0]], np.uint8)
    #result = cv2.dilate(img, kernel, iterations=3)

    # dilate once to grow text
    kernel = np.ones((3, 3), np.uint8)
    result = cv2.dilate(img, kernel, iterations=1)

    # create copies and shift them up/down
    result_shift_1 = result.copy()
    result_shift_1 = np.roll(result_shift_1, 1, axis=0)
    result_shift_2 = result.copy()
    result_shift_2 = np.roll(result_shift_2, -1, axis=0)
    result_shift_3 = result.copy()
    result_shift_3 = np.roll(result_shift_3, 2, axis=0)
    result_shift_4 = result.copy()
    result_shift_4 = np.roll(result_shift_4, -2, axis=0)

    # add shifted copies to original
    result = cv2.add(result, result_shift_1)
    result = cv2.add(result, result_shift_2)
    result = cv2.add(result, result_shift_3)
    result = cv2.add(result, result_shift_4)

    return result

def crop_image(img, margin):
    w = img.shape[1]
    h = img.shape[0]
    img = img[margin : h - margin, margin : w - margin]
    return img

def extract_annotations(rgb_path, ir_path, bias_path, out_path):
    DEBUG = False

    start = timer()
    img_rgb = cv2.imread(rgb_path)
    img_IR = cv2.imread(ir_path)
    img_IR = img_IR[:,:,2]
    img_bias = cv2.imread(bias_path, cv2.IMREAD_GRAYSCALE) 
    end = timer()
    print(f'{ir_path},extract,1,read_images,{end-start}')

    ## crop images
    #crop_margin = 20
    #img_rgb = crop_image(img_rgb, crop_margin)
    #img_IR = crop_image(img_IR, crop_margin)
    #img_bias = crop_image(img_bias, crop_margin)

    #img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

    #img_rgb_og = img_rgb.copy()
    #img_defringe_blue, img_defringe_red = remove_fringing(img_rgb)
    #img_rgb_blend = cv2.addWeighted(img_rgb_og, 0.6, img_rgb_defringe, 0.4, 0.0)
    #img_rgb_blend = cv2.addWeighted(img_defringe_blue, 0.5, img_defringe_red, 0.5, 0.0)

    if(DEBUG):
        fig, axes = plt.subplots(1, 3)
        axes[0].set_title('RGB image')
        axes[1].set_title('IR image')
        axes[2].set_title('bias image')
        axes[0].imshow(img_rgb)
        axes[1].imshow(img_IR, 'gray')
        axes[2].imshow(img_bias, 'gray')
        plt.show()

    start = timer()
    img_IR_clean = overlay_bias_image(img_bias, img_IR, 0.5)
    end = timer()
    print(f'{ir_path},extract,2,overlay_bias_image,{end-start}')

    #_, img_IR_thresh = cv2.threshold(img_IR_clean, 125, 255, 0)
    _, img_IR_thresh = cv2.threshold(img_IR_clean, 110, 255, 0)

    inverted = cv2.bitwise_not(img_IR_thresh)

    start = timer()
    img_IR_dilate = grow_printed_text(inverted)
    end = timer()
    print(f'{ir_path},extract,3,grow_printed_text,{end-start}')

    if(DEBUG):
        fig, axes = plt.subplots(1, 3)
        axes[0].set_title('IR clean')
        axes[1].set_title('IR thresh')
        axes[2].set_title('IR dilate')
        axes[0].imshow(img_IR_clean, 'gray')
        axes[1].imshow(img_IR_thresh, 'gray')
        axes[2].imshow(img_IR_dilate, 'gray')
        plt.show()

    # remove color fringing from RGB image
    start = timer()
    img_rgb_defringe = remove_fringing(img_rgb)
    end = timer()
    print(f'{ir_path},extract,4,defringe,{end-start}')

    if(DEBUG):
        fig, axes = plt.subplots(1, 2)
        axes[0].set_title('RGB image')
        axes[1].set_title('defringe')
        axes[0].imshow(img_rgb)
        axes[1].imshow(img_rgb_defringe)
        plt.show()

    start = timer()
    img_rgb_clean = remove_rgb_background(img_rgb_defringe, 20, 200)
    end = timer()
    print(f'{ir_path},extract,5,remove_rgb_background,{end-start}')

    if(DEBUG):
        fig, axes = plt.subplots(1, 3)
        axes[0].set_title('sat 80')
        axes[1].set_title('sat 100')
        axes[2].set_title('sat 120')
        axes[0].imshow(get_rgb_colormask(img_rgb_clean, 80, 10), 'gray')
        axes[1].imshow(get_rgb_colormask(img_rgb_clean, 100, 10), 'gray')
        axes[2].imshow(get_rgb_colormask(img_rgb_clean, 120, 10), 'gray')
        plt.show()

    """
    if(DEBUG):
        fig, axes = plt.subplots(1, 3)
        axes[0].set_title('val 0')
        axes[1].set_title('val 10')
        axes[2].set_title('val 20')
        axes[0].imshow(get_rgb_colormask(img_rgb_clean, 120, 0), 'gray')
        axes[1].imshow(get_rgb_colormask(img_rgb_clean, 120, 10), 'gray')
        axes[2].imshow(get_rgb_colormask(img_rgb_clean, 120, 20), 'gray')
        plt.show()
    """

    start = timer()
    img_rgb_mask = get_rgb_colormask(img_rgb_clean, 100, 10)
    img_rgb_mask = cv2.bitwise_not(img_rgb_mask)
    img_mask_result = cv2.subtract(img_IR_dilate, img_rgb_mask)
    end = timer()
    print(f'{ir_path},extract,6,prepare_rgb_image,{end-start}')

    #img_IR_dilate = cv2.cvtColor(img_IR_dilate, cv2.COLOR_GRAY2BGR)
    img_mask_result = cv2.cvtColor(img_mask_result, cv2.COLOR_GRAY2BGR)

    if(DEBUG):
        fig, axes = plt.subplots(1, 4)
        axes[0].set_title('RGB clean')
        axes[1].set_title('IR dilate')
        axes[2].set_title('rgb mask')
        axes[3].set_title('result mask')
        axes[0].imshow(img_rgb_clean)
        axes[1].imshow(img_IR_dilate, 'gray')
        axes[2].imshow(img_rgb_mask, 'gray')
        axes[3].imshow(img_mask_result, 'gray')
        plt.show()

    #img_annotations = cv2.add(img_rgb_clean, img_IR_dilate)
    img_annotations = cv2.add(img_rgb_clean, img_mask_result)

    #_, annotation_thresh = cv2.threshold(img_annotations, 200, 255, cv2.THRESH_BINARY)
    #annotation_thresh = cv2.bitwise_not(annotation_thresh)
    #annotation_thresh = cv2.cvtColor(annotation_thresh, cv2.COLOR_BGR2GRAY)
    #annotations = cv2.bitwise_and(img_rgb, img_rgb, mask=annotation_thresh)

    if(DEBUG):
        fig, axes = plt.subplots(1, 3)
        axes[0].set_title('RGB image')
        axes[1].set_title('cleaned RGB')
        axes[2].set_title('annotations')
        axes[0].imshow(img_rgb)
        axes[1].imshow(img_rgb_clean)
        axes[2].imshow(img_annotations)
        plt.show()

    if not DEBUG:
        start = timer()
        save_image(img_annotations, out_path)
        end = timer()
        print(f'{ir_path},extract,7,save_result,{end-start}')
    return out_path

if __name__ == "__main__":
    if len(sys.argv) > 4:
        # TODO exception handling
        rgb_path = sys.argv[1]
        ir_path = sys.argv[2]
        bias_path = sys.argv[3]
        out_path = sys.argv[4]
        extract_annotations(rgb_path, ir_path, bias_path, out_path)
    else:
        print('Too few arguments.')
        print('Usage: python3 extraction.py RGB_IMAGE IR_IMAGE BIAS_IMAGE OUTPUT')
