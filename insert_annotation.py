from pdf_annotate import PdfAnnotator, Location, Appearance
import sys
import cv2
import numpy as np
from bbox import get_text_bb
from pdf2image import convert_from_path
from matplotlib import pyplot as plt
import os

# in case the brightness distribution of the IR scan is uneven,
# it is recommended to record an empty bias frame and use it
# to normalize the brightness distribution over the image
def overlay_bias_image(bias_image, scan, alpha=1.0):
    bias_inverted = cv2.bitwise_not(bias_image)
    result = cv2.addWeighted(scan, alpha, bias_inverted, 1.0 - alpha, 0.0)
    return result

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

if(len(sys.argv) < 2):
    print('too few arguments')
    print('usage: python3 insert_annotation.py path')
    sys.exit(0)

path = sys.argv[1]
pdf_path = path.split('_')[0] + '.pdf'
pdf_page = int(path.split('_')[-1])
annotation_path = path + '_annotation.png'
ir_scan_path = path + '_IR.png'
temp_image_path = 'temp.png'
out_path = 'output.pdf'

# TODO maybe scale down images for faster processing?
ir_scan = cv2.imread(ir_scan_path) # todo grayscale
ir_scan = ir_scan[:,:,2]
annot = cv2.imread(annotation_path, cv2.IMREAD_UNCHANGED)
pdf_pages = convert_from_path(pdf_path, 300, first_page=pdf_page, last_page=pdf_page, grayscale=True, size=(ir_scan.shape[1], None))

# if a bias image is given, use it to normalize brightness distribution over the image
if(len(sys.argv) > 3):
    bias_image_path = sys.argv[2]
    bias_image = cv2.imread(bias_image_path)
    ir_scan = overlay_bias_image(bias_image, ir_scan, 0.5)
    ir_scan = normalize_image(ir_scan)

# have to be the same size
#print(ir_scan.shape)
#print(annot.shape)

pdf_image = np.array(pdf_pages[0])

# calculate bounding boxes
bb_ir_scan = get_text_bb(ir_scan)
bb_pdf_image = get_text_bb(pdf_image)

# use corners of bounding boxes to calculate homography
# TODO: move to function and/or external file
pts_ir = np.array([ [bb_ir_scan[0],                 bb_ir_scan[1]],
                    [bb_ir_scan[0] + bb_ir_scan[2], bb_ir_scan[1]],
                    [bb_ir_scan[0] + bb_ir_scan[2], bb_ir_scan[1] + bb_ir_scan[3]],
                    [bb_ir_scan[0],                 bb_ir_scan[1] + bb_ir_scan[3]]])

pts_pdf = np.array([[bb_pdf_image[0],                   bb_pdf_image[1]],
                    [bb_pdf_image[0] + bb_pdf_image[2], bb_pdf_image[1]],
                    [bb_pdf_image[0] + bb_pdf_image[2], bb_pdf_image[1] + bb_pdf_image[3]],
                    [bb_pdf_image[0],                   bb_pdf_image[1] + bb_pdf_image[3]]])

homography, _ = cv2.findHomography(pts_ir, pts_pdf)

# not required as annotation and ir_scan are the same size
#annot = cv2.resize(annot, (src1.shape[1], src1.shape[0]))

result = cv2.warpPerspective(annot, homography, (pdf_image.shape[1], pdf_image.shape[0]))

# required as PdfAnnotator takes string or ImageFile as argument
# TODO: maybe modify PdfAnnotator?
cv2.imwrite(temp_image_path, result)

# source: https://github.com/plangrid/pdf-annotate
# pdf default scale: 612x792
annotator = PdfAnnotator(pdf_path)
annotator.add_annotation(
    'image',
    Location(x1=0, y1=0, x2=612, y2=792, page=pdf_page-1),
    Appearance(stroke_color=(1, 0, 0), stroke_width=5, image=temp_image_path),
)
annotator.write(out_path)

os.remove(temp_image_path)
