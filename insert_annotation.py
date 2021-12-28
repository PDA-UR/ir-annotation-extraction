from pdf_annotate import PdfAnnotator, Location, Appearance
import sys
import cv2
import numpy as np
from bbox import get_text_bb
from pdf2image import convert_from_path, pdfinfo_from_path
from matplotlib import pyplot as plt
import os
from util import overlay_bias_image, normalize_image

# TODO split up into separate functions
def insert_annotation(annotation_path, ir_scan_path, bias_image_path, pdf_path, pdf_page, out_path):
    temp_image_path = 'temp.png'

    # TODO maybe scale down images for faster processing?
    ir_scan = cv2.imread(ir_scan_path) # todo grayscale
    ir_scan = ir_scan[:,:,2]
    annot = cv2.imread(annotation_path, cv2.IMREAD_UNCHANGED)

    pdf_pages = convert_from_path(pdf_path, 300, first_page=pdf_page, last_page=pdf_page, grayscale=True, size=(ir_scan.shape[1], None))

    # use bias image to normalize brightness distribution over the image
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

    ## insert annotation into PDF

    # required as PdfAnnotator takes string or ImageFile as argument
    # TODO: maybe modify PdfAnnotator?
    cv2.imwrite(temp_image_path, result)

    pdf_info = pdfinfo_from_path(pdf_path)
    pdf_aspect_text = pdf_info['Page size']
    pdf_aspect = (float(pdf_aspect_text.split(' ')[0]), float(pdf_aspect_text.split(' ')[2]))

    # source: https://github.com/plangrid/pdf-annotate
    # scales are in dots per inch (default: 72)
    # example US letter: 11" x 72 dpi = 792 dots
    # values:
    # 612 x 792 (US letter)
    # 595.276 x 841.89 (A4)
    pdf_size = dict()
    pdf_size['A4'] = (595.276, 841.89)
    pdf_size['A5'] = (841.89 / 2, 595.276)
    pdf_size['letter'] = (612, 792)
    pdf_size['auto'] = pdf_aspect

    pdf_format = 'auto'

    annotator = PdfAnnotator(pdf_path)
    annotator.add_annotation(
        'image',
        Location(x1=0, y1=0, x2=pdf_size[pdf_format][0], y2=pdf_size[pdf_format][1], page=pdf_page-1),
        Appearance(stroke_color=(1, 0, 0), stroke_width=5, image=temp_image_path),
    )
    annotator.write(out_path)

    os.remove(temp_image_path)

#path = sys.argv[1]
#pdf_path = path.split('_')[0] + '.pdf'
#pdf_page = int(path.split('_')[-1])
#annotation_path = path + '_annotation.png'
#ir_scan_path = path + '_IR.png'
#temp_image_path = 'temp.png'
##out_path = path.split('_')[0] + '_' + path.split('_')[1] + '_out.pdf'
#out_path = pdf_path

if __name__ == '__main__':
    #if(len(sys.argv) < 2):
    #    print('too few arguments')
    #    print('usage: python3 insert_annotation.py path')
    #    sys.exit(0)

    # TODO: check if paths are valid
    annotation_path = sys.argv[1]
    ir_scan_path = sys.argv[2]
    bias_image_path = sys.argv[3]
    pdf_path = sys.argv[4]
    pdf_page = int(sys.argv[5])
    out_path = sys.argv[6]

    insert_annotation(annotation_path, ir_scan_path, bias_image_path, pdf_path, pdf_page, out_path)
