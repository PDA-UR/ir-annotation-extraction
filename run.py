import sys
import shutil
from glob import glob
from extraction import extract_annotations
from insert_annotation import insert_annotation
from PyPDF2 import PdfFileReader

if(len(sys.argv) < 5):
    print('too few arguments!')
    print('usage: python3 run.py SCAN_DIRECTORY BIAS_IMAGE PDF OUTPUT')

# TODO: check if paths are valid
directory_path = sys.argv[1]
bias_path = sys.argv[2]
pdf_path = sys.argv[3]
out_path = sys.argv[4]

# create working copy for PDF
shutil.copy(pdf_path, out_path)

# get number of pages in PDF
pdf = PdfFileReader(open(pdf_path, 'rb'))
pdf_pages = pdf.getNumPages()

# get lists with available files
# suffixes:
# - extracted annotations: _annotation.png
# - IR scans: _IR.png
# - RGB scans: _RGB.png
annotation_paths = sorted(glob(f'{directory_path}*_annotation.png'))
ir_paths = sorted(glob(f'{directory_path}*_IR.png'))
rgb_paths = sorted(glob(f'{directory_path}*_RGB.png'))
paths_trunkated = sorted(list(map(lambda p : p.replace('_IR.png', ''), ir_paths)))

# iterate over PDF pages and check if there are matching IR scans
for p in range(1, pdf_pages + 1):
    matching = [s for s in paths_trunkated if f'_{p:02d}' in s]
    if len(matching) == 1:
        trunkated = matching[0]
        matching_ir = [s for s in ir_paths if trunkated in s]
        matching_rgb = [s for s in rgb_paths if trunkated in s]
        matching_annotation = [s for s in annotation_paths if trunkated in s]

        if(len(matching_ir) != 1):
            print('can not find IR image', matching_ir)
            continue

        if(len(matching_rgb) != 1):
            print('can not find RGB image', matching_rgb)
            continue

        if(len(matching_annotation) > 1):
            print('found too many matching annotation files', matching_rgb)
            continue
        elif(len(matching_annotation) == 0):
            print(f'extracting: {trunkated}')
            # TODO exception handling
            extraction_result_path = extract_annotations(matching_rgb[0], matching_ir[0], bias_path, f'{trunkated}_annotation.png')
            matching_annotation.append(extraction_result_path)

        print('inserting annotations')
        insert_annotation(matching_annotation[0], matching_ir[0], bias_path, out_path, p, out_path)
        print(f'{out_path} page {p} written')
    elif len(matching) == 0:
        print(f'no annotations found for page {p}')
        continue 
    else:
        # manual selection I guess?
        pass
