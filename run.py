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
#paths_trunkated = list(map(lambda p : p.replace('_annotation.png', ''), annotation_paths))
#paths_trunkated += list(map(lambda p : p.replace('_IR.png', ''), ir_paths))
#paths_trunkated = sorted(list(set(paths_trunkated)))
#print(paths_trunkated)

# iterate over PDF pages and check if there are matching IR scans
for p in range(1, pdf_pages + 1):
    annotations = [s for s in annotation_paths if f'_{p:02d}' in s]
    ir = [s for s in ir_paths if f'_{p:02d}' in s]

    #base_path = f'{directory_path}_{p:02d}'
    #print(base_path)
    #break

    # first, check if there is already an annotation file
    if len(annotations) > 1:
        print('multiple matching annotation files found, skipping...')
        # skip for now, TODO maybe implement manual selection later?
        continue
    elif len(annotations) == 0:
        # we don't have annotation files -> try to extract them from IR scans

        if(len(ir) == 0):
            print('no annotation files or IR scans found, skipping...')
            continue
        elif(len(ir) > 1):
            print('multiple matching IR scans found, skipping...')
            # skip for now, TODO maybe implement manual selection later?
            continue
        else:
            rgb = [s for s in rgb_paths if f'_{p:02d}' in s]
            if(len(rgb) != 1):
                print('RGB scans not found or ambiguous, skipping...')
                continue

            print(f'extracting annotations for page {p}')
            extraction_result_path = extract_annotations(rgb[0], ir[0], bias_path, ir[0].replace('_IR.png', '_annotation.png'))
            print(f'extracted annotations written to {extraction_result_path}')
            annotations.append(extraction_result_path)

    if(len(ir) == 1):
        print('inserting annotations')
        insert_annotation(annotations[0], ir[0], bias_path, out_path, p, out_path)
        print(f'{out_path} page {p} written')
    else:
        print('IR scans not found or ambiguous, skipping...')
        continue
