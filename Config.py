

ZOOM_FACTOR = 0.25
WEAK_TEXT = 0
MEDIUM_TEXT = 1
STRONG_TEXT = 2
FINE_MODE = 3
FINEST_MODE = 4

SCAN_TYPE_NUM_LIST = [WEAK_TEXT, MEDIUM_TEXT, STRONG_TEXT, FINE_MODE, FINEST_MODE]

DEFAULT_MODE = FINEST_MODE

HIGHLIGHT_OPTION_STR = '-h'

DEFAULT_MODE_MESSAGE = "Uses default scan Type Number 2 = strong text (Alternatively pass arguments 0, 1, 2, 3, 4 or 5 for different sensitivity in contrast and threshhold values"
HIGHLIGHT_MODE_MESSAGE = "Use additional highlighting for yellow text markers"

IR_SCAN_EMPTY_NAME = 'IR_emptyScan.png'
IR_SCAN_TEXT_NAME = 'IR_textScan.png'
RGB_SCAN_TEXT_NAME = 'RGB_textScan.png'

EXTRACTED_ANN_DEFAULT_NAME = 'annotations.png'
EXTRACTED_ANN_WITH_ALPHA_CHANNEL ='annotations_transparent_bg.png'
EXTRACTED_ORIGINAL_TEXT_NAME =  'backround_text.png'

IR_DILATION_ITERATIONS = 3

### constants for whitepoint and blackpoint in rgb-scan
THRESH_WHITE_RGBA = 195
THRESH_BLACK_RGBA = 45

THRESH_WHITE_RGBA_SOFT =210
THRESH_BLACK_RGBA_SOFT = 20
THRESHOLD_BINARY_IR_DEFAULT =127
###

#constants for document with weak / small text
IR_ALPHA_W= 0.99
THRESH_WHITE_IR_W= 210
THRESH_BLACK_IR_W = 45
 
THRESHOLD_BINARY_IR_WEAK = 147

### Constants for Input with medium textstrength
IR_ALPHA_M = 0.85
THRESH_WHITE_IR_M = 185
THRESH_BLACK_IR_M = 97

THRESHOLD_BINARY_IR_MEDIUM  = 100
THRESHOLD_BINARY_IR_MEDIUMSTRONG = 115

### Constants for Input with strong text
IR_ALPHA_S = 0.60
THRESH_WHITE_IR_S = 160
THRESH_BLACK_IR_S = 80

THRESHOLD_BINARY_IR_STRONG = 81



def getEffectFilterValuesByScanType(scanType = DEFAULT_MODE):
     if (scanType == WEAK_TEXT): 
         return (IR_ALPHA_W, THRESH_BLACK_IR_W, THRESH_WHITE_IR_W, THRESH_BLACK_RGBA, THRESH_WHITE_RGBA, THRESHOLD_BINARY_IR_WEAK) #highest threshold option ->weak extraction
     elif (scanType == MEDIUM_TEXT):
         return (IR_ALPHA_M, THRESH_BLACK_IR_M, THRESH_WHITE_IR_M, THRESH_BLACK_RGBA, THRESH_WHITE_RGBA, THRESHOLD_BINARY_IR_MEDIUMSTRONG) #mediumstrong binary thrsh, hard IR contrast option
     elif (scanType == STRONG_TEXT):
         return (IR_ALPHA_S, THRESH_BLACK_IR_S, THRESH_WHITE_IR_S, THRESH_BLACK_RGBA, THRESH_WHITE_RGBA, THRESHOLD_BINARY_IR_DEFAULT) #mid thrsh, (strong) text strength optio
     elif (scanType == FINE_MODE):
         return(IR_ALPHA_S, THRESH_BLACK_IR_S, THRESH_WHITE_IR_S, THRESH_BLACK_RGBA_SOFT, THRESH_WHITE_RGBA, THRESHOLD_BINARY_IR_MEDIUMSTRONG) #Fine Mode: mediumstrong binary thresh
     elif (scanType == FINEST_MODE):
         return(IR_ALPHA_S, THRESH_BLACK_IR_S, THRESH_WHITE_IR_S, THRESH_BLACK_RGBA_SOFT, THRESH_WHITE_RGBA, THRESHOLD_BINARY_IR_MEDIUM) #Finest Mode: lowest mid thresh ,NO ADDED CONTOURS 
     else:
         return (0,0,0,0,0,0)
         
