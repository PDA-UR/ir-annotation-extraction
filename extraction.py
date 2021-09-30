#!/usr/bin/env python

### General sources: https://pypi.org/project/blend-modes/
#used thresholding methods : https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
#used erdoding dilating methods : https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
#used logical operations for Inversion and Image Addition : https://docs.opencv.org/master/d0/d86/tutorial_py_image_arithmetics.html
###

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from blend_modes import overlay
from PIL import Image
import Config as cfg
import os
import sys

alpha_IROverlay = None
blackpointThresh_IR = None
whitepointThresh_IR = None
blackpointThresh_RGB = None
whitepointThresh_RGB = None

thresholdBinary_IR = None

userInputDocType = None

def overlayIRScans(alpha, emptyScan, textScan):
    beta = (1.0 - alpha)
    emptyInverted = cv.bitwise_not(emptyScan)
    imgIROverlayed = cv.addWeighted(textScan, alpha, emptyInverted, beta, 0.0)
    #showImage(imgIROverlayed, 'OVERLAY') 
    return imgIROverlayed

def adjustHistogramValues(scannedImg, isIRscan):
    imgFin = None
    if (isIRscan == True):
        ret, imgFin = cv.threshold(scannedImg, whitepointThresh_IR, 255, cv.THRESH_TRUNC)
        ret, imgFin = cv.threshold(imgFin, blackpointThresh_IR, 255, cv.THRESH_TOZERO)
        ret, imgFin = cv.threshold(imgFin, thresholdBinary_IR, 255, cv.THRESH_BINARY)
    elif(isIRscan == False):
        # Set maximum Values of original Image (RGB-LED-Scan) --> whitepoint
        ret, imgFin = cv.threshold(scannedImg, whitepointThresh_RGB, 255, cv.THRESH_TRUNC)
        imgFin[np.where((imgFin == [whitepointThresh_RGB, whitepointThresh_RGB, whitepointThresh_RGB]).all(axis = 2))] = [255,255,255] #source: https://answers.opencv.org/question/97416/replace-a-range-of-colors-with-a-specific-color-in-python/ -> answered by User: Missing
        # Set minimum Values of original Image (RGB-LED-Scan) --> blackpoint:
        ret, imgFin = cv.threshold(imgFin, blackpointThresh_RGB, 255, cv.THRESH_TOZERO)
    return imgFin


# IR-Scan Invertieren ,Erosion ,Dilatation
def prepareIRScan(irScanImg, iterationNum = cfg.IR_DILATION_ITERATIONS):
    #showImage(irScanImg, 'before Inversion')
    invertedImg = cv.bitwise_not(irScanImg)
    kernel = np.ones((5,5),np.uint8) #to do change kernel size 
    if (userInputDocType == cfg.WEAK_TEXT):
        iterationNum = 2
        kernel = np.ones((50,50),np.uint8)
    #print(iterationNum)
    dilatedImg = cv.dilate(invertedImg,kernel,iterations = iterationNum)
    return dilatedImg

def showImage(img, imageName ='Default'):
    resizedImg = cv.resize(img,None,fx=cfg.ZOOM_FACTOR,fy=cfg.ZOOM_FACTOR)
    cv.imshow(imageName, resizedImg)
    cv.waitKey(0)

def saveImage(img, fileName):
    #imgFilePath = str(os.getcwd()) + "/" + str(fileName)
    cv.imwrite(fileName, img)

# Binary Threshold and Contours generation, save extracted Background Text 
def addContours(imgWithAnnotations):
    imgray = cv.cvtColor(imgWithAnnotations, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, thresholdBinary_IR, 255, 0)
    saveImage(thresh, cfg.EXTRACTED_ORIGINAL_TEXT_NAME)
    if (userInputDocType != cfg.FINEST_MODE):
        contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        #print (str(len(contours)) + " CONTOURS NUM")
        cv.drawContours(thresh, contours, -1, (0,255,0), 3)
    #showImage(thresh, 'contours')
    return thresh

#improve visibility of yellow and orange Markers in final Image (optional) 
# source: https://stackoverflow.com/questions/57262974/tracking-yellow-color-object-with-opencv-python --> answer by nathancy
def addMarkerHighlights(origImg, extractedImg): 
    image = cv.cvtColor(origImg, cv.COLOR_BGR2HSV)
    lower = np.array([10, 13, 225], dtype="uint8")
    upper = np.array([60, 255, 255], dtype="uint8")
    mask = cv.inRange(image, lower, upper)
    kernel = np.ones((1,1),np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    
    cnts = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    highlightedImg = extractedImg.copy()
    for c in cnts:
        x,y,w,h = cv.boundingRect(c)
        cv.rectangle(highlightedImg, (x, y), (x + w, y + h), (0,200,255), 0)   
    #showImage(mask,'mask')
    #showImage(highlightedImg,'original highlighted')
    return highlightedImg

def saveTransparentImage(img, filename):
#source: https://stackoverflow.com/questions/55673060/how-to-set-white-pixels-to-transparent-using-opencv --> answer by Qwertford
    # read the image
    image_bgr = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
    # get the image dimensions (height, width and channels)
    h, w, c = image_bgr.shape
    # append Alpha channel -- required for BGRA (Blue, Green, Red, Alpha)
    image_bgra = np.concatenate([image_bgr, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
    # create a mask where white pixels ([255, 255, 255]) are True
    white = np.all(image_bgr == [255, 255, 255], axis=-1)
    # change the values of Alpha to 0 for all the white pixels
    image_bgra[white, -1] = 0
    # save the image
    saveImage(image_bgra, filename)
        

def main(scanTypeNum, filename, highlightOption = False):
    
    #Scan Type (User Input) TO DO --> getTypeByClassification
    global userInputDocType
    userInputDocType = scanTypeNum
    
    # get directory path names
    # TODO move to config
    rgbScanPath = filename + '_RGB.png'
    IRtextScanPath = filename + '_IR.png'
    IRemptyScanPath = 'bias.png'

    # Read and assign Scanned Images, in RGBA mode (option = -1)
    rgbImg = cv.imread(rgbScanPath, -1) 
    IRtextImg = cv.imread(IRtextScanPath,-1)
    IRemptyImg = cv.imread(IRemptyScanPath,-1) 
    if (IRemptyImg is None): #falls leere IR scan .png Datei fehlt wird overlay mit zwei identischen IR text scans durchgeführt 
        IRemptyImg = IRtextImg
        print('Empty IR scan file not found. Extraction uses plain IR text scan instead.')
       
    # Assign Filter Mask Values For Exraction 
    global alpha_IROverlay, blackpointThresh_IR, whitepointThresh_IR, blackpointThresh_RGB, whitepointThresh_RGB, thresholdBinary_IR
    alpha_IROverlay, blackpointThresh_IR, whitepointThresh_IR, blackpointThresh_RGB, whitepointThresh_RGB, thresholdBinary_IR = ((cfg.getEffectFilterValuesByScanType(userInputDocType)) if (userInputDocType in cfg.SCAN_TYPE_NUM_LIST) else (cfg.getEffectFilterValuesByScanType()))
    
    #Prepare IR, draw and count contours for image additon 
    irImgWhiteBalanced = overlayIRScans(alpha_IROverlay, IRemptyImg, IRtextImg)
    imgWithContours = addContours(irImgWhiteBalanced)
    irImgContours = prepareIRScan(imgWithContours)

    # Prepare images for addition --> Callibration of Brightness and Colors, Inversions; Delation
    #irImgWhiteBalanced = overlayIRScans(alpha_IROverlay, IRemptyImg, IRtextImg)
    rgbImgChanged = adjustHistogramValues(rgbImg, False)
    #irImgChanged =  prepareIRScan(adjustHistogramValues(irImgWhiteBalanced, True))
    
    
    # Image Addition OLD 
    #addedImg = cv.add(rgbImgChanged, irImgChanged)
    
    #Image Addition NEW  (changed color channels -> merged for contours and Highlighting Option)
    addedContourImg = cv.add(rgbImgChanged, cv.merge((irImgContours,irImgContours,irImgContours)))
    
    #Add yellow/orange markers Highlights if second argument in bash console equals -h
    extractionResultImg = None
    if (highlightOption == True):
        bgrOrigImg = cv.imread(rgbScanPath,cv.IMREAD_COLOR) #Read original Image in BGR mode
        finalImgWithHighlights = addMarkerHighlights(bgrOrigImg, addedContourImg)
        extractionResultImg = finalImgWithHighlights
    else:
        extractionResultImg = addedContourImg
        
    # Save and show final extraction result 
    #saveImage(extractionResultImg, filename + '_annotation.png')
    saveTransparentImage(extractionResultImg, filename + '_annotation.png')
    #saveTransparentImage(extractionResultImg)
    #showImage(extractionResultImg, 'Extraction Result')



if __name__ == "__main__":
    main(cfg.MEDIUM_TEXT, sys.argv[1], True)

    #if len(sys.argv) < 2:
    #    main(cfg.DEFAULT_MODE)
    #    print("AS MAIN - default scan type param : 2")
    #else:
    #    inputNum = int(sys.argv[1])
    #    scanTypeParam = (inputNum if (inputNum in cfg.SCAN_TYPE_NUM_LIST) else cfg.DEFAULT_MODE)
    #    print("AS MAIN - with scan type param : " + str(scanTypeParam))
    #    if (len(sys.argv) > 2):
    #        if(sys.argv[2] == cfg.HIGHLIGHT_OPTION_STR):
    #            main(scanTypeParam, True)
    #        else:
    #            main(scanTypeParam, False)            
    #    else:
    #        main(scanTypeParam)
