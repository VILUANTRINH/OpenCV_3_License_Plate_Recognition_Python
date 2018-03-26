# Main.py

import cv2
import numpy as np
import os
from os.path import join
import argparse
import glob
import Main

import DetectChars
import DetectPlates
import PossiblePlate

# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

save = False
showSteps = False
sourceFolder = ""
targetFolder = ""
###################################################################################################
def main():    
    # Argument parse
    parser = argparse.ArgumentParser(usage="A training program for recognizing digits and characters")
    subparsers = parser.add_subparsers(help='sub-command help')
    
    # detect image
    detectImage = subparsers.add_parser("detect", help="detect -h")
    detectImage.add_argument("--save", type=bool, default=False, help="Save final image result or not, default = False")
    detectImage.add_argument("--steps", type=bool, default=False, help="Show steps or not, default = False")
    detectImage.add_argument("image", type=str, help="path for the image")
    
    # gen imgages
    genImage = subparsers.add_parser("gen", help="gen -h")
    genImage.add_argument("gen", type=str, nargs=2, default=False, help="source image folder/ target folder")
    
    args = parser.parse_args()
    # Check gen images
    if hasattr(args, "gen"):
        [Main.sourceFolder, Main.targetFolder] = args.gen
        images = []
        exts = ['*.png', '*.jpg', '*.jpeg']
        for ext in exts:
            for filePath in glob.glob(join(Main.sourceFolder, ext)):
                im = cv2.imread(filePath)
                listOfPossiblePlates = DetectPlates.detectPlatesInScene(im, filePath)
                listOfPlates = DetectChars.detectCharsInPlates(listOfPossiblePlates, filePath)
        return
    
    Main.save = args.save; Main.showSteps = args.steps
    
    print(save, showSteps)
    if save == True and not os.path.isdir("outputs"):
        os.makedirs("outputs")

    filePath = args.image
    print(args.image)
    imgOriginalScene  = cv2.imread(filePath)               # open image

    if imgOriginalScene is None:                            # if image was not read successfully
        print("\nerror: image not read from file \n\n")      # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit program
    # end if

    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene, filePath)           # detect plates

    listOfPlates = DetectChars.detectCharsInPlates(listOfPossiblePlates, filePath)        # detect chars in plates
    print("Number of Plates Found:", len(listOfPlates))
    cv2.imshow("imgOriginalScene", imgOriginalScene)            # show scene image

    if len(listOfPlates) == 0:                          # if no plates were found
        print("\nno license plates were detected\n")             # inform user no plates were found
    else:                                                       # else
                # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
        for i, licPlate in enumerate(listOfPlates):
            cv2.imshow("imgPlate_"+str(i), licPlate.imgPlate)           # show crop of plate and threshold of plate
            cv2.imshow("imgThresh_"+str(i), licPlate.imgThresh)

            if len(licPlate.strChars) == 0:                     # if no chars were found in the plate
                print("\nno characters were detected\n\n")       # show message
            # end if

            drawRedRectangleAroundPlate(imgOriginalScene, licPlate)             # draw red rectangle around plate

            print("\nlicense plate read from image = " + licPlate.strChars + "\n")       # write license plate text to std out
            print("----------------------------------------")

            writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)           # write license plate text on the image

        cv2.imshow("imgOriginalScene", imgOriginalScene)                # re-show scene image

        cv2.imwrite("outputs/imgOriginalScene.png", imgOriginalScene)           # write image out to file

        cv2.waitKey(0)					# hold windows open until user presses a key
    # end if else
    cv2.destroyAllWindows()
    return
# end main

###################################################################################################
def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):

    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)            # get 4 vertices of rotated rect

    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)         # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)
# end function

###################################################################################################
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                             # this will be the center of the area the text will be written to
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                          # this will be the bottom left of the area that the text will be written to
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                      # choose a plain jane font
    fltFontScale = float(plateHeight) / 100.0                    # base font scale on height of plate area
    intFontThickness = int(round(fltFontScale * 1.5))           # base font thickness on font scale

    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)        # call getTextSize

            # unpack roatated rect into center point, width and height, and angle
    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)              # make sure center is an integer
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)         # the horizontal location of the text area is the same as the plate

    if intPlateCenterY < (sceneHeight * 0.75):                                                  # if the license plate is in the upper 3/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      # write the chars in below the plate
    else:                                                                                       # else if the license plate is in the lower 1/4 of the image
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      # write the chars in above the plate
    # end if

    textSizeWidth, textSizeHeight = textSize                # unpack text size width and height

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))           # calculate the lower left origin of the text area
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY - intPlateHeight / 2)          # based on the text area center, width, and height

            # write the text on the image
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)
# end function

###################################################################################################
if __name__ == "__main__":
    main()