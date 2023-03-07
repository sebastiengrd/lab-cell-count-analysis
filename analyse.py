
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from labAnalysisHelper import labAnalysisHelper
import os


def analyseImage(imagePath):

    print(f"Analysing {imagePath}")

    name = imagePath.split("/")[-1].split(".")[0]

    outputFolderPath = f"output/{name}/"

    if(not os.path.exists(outputFolderPath)):
        os.makedirs(outputFolderPath)
    else:
        #remove all files in folder
        for file in os.listdir(outputFolderPath):
            os.remove(os.path.join(outputFolderPath, file))





    # labAnalysis = labAnalysisHelper("2023-02-02_MM-parameter_Jillian02_0h_DAPI_HC2.jpg")
    labAnalysis = labAnalysisHelper(imagePath)


    removedOutterEdge = labAnalysis.removeOutterEdge()
    # labAnalysis.showImage()


    # labAnalysis.showHistogram()


    labAnalysis.applyThreshold(42, removedOutterEdge)
    # labAnalysis.showImage()
    labAnalysis.saveImage("thresholded", folder=outputFolderPath)
    thresholded = labAnalysis.currentImage.copy()


    #finalCopyAnnotated = cv.imread("image1.jpg")
    finalCopyAnnotated = cv.imread(imagePath)


    # Find all the regions in thresholded that have a surface area of more than 113 pixels
    # and label them
    ret, labels = cv.connectedComponents(thresholded)

    component_sizes = np.bincount(labels.flat)
    large_components = np.where(component_sizes > 113)[0]
    mask = np.zeros(labels.shape, dtype=np.uint8)
    mask[np.isin(labels, large_components)] = 255

    bigRegionsMask = cv.bitwise_and(mask, mask, mask=thresholded)

    totalNumberOfCellsForBigRegions = 0

    contours, hierarchy = cv.findContours(bigRegionsMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
    for cnt in contours:
        # Find the area of the contour
        area = cv.contourArea(cnt)

        # Find the centroid of the contour
        M = cv.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        numberOfCells = int(area/30)
        totalNumberOfCellsForBigRegions += numberOfCells

        # Draw a circle and text on the original image
        cv.circle(finalCopyAnnotated, (cX, cY), 10, (0, 255, 0), 1)
        cv.putText(finalCopyAnnotated, str(numberOfCells), (cX - 20, cY - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Show the original image with the circles
    labAnalysis.saveImage("maskOfBigRegions", finalCopyAnnotated, folder=outputFolderPath)


    notBigRegion = cv.bitwise_not(bigRegionsMask)
    thresholdedWithoutBigRegions = cv.bitwise_and(thresholded, thresholded, mask=notBigRegion)
    # labAnalysis.saveImage("maskOfBigRegions", finalCopyAnnotated, folder=outputFolderPath)
    labAnalysis.saveImage("thresholdedWithoutBigRegions", thresholdedWithoutBigRegions, folder=outputFolderPath)
    circles = cv.HoughCircles(thresholdedWithoutBigRegions,cv.HOUGH_GRADIENT,1,5, param1=10,param2=5,minRadius=1,maxRadius=6)

    circles = np.uint16(np.around(circles))


    totalNumberOfCellsForIndividualCircles = 0

    for i in circles[0,:]:
        # draw the outer circle
        cv.circle(finalCopyAnnotated,(i[0],i[1]),i[2],(0,255,0),1)

        totalNumberOfCellsForIndividualCircles += 1

    labAnalysis.saveImage("finalAnnonated", finalCopyAnnotated, folder=outputFolderPath)

    # print(f"Total number of cells: {totalNumberOfCellsForBigRegions + totalNumberOfCellsForIndividualCircles}")
    print(f"Total number of cells for image {name}: {totalNumberOfCellsForBigRegions + totalNumberOfCellsForIndividualCircles}")
    print(f"----------- Done analysing {name} -----------")


    return totalNumberOfCellsForBigRegions + totalNumberOfCellsForIndividualCircles


if __name__=="__main__":
    analyseImage("toAnalyse/2023-02-02_MM-parameter_Jillian02_0h_DAPI_HC2.jpg")
