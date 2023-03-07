import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure


class labAnalysisHelper:
    def __init__(self, imagePath):
        self.imagePath = imagePath
        self.extension = imagePath.split(".")[-1]
        self.image = cv.imread(self.imagePath, 0)
        self.currentImage = self.image

    def selectDefaultIfNone(self, image=None):
        if image is None:
            image = self.currentImage
        return image
        
    def showImage(self, image=None, title="Image"):
        image = self.selectDefaultIfNone(image)
        
        plt.figure(figsize=(10, 10))
        plt.title(title)
        plt.imshow(image, cmap='gray')
        plt.show()

    def showHistogram(self, image=None, title="Histogram"):
        image = self.selectDefaultIfNone(image)
        
        plt.figure(figsize=(10, 10))
        plt.title(title)
        plt.hist(image.ravel(), 256, [0, 256])
        plt.show()

    def saveImage(self, name="output", image=None, folder="output"):
        image = self.selectDefaultIfNone(image)
        
        cv.imwrite(f"{folder}/{name}.{self.extension}", image)
        print(f"Saved {folder}/{name}.{self.extension}")

    def removeOutterEdge(self, image=None, radius=5550):
        image = self.selectDefaultIfNone(image)

        mask = np.zeros(image.shape, np.uint8)
        cv.circle(mask, (image.shape[0]//2, image.shape[1]//2), radius, 255, -1)
        masked = cv.bitwise_and(image, image, mask=mask)

        self.currentImage = masked

        return masked
    

    def applyOtsuThreshold(self, image=None):
        image = self.selectDefaultIfNone(image)
        
        # ret, thresh = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        thresh = cv.adaptiveThreshold(image ,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,35,5)
        self.currentImage = thresh
        # print(f"Otsu threshold: {thr}")

        return thresh
    
    def applyThreshold(self, threshold=127, image=None):
        image = self.selectDefaultIfNone(image)
        
        # ret, thresh = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
        # self.currentImage = thresh
        thresh = cv.adaptiveThreshold(image ,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,101,-5)
        self.currentImage = thresh

        print(f"Threshold")

        return thresh
    
    def applyWaterShed(self, binaryImage=None):

        binaryImage = self.selectDefaultIfNone(binaryImage)
        dist3d = cv.cvtColor(binaryImage, cv.COLOR_GRAY2BGR)
        dist3d = dist3d.astype(np.uint8)
        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv.morphologyEx(binaryImage,cv.MORPH_OPEN,kernel, iterations = 2)
        # sure background area
        sure_bg = cv.dilate(opening,kernel,iterations=3)
        # Finding sure foreground area
        dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
        ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0

        markers = cv.watershed(dist3d, markers)
        imgcopy = dist3d.copy()
        imgcopy[markers == -1] = [0, 255, 0]

        #draw circles on the image to show the center of the circles and the radius
        for i in range(0, len(markers)):
            for j in range(0, len(markers[i])):
                if markers[i][j] == -1:
                    cv.circle(imgcopy, (j, i), 10, (0, 255, 0), 2)

        self.currentImage = imgcopy
