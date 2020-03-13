#The starting point of the backend works
#The image URL from UI is fed to the perform_ocr() here

from segmentation_words import segmentLines

import cv2
import os

#input image should have white bg, black text
#raw_image

def detectLines(img_url):

    raw_image = cv2.imread(img_url,0)
    segmentLines(raw_image, img_url)

    return None
