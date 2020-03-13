
from preprocess import smoothen_binarize_and_dilate
from preprocess import smoothen_and_binarize
from preprocess import getTransformationMatrix
from preprocess import rotate
from LineSegmentation import get_lines_threshold, findLines
import cv2
import numpy as np
import os


cwd = os.getcwd()


def segmentLines(raw_image, img_url):


    #Returns a list/array of all the words found along with the number of words on each line.

    #preprocessing of the image

    #img_for_det used for detecting the character and lines boundaries
    img_for_det = smoothen_and_binarize(raw_image)

    #img_for_ext used for the actual extraction of the characters
    img_for_ext = smoothen_binarize_and_dilate(raw_image)

    #get the rotated angle of the tilt
    #M = getTransformationMatrix(img_for_det) # M is transformation matrix
    #rotate the iamge with M
    #img_for_det = rotate(img_for_det,M)
    #rotate image that will be used for extraction too
    #img_for_ext = rotate(img_for_ext,M)

    H,W = img_for_ext.shape
    #get threshold to determine how much gap should be considered as the line gap
    LinesThres = get_lines_threshold(40, img_for_det)
    ycoords = findLines(img_for_det, LinesThres)
    #print("coords: {}".format(ycoords))

    # crop out text lines from images and save them
    if not os.path.exists(cwd+"/dataset/"+img_url.split("/")[-1][:-4]+"/lines"):

        os.makedirs(cwd+"/dataset/"+img_url.split("/")[-1][:-4]+"/lines")

    if len(ycoords)%2 !=0:
        ycoords.append(H)
    portions = list(zip(ycoords[::2], ycoords[1::2]))

    portions += [(portions[i][-1], portions[i+1][0]) for i in range(len(portions)-1)]

    #print(portions)
    for idx, portion in enumerate(portions):
        cropped_im= img_for_ext[portion[0]:portion[1], 0: W]
        #cv2.imshow(str(idx)+"th cropped image", cropped_im)
        cv2.imwrite(cwd+"/dataset/"+img_url.split("/")[-1][:-4]+"/lines/"+str(idx)+".png", cropped_im)



    # save image with lines printed =======
    if not os.path.exists(cwd+"/dataset/"+img_url.split("/")[-1][:-4]+"/img_with_lines"):

        os.makedirs(cwd+"/dataset/"+img_url.split("/")[-1][:-4]+"/img_with_lines")
    img_with_lines = img_for_ext.copy()
    for i in ycoords:
        cv2.line(img_with_lines,(0,i),(img_with_lines.shape[1],i),255,1)
    cv2.imwrite(cwd+"/dataset/"+img_url.split("/")[-1][:-4]+"/img_with_lines/"+img_url.split("/")[-1][:-4]+".png", img_with_lines)
    print("done saving to", cwd+"/dataset/"+img_url.split("/")[-1][:-4]+"/lines")



    return None
