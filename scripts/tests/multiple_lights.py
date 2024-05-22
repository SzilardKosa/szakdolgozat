from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2

cap = cv2.VideoCapture(1)

radius = 5

while(True):
    # Capture frame-by-frame
    ret, image = cap.read()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    cv2.imshow('blured',blurred)

    # threshold the image to reveal light regions in the
    # blurred image
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('thresh_1',thresh)

    # perform a series of erosions and dilations to remove
    # any small blobs of noise from the thresholded image
    thresh = cv2.erode(thresh, None, iterations=2)
    cv2.imshow('thresh_2',thresh)
    thresh = cv2.dilate(thresh, None, iterations=4)
    cv2.imshow('thresh_3',thresh)

    # perform a connected component analysis on the thresholded
    # image, then initialize a mask to store only the "large"
    # components
    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
    
    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
    
        # otherwise, construct the label mask and count the
        # number of pixels 
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
    
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 300:
            mask = cv2.add(mask, labelMask)

    # find the contours in the mask, then sort them from left to
    # right
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]
    
    # loop over the contours
    for (i, c) in enumerate(cnts):
        # draw the bright spot on the image
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(cX), int(cY)), int(radius),
            (0, 0, 255), 3)
        cv2.putText(image, "#{}".format(i + 1), (x, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('frame',image)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
