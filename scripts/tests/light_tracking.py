import numpy as np
import cv2

cap = cv2.VideoCapture(1)

radius = 5

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # apply a Gaussian blur to the image then find the brightest
    # region
    gray = cv2.GaussianBlur(gray, (radius, radius), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    cv2.circle(frame, maxLoc, 10, (255, 0, 0), 5)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('gray', gray)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
