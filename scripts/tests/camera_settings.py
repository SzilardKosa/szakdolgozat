import cv2
import os

cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

print(cap.get(cv2.CAP_PROP_FPS))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

os.system("qv4l2 -d /dev/video1 &")

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame',frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()