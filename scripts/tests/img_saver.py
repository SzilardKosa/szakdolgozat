import cv2

cap = cv2.VideoCapture(1)

i=0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame',frame)

    k = cv2.waitKey(1) & 0xff
    if k == ord('s'):
        cv2.imwrite(f'frame_{i}.png', frame)
        i+=1
    if k == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()