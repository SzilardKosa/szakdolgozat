"""
This modul tests VideoCapture with multiple cameras.
"""
import cv2
import time

# Turning on cameras
caps = []
cam_num = 4 # number of cameras to use NOTE: change this for the curret setup
fps_setting = 30 # setting the frame rate of the cameras NOTE: change this

for i in range(cam_num):
    cap = cv2.VideoCapture(i+1)
    cap.set(5, fps_setting)
    ret, frame = cap.read() # reading one frame to turn on the cameras
    caps.append(cap)


# Measuring fps
fps = caps[0].get(cv2.CAP_PROP_FPS)
print(f"Frames per second using cap.get(cv2.CAP_PROP_FPS) : {fps}")

num_frames = 1200

print(f"Capturing {num_frames} frames")

start = time.time()

for i in range(0, num_frames) :
    for j, cap in enumerate(caps):
        ret, frame = cap.read()
        cv2.imshow(f'{j}_frame',frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

seconds = time.time() - start
print("Time taken : {0} seconds".format(seconds))
fps  = num_frames / seconds
print("Estimated frames per second : {0}".format(fps))


# # Viewing picutres of the cameras
# while(True):
#     for i, cap in enumerate(caps):
#         ret, frame = cap.read()
#         cv2.imshow(f'{i}_frame',frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()