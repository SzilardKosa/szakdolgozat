import numpy as np
import cv2

# parameters
grid_width = 6
grid_height = 9
max_img_num = 30
cam_index = 1

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((grid_width*grid_height,3), np.float32)
objp[:,:2] = np.mgrid[0:grid_width,0:grid_height].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

cap = cv2.VideoCapture(cam_index)
imgs = []

while(True):
    ready, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (grid_width,grid_height), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        imgs.append(img.copy())

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (grid_width,grid_height), corners2,ret)
    
        cv2.imshow('img',img)
        key = cv2.waitKey(1000) & 0xFF

        print(len(imgs))
    else:
        cv2.imshow('img',img)
        key = cv2.waitKey(1) & 0xFF

    if key == 27 or max_img_num-1<len(imgs):
        break

cap.release()
cv2.destroyAllWindows()

# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# New camera matrix with a free scaling parameter (alpha)
img = imgs[0]
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png', dst)
cv2.imwrite('orignal.png', img)

print(mtx)
print(newcameramtx)
# print(rvecs)
# print(tvecs)

# Calculating error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print("total error: ", mean_error/len(objpoints))




# Drawing 3d cube on the chessboard
axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img

for img in imgs:
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (grid_width,grid_height), None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors again. We should get the same results
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
        # print(rvecs)
        # print(tvecs)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img, corners2, imgpts)
        cv2.imshow('img',img)
        k = cv2.waitKey(1000) & 0xff
        if k == 27:
            break

cv2.destroyAllWindows()