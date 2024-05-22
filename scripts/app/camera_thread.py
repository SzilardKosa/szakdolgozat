from PyQt5 import QtCore
import pyqtgraph.opengl as gl
import os
import cv2
import time
import numpy as np
import math
from models import Camera, RED, GREEN, BLUE, YELLOW, HUMAN_COLORS_3D, HUMAN_COLORS_2D
from config import NUM_OF_CAMERAS, ACTIVE_CAMERA_MODE
import sba
from itertools import combinations
import itertools
from pose_engine import PoseEngine, KEYPOINTS, EDGES, HEADPOINTS
from copy import deepcopy

class CameraThread(QtCore.QThread):

    # custome signal to update main window
    update_fps_status = QtCore.pyqtSignal(float)
    update_camera_labels = QtCore.pyqtSignal()
    plot_calibration = QtCore.pyqtSignal()
    plot_light = QtCore.pyqtSignal()
    plot_ground = QtCore.pyqtSignal()
    plot_humans = QtCore.pyqtSignal()

    def __init__(self, cameras, plotter):
        super().__init__()

        self.plotter = plotter

        self.cameras = cameras
        self.last_frames = []

        # default camera settings
        self.frame_width = 640
        self.frame_height = 480
        self.frame_per_sec = 30

        for camera in self.cameras:
            if ACTIVE_CAMERA_MODE:
                camera.start()
                camera.set_fps(self.frame_per_sec)
                camera.set_resolution(self.frame_width, self.frame_height)
            self.last_frames.append(None)
        
        self.stop = False
        if ACTIVE_CAMERA_MODE:
            self.state = "get_frames"
        else:
            self.state = "inactive_cameras"

        # real fps measurement variables
        self.time_start = time.time()
        self.real_fps = 0

        # count back parameters
        self.count_back_start = None
        self.count_back_duration = None
        self.after_count_back = None

        # intrinsic parameters
        self.grid_width = None
        self.grid_height = None
        self.num_of_pics = None
        self.cam_to_calib = None

        # extrinsic parameters
        self.radius = None
        self.video_recording = None
        self.video_out = None
        self.extrinsic_imgpoints = [[] for i in range(NUM_OF_CAMERAS)]
        self.method = None
        self.prob = None
        self.threshold = None

        self.point_3d = None
        self.avg_error = None
        self.masks = None
        self.points_2d = None
        self.points_2d_masked = None
        self.points_2d_norm = None
        self.points_2d_norm_hom = None
        self.essential_matrices = None
        self.fundamental_matrices = None

        self.ground_points = None

        self.model = None
        self.dist_threshold = None
        self.min_cam = None
        self.pose_threshold = None
        self.keypoint_threshold = None
        self.prev_humans = None
        self.tracked_humans = None
        self.tracking_threshold = None
        self.filter_alpha = None
        self.human_id = None
    
    def run(self):
        while not self.stop:

            self.real_fps = self.real_fps*0.9 + (1/(time.time()-self.time_start))*0.1
            self.time_start = time.time()

            if self.state == "get_frames":
                self.run_get_frames()
            
            elif self.state == "count_back":
                self.run_count_back()
            
            elif self.state == "video_settings":
                self.run_video_settings()
                self.state = "get_frames"

            elif self.state == "calib_intrinsic":
                self.run_calibrate_intrinsic()
                self.state = "get_frames"

            elif self.state == "collect_extrinsic":
                self.run_collect_extrinsic()
            
            elif self.state == "close_recording":
                self.run_close_recording()
            
            elif self.state == "calib_extrinsic":
                self.run_calibrate_extrinsic()
                self.back_to_default()
            
            elif self.state == "bundle_adjustment":
                self.run_sba()
                self.back_to_default()
            
            elif self.state == "inactive_cameras":
                time.sleep(1)
            
            elif self.state == "mocap":
                self.run_mocap()

            elif self.state == "light_tracking":
                self.run_light_tracking()
            
            self.update_fps_status.emit(self.real_fps)

        for camera in self.cameras:
            camera.stop()
    
    def back_to_default(self):
        if ACTIVE_CAMERA_MODE:
            self.state = "get_frames"
        else:
            self.state = "inactive_cameras"
    
    # default state
    def run_get_frames(self):
        for i, camera in enumerate(self.cameras):
            self.last_frames[i]=camera.get_frame()
    
    # count back state
    def start_count_back(self, duration, next_state):
        self.count_back_start = time.time()
        self.after_count_back = next_state
        self.count_back_duration = duration
        self.state = "count_back"

    def run_count_back(self):
        delta_time = time.time()-self.count_back_start
        if delta_time < self.count_back_duration:
            for i, camera in enumerate(self.cameras):
                frame = camera.get_frame()
                display_text = str(math.ceil(self.count_back_duration-delta_time))
                digits = len(display_text)
                cv2.putText(frame,
                            f"{display_text}",
                            (int(self.frame_width/2)-digits*60, int(self.frame_height/2)+50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            6, (255, 0, 0), 10)
                self.last_frames[i]=frame
        else:
            self.state = self.after_count_back

    # settings state
    def set_res_and_fps(self, setting):
        res, fps = setting.split("@")
        width, height = res.split("x")
        fps, _ = fps.split(" ")
        self.frame_width = int(width)
        self.frame_height = int(height)
        self.frame_per_sec = int(fps)
        self.state = "video_settings"

    def run_video_settings(self):
        for camera in self.cameras:
            camera.set_fps(self.frame_per_sec)
            camera.set_resolution(self.frame_width, self.frame_height)
    
    # intrinsic calib state
    def set_intrinsic(self, delay, width, height, num_of_pics, cam_num):
        self.grid_width = width
        self.grid_height = height
        self.num_of_pics = num_of_pics
        self.cam_to_calib = cam_num
        self.start_count_back(delay, "calib_intrinsic")
    
    def run_calibrate_intrinsic(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((self.grid_width*self.grid_height,3), np.float32)
        objp[:,:2] = np.mgrid[0:self.grid_width,0:self.grid_height].T.reshape(-1,2)
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        camera = self.cameras[self.cam_to_calib]
        imgs = []

        # reading images
        while True:
            for i in range(6):
                img = camera.get_frame()
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (self.grid_width, self.grid_height), None)
            if ret == True:
                print(corners[0])
                objpoints.append(objp)
                
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                print(corners2[0])
                imgpoints.append(corners2)
                imgs.append(img.copy())

                img = cv2.drawChessboardCorners(img, (self.grid_width, self.grid_height), corners2, ret)

                time.sleep(1)
                print(len(imgs))

            self.last_frames[self.cam_to_calib] = img

            if self.num_of_pics == len(imgs):
                break

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # New camera matrix with a free scaling parameter (alpha)
        img = imgs[0]
        alpha = 1
        h,  w = img.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), alpha, (w,h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        # x,y,w,h = roi
        # dst = dst[y:y+h, x:x+w]
        app_dir = os.path.dirname(os.path.realpath(__file__))
        os.chdir("calib_data/intrinsic/calibrated_img")
        cv2.imwrite('calibresult.png', dst)
        cv2.imwrite('orignal.png', img)
        os.chdir(app_dir)

        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error

        print("total error: ", mean_error/len(objpoints))

        camera.set_intrinsic(mtx, dist, mean_error/len(objpoints), newcameramtx, roi, alpha)
        self.update_camera_labels.emit()
    
    # extrinsic calib state
    def start_extrinsic(self, delay, radius, video_recording):
        if radius%2 == 0:
            radius +=1
        self.radius = radius
        self.video_recording = video_recording
        if video_recording:
            if NUM_OF_CAMERAS == 1:
                width = self.frame_width
                height = self.frame_height
            elif NUM_OF_CAMERAS == 2:
                width = self.frame_width*2
                height = self.frame_height
            elif NUM_OF_CAMERAS == 3:
                width = self.frame_width*3
                height = self.frame_height
            elif NUM_OF_CAMERAS == 4:
                width = self.frame_width*2
                height = self.frame_height*2
            self.video_out = cv2.VideoWriter('calib_data/extrinsic/calibration_videos/out.avi',
                                             cv2.VideoWriter_fourcc('M','J','P','G'),
                                             self.frame_per_sec,
                                             (width, height))
        self.extrinsic_imgpoints = [[] for i in range(NUM_OF_CAMERAS)]
        self.start_count_back(delay, "collect_extrinsic")
    
    def stop_extrinsic(self):
        if self.video_recording:
            self.state = "close_recording"
        else:
            self.state = "get_frames"

    def run_collect_extrinsic(self):
        for i, camera in enumerate(self.cameras):
            frame = camera.get_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray = cv2.GaussianBlur(gray, (self.radius, self.radius), 0)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
            cv2.circle(frame, maxLoc, self.radius*2, (255, 0, 0), 5)
            self.extrinsic_imgpoints[i].append(maxLoc)
            self.last_frames[i]=frame
        if self.video_recording:
            if NUM_OF_CAMERAS == 1:
                frame = self.last_frames[0]
            elif NUM_OF_CAMERAS == 2:
                frame1 = self.last_frames[0]
                frame2 = self.last_frames[1]
                frame = np.concatenate((frame1, frame2), axis=1)
            elif NUM_OF_CAMERAS == 3:
                frame1 = self.last_frames[0]
                frame2 = self.last_frames[1]
                frame3 = self.last_frames[2]
                frame = np.concatenate((frame1, frame2, frame3), axis=1)
            elif NUM_OF_CAMERAS == 4:
                frame1 = self.last_frames[0]
                frame2 = self.last_frames[1]
                frame3 = self.last_frames[2]
                frame4 = self.last_frames[3]
                frame_top = np.concatenate((frame1, frame2), axis=1)
                frame_button = np.concatenate((frame3, frame4), axis=1)
                frame = np.concatenate((frame_top, frame_button), axis=0)
            self.video_out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def run_close_recording(self):
        self.video_out.release()
        self.state = "get_frames"

    def save_extrinsic_points(self):
        app_dir = os.path.dirname(os.path.realpath(__file__))
        os.chdir("calib_data/extrinsic/extrinsic_parameters")
        for i in range(NUM_OF_CAMERAS):
            points = np.array(self.extrinsic_imgpoints[i], dtype=int)
            np.savetxt(f'points{i+1}.txt', points, fmt="%d")
        os.chdir(app_dir)

    def load_extrinsic_points(self):
        app_dir = os.path.dirname(os.path.realpath(__file__))
        os.chdir("calib_data/extrinsic/extrinsic_parameters")
        for i in range(NUM_OF_CAMERAS):
            points = np.loadtxt(f'points{i+1}.txt', dtype=int)
            self.extrinsic_imgpoints[i] = points.tolist()
        os.chdir(app_dir)
    
    def set_extrinsic(self, method, prob, threshold):
        if method == "RANSAC":
            self.method = cv2.RANSAC
        elif method == "LMEDS":
            self.method = cv2.LMEDS
        self.prob = prob
        self.threshold = threshold
        self.state = "calib_extrinsic"

    def run_calibrate_extrinsic(self):
        # Calculating relative position between 0-1, 1-2, 2-3, 3-0
        cams1 = [i for i in range(NUM_OF_CAMERAS)]
        cams2 = [i+1 for i in range(NUM_OF_CAMERAS)]
        cams2[-1] = 0

        essential_matrices = []
        masks = []
        points = []
        points_norm = []
        points_norm_hom =[]
        for i in range(NUM_OF_CAMERAS):
            _points = np.array(self.extrinsic_imgpoints[i], dtype=float)
            _points_norm = cv2.undistortPoints(np.expand_dims(_points, axis=1),
                        self.cameras[i].camera_matrix, self.cameras[i].dist_coeffs)
            _points_norm_hom = cv2.convertPointsToHomogeneous(_points_norm)[:,0]
            points.append(_points)
            points_norm.append(_points_norm)
            points_norm_hom.append(_points_norm_hom)
        
        print("############### EXTRINSIC CALIBRATION ###################\n")

        # 1 STEP
        # Essetinal matrix and mask
        #region
        for cam1, cam2 in zip(cams1, cams2):
            points1_norm = points_norm[cam1]
            points2_norm = points_norm[cam2]

            E, mask = cv2.findEssentialMat(points1_norm, points2_norm,
                                            focal=1.0, pp=(0., 0.),
                                            method=self.method,
                                            prob=self.prob,
                                            threshold=self.threshold)
            essential_matrices.append(E)
            masks.append(mask)
            # Essential error
            points1_norm_hom = points_norm_hom[cam1]
            points2_norm_hom = points_norm_hom[cam2]
            error = []
            for i in range(points1_norm_hom.shape[0]):
                error.append(np.dot(points2_norm_hom[i], np.dot(E, points1_norm_hom[i].T)))
            print(f"----------------- ESSENTIAL MATRIX {cam1}-{cam2} -----------------\n")
            print(f"Average E error: {sum(error)/len(error)}")
            print(f"Max E error: {max(error)}")
            print(f"Min E error: {min(error)}\n")
        
        # Filtering the points with the masks
        new_mask = np.ones(masks[0].shape).astype(np.uint8)
        for mask in masks:
            new_mask = np.logical_and(mask, new_mask).astype(np.uint8)

        print(f"---------------- FILTERING OUTLIERS {cam1}-{cam2} ----------------\n")
        print(f"From {new_mask.size}, the number of good points: {np.count_nonzero(new_mask)}")
        print("Percentage of the good points {0:.2f}\n".format(np.count_nonzero(new_mask)/new_mask.size*100))
        
        # Calculating the new points
        new_points = []
        new_points_norm = []
        new_points_norm_hom =[]
        for i in range(NUM_OF_CAMERAS):
            _points = [points[i][j] for j in range(len(points[i])) if new_mask[j]]
            _points_norm = [points_norm[i][j] for j in range(len(points_norm[i])) if new_mask[j]]
            _points_norm_hom = [points_norm_hom[i][j] for j in range(len(points_norm_hom[i])) if new_mask[j]]
            new_points.append(np.array(_points))
            new_points_norm.append(np.array(_points_norm))
            new_points_norm_hom.append(np.array(_points_norm_hom))
        #endregion

        # 2 STEP
        # Recover pose
        #region
        # Putting the first camera in the origo
        self.cameras[0].R = np.eye(3, 3)
        self.cameras[0].t = np.zeros((3, 1))
        self.cameras[0].M = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
        self.cameras[0].P = np.dot(self.cameras[0].camera_matrix, self.cameras[0].M)

        for cam1, cam2 in zip(cams1, cams2):

            print(f"------------------- RECOVER POSE {cam1}-{cam2} -------------------\n")

            pts, R, t, _ = cv2.recoverPose(essential_matrices[cam1], new_points_norm[cam1], new_points_norm[cam2])

            # calculating the scale
            M_1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
            M_2 = np.hstack((R, t))
            point_3d = self.triangulate_points( M_1, M_2, new_points_norm[cam1], new_points_norm[cam2])

            current_sum = np.sum(point_3d)
            print(f"Cam {cam2}")
            self.calculate_reprojection_error(new_points[cam2], point_3d, self.cameras[cam2], R, t)

            if cam1 != 0:
                # calculating the scale from before
                print(f"Scale: {prev_sum/current_sum}\n")
                t*=prev_sum/current_sum

            M_1 = np.hstack((R.T, np.dot(-1*R.T, t)))
            M_2 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
            point_3d = self.triangulate_points( M_1, M_2, new_points_norm[cam1], new_points_norm[cam2])

            prev_sum = np.sum(point_3d)
            print(f"Cam {cam1}")
            self.calculate_reprojection_error(new_points[cam1], point_3d, self.cameras[cam1], R.T, np.dot(-1*R.T, t))
            self.cameras[cam2].R = R
            self.cameras[cam2].t = t
        #endregion

        # 3 STEP
        # Global positions
        #region
        print("--------------------- GLOBAL POSE  ---------------------\n")

        self.cameras[1].M = np.hstack((self.cameras[1].R, self.cameras[1].t))
        self.cameras[1].P = np.dot(self.cameras[1].camera_matrix, self.cameras[1].M)

        for cam in cams2[1:]:
            self.cameras[cam].t += np.dot(self.cameras[cam].R, self.cameras[cam-1].t)
            self.cameras[cam].R = np.dot(self.cameras[cam].R,  self.cameras[cam-1].R)
            self.cameras[cam].M = np.hstack((self.cameras[cam].R, self.cameras[cam].t))
            self.cameras[cam].P = np.dot(self.cameras[cam].camera_matrix, self.cameras[cam].M)

        M_matrices = [cam.M for cam in self.cameras]
        point_3d = self.triangulate_nviews( M_matrices, new_points_norm)        

        repojec_errors = []
        for cam in cams1:
            print(f"Cam {cam}")
            repojec_error = self.calculate_reprojection_error(new_points[cam],
                                                    point_3d,
                                                    self.cameras[cam],
                                                    self.cameras[cam].R,
                                                    self.cameras[cam].t)
            self.cameras[cam].error_extrinsic = sum(repojec_error)/len(repojec_error)
            repojec_errors.append(repojec_error)
        avg_error = np.average(np.array(repojec_errors), axis=0)

        # Storing fundamental/essential matrices
        essential_matrices = []
        fundamental_matrices = []
        for cam1, cam2 in combinations(cams1,2):
            points1_norm = points_norm[cam1]
            points2_norm = points_norm[cam2]

            E, _ = cv2.findEssentialMat(points1_norm, points2_norm,
                                            focal=1.0, pp=(0., 0.),
                                            method=self.method,
                                            prob=self.prob,
                                            threshold=self.threshold)
            F = np.dot(np.linalg.inv(self.cameras[cam2].camera_matrix).T, np.dot(E, np.linalg.inv(self.cameras[cam1].camera_matrix)))
            essential_matrices.append(E)
            fundamental_matrices.append(F)
            # Essential error
            # region
            # points1_norm_hom = points_norm_hom[cam1]
            # points2_norm_hom = points_norm_hom[cam2]
            # error = []
            # for i in range(points1_norm_hom.shape[0]):
            #     error.append(np.dot(points2_norm_hom[i], np.dot(E, points1_norm_hom[i].T)))
            # print(f"----------------- FUNDAMENTAL MATRIX {cam1}-{cam2} -----------------\n")
            # print(f"Average E error: {sum(error)/len(error)}")
            # print(f"Max E error: {max(error)}")
            # print(f"Min E error: {min(error)}\n")
            # points1_hom = cv2.convertPointsToHomogeneous(points[cam1].reshape(-1,1,2))[:,0]
            # points2_hom = cv2.convertPointsToHomogeneous(points[cam2].reshape(-1,1,2))[:,0]
            # error = []
            # for i in range(points1_hom.shape[0]):
            #     error.append(np.dot(points2_hom[i], np.dot(F, points1_hom[i].T)))
            # print(f"Average F error: {sum(error)/len(error)}")
            # print(f"Max F error: {max(error)}")
            # print(f"Min F error: {min(error)}\n")
            # endregion
            

        self.avg_error = avg_error
        self.point_3d = point_3d
        self.points_2d = points
        self.points_2d_masked = new_points
        self.points_2d_norm = points_norm
        self.points_2d_norm_hom = points_norm_hom
        self.masks = masks
        self.essential_matrices = essential_matrices
        self.fundamental_matrices = fundamental_matrices

        self.plot_calibration.emit()
        print("#########################################################")
        #endregion

        self.update_camera_labels.emit()

        # calculate all points
        #region
        #  point_3d = []
        # masks = np.array(self.masks)
        # points_2d_norm = np.array(self.points_2d_norm)
        # for i in range(masks.shape[1]):
        #     mask = masks[:,i,0]
        #     points = points_2d_norm[:,i]
        #     M_matrices = [cam.M for j,cam in enumerate(self.cameras) if mask[j]]
        #     points_norm = [pts.reshape(1,1,2) for j,pts in enumerate(points) if mask[j]]
        #     if 1<len(M_matrices):
        #         point_3d.append(self.triangulate_nviews( M_matrices, points_norm)[0])
        
        # self.point_3d = np.array(point_3d)
        # self.plot_calibration.emit()
        #endregion

        # Writing the points in a video
        # region
        # width = self.frame_width*2
        # height = self.frame_height
        # mask_out = cv2.VideoWriter('calib_data/extrinsic/calibration_videos/mask.avi',
        #                             cv2.VideoWriter_fourcc('M','J','P','G'),
        #                             self.frame_per_sec,
        #                             (width, height))
        # E = essential_matrices[0]
        # F = np.dot(np.linalg.inv(self.cameras[1].camera_matrix).T, np.dot(E, np.linalg.inv(self.cameras[0].camera_matrix)))
        # for i, inlier in enumerate(masks[0]):
        #     frame1 = np.ones((self.frame_height, self.frame_width, 3)).astype(np.uint8)*100
        #     frame2 = np.ones((self.frame_height, self.frame_width, 3)).astype(np.uint8)*90
        #     pt1 = self.extrinsic_imgpoints[0][i]
        #     pt2 = self.extrinsic_imgpoints[1][i]
        #     if inlier:
        #         cv2.circle(frame1, tuple(pt1), 10, (0, 255, 0), 5)
        #         cv2.circle(frame2, tuple(pt2), 10, (0, 255, 0), 5)
        #     else:
        #         cv2.circle(frame1, tuple(pt1), 10, (0, 0, 255), 5)
        #         cv2.circle(frame2, tuple(pt2), 10, (0, 0, 255), 5)
            
        #     pt1 = np.array(pt1, dtype=float)
        #     pt2 = np.array(pt2, dtype=float)
        #     pt1_hom = cv2.convertPointsToHomogeneous(pt1.reshape(-1,1,2))[:,0]
        #     pt2_hom = cv2.convertPointsToHomogeneous(pt2.reshape(-1,1,2))[:,0]
        #     pt1_norm = cv2.undistortPoints(pt1.reshape(-1,1,2),
        #                 self.cameras[0].camera_matrix, self.cameras[0].dist_coeffs)
        #     pt2_norm = cv2.undistortPoints(pt2.reshape(-1,1,2),
        #                 self.cameras[1].camera_matrix, self.cameras[1].dist_coeffs)
        #     pt1_norm_hom = cv2.convertPointsToHomogeneous(pt1_norm)[:,0]
        #     pt2_norm_hom = cv2.convertPointsToHomogeneous(pt2_norm)[:,0]

        #     E_error = np.dot(pt2_norm_hom, np.dot(E, pt1_norm_hom.T))[0,0]
        #     F_error = np.dot(pt2_hom, np.dot(F, pt1_hom.T))[0,0]
        #     print(np.dot(F, pt1_hom.T))
        #     cv2.putText(frame1,
        #                 "y'T*E*y error: {0:+.3f}".format(E_error),
        #                 (10, 50),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 1, (255, 0, 0), 1)
        #     cv2.putText(frame1,
        #                 "x'T*F*x error: {0:+.3f}".format(F_error),
        #                 (10, 100),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 1, (255, 0, 0), 1)
        #     # Find epilines corresponding to points in right image (second image) and
        #     # drawing its lines on left image
        #     lines1 = cv2.computeCorrespondEpilines(pt2.reshape(-1,1,2), 2, F)
        #     lines1 = lines1.reshape(-1,3)
        #     frame1 = self.drawlines(frame1, lines1, [pt1.astype(np.int)])
        #     a = lines1[0,0]
        #     b = lines1[0,1]
        #     c = lines1[0,2]
        #     dist = abs(a*pt1[0]+b*pt1[1]+c)
        #     cv2.putText(frame1,
        #                 "distance: {0:.2f}".format(dist),
        #                 (10, 150),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 1, (255, 0, 0), 1)

        #     # Find epilines corresponding to points in left image (first image) and
        #     # drawing its lines on right image
        #     lines2 = cv2.computeCorrespondEpilines(pt1.reshape(-1,1,2), 1, F)
        #     lines2 = lines2.reshape(-1,3)
        #     frame2 = self.drawlines(frame2, lines2, [pt2.astype(np.int)])
        #     a = lines2[0,0]
        #     b = lines2[0,1]
        #     c = lines2[0,2]
        #     dist = abs(a*pt2[0]+b*pt2[1]+c)
        #     print(lines2)
        #     cv2.putText(frame2,
        #                 "distance: {0:.2f}".format(dist),
        #                 (10, 150),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 1, (255, 0, 0), 1)
            
        #     frame = np.concatenate((frame1, frame2), axis=1)
        #     mask_out.write(frame)
        # mask_out.release()
        # endregion

    def drawlines(self, frame, lines, pts):
        ''' frame - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        h, w, _ = frame.shape
        for l,pt in zip(lines, pts):
            color = (0, 255, 255)
            x0,y0 = map(int, [0, -l[2]/l[1]])
            x1,y1 = map(int, [w, -(l[2]+l[0]*w)/l[1]])
            frame = cv2.line(frame, (x0,y0), (x1,y1), color, 1)
            frame = cv2.circle(frame,tuple(pt),5,color,-1)
        return frame

    def calculate_reprojection_error(self, imgpoints, objpoints, camera, R, t):
        rvec, _ = cv2.Rodrigues(R)
        mtx = camera.camera_matrix
        dist = camera.dist_coeffs
        repojec_error = []
        for i in range(objpoints.shape[0]):
            imgpoint2, _ = cv2.projectPoints(np.expand_dims(objpoints[i], axis=0),
                                              rvec, t, mtx, dist)
            repojec_error.append(cv2.norm(imgpoints[i], imgpoint2[0,0], cv2.NORM_L2))
        
        print(f"Average reprojection error: {sum(repojec_error)/len(repojec_error)}")
        print(f"Max reprojection error: {max(repojec_error)}")
        print(f"Min reprojection error: {min(repojec_error)}\n")
        return repojec_error
    
    def triangulate_points(self, M1, M2, points1_norm, points2_norm):
        """
        triangulation is done from the undistorted points,
        so for the projection matrix we have to use the M matrix
        """
        point_4d_hom = cv2.triangulatePoints(M1,
                                             M2,
                                             points1_norm,
                                             points2_norm)
        point_3d = cv2.convertPointsFromHomogeneous(point_4d_hom.T)[:,0]
        return point_3d
    
    def triangulate_nviews(self, M_matrices, points_norm):
        """
        it calculates the 3d position from the average of all the possible pairs
        """
        point_3ds = []
        for M_pair, points_pair in zip(combinations(M_matrices, 2), combinations(points_norm, 2)):
            point_3ds.append(self.triangulate_points(M_pair[0],
                                                     M_pair[1],
                                                     points_pair[0],
                                                     points_pair[1]))
        point_3d = np.mean(point_3ds, axis=0)
        
        return point_3d

    def start_bundle_adjustment(self):
        self.state = "bundle_adjustment"
    
    def run_sba(self):

        # # Storing the parameters
        # camera_parameters = []
        # for camera in self.cameras:
        #     fx = camera.camera_matrix[0,0]
        #     fy = camera.camera_matrix[1,1]
        #     ar = fx/fy
        #     cx = camera.camera_matrix[0,2]
        #     cy = camera.camera_matrix[1,2]
        #     s  = camera.camera_matrix[0,1]
        #     [r2, r4, t1, t2, r6] = camera.dist_coeffs.tolist()[0]
        #     quaternion = sba.quaternions.quatFromRotationMatrix(camera.R)
        #     print(quaternion.asVector().tolist())
        #     [q0, qi, qj, qk] =  quaternion.asVector().tolist()
        #     [tx], [ty], [tz] = camera.t.tolist()
        #     camera_parameters.append([fx, cx, cy, ar, s,
        #                               r2, r4, t1, t2, r6,
        #                               q0, qi, qj, qk,
        #                               tx, ty, tz])
        # q0, qi, qj, qk =  1., 0., 0., 0.
        # tx, ty, tz = 0., 0., 0.
        # origo_camera = camera_parameters[0][0:10] + [q0, qi, qj, qk, tx, ty, tz]
        # camera_parameters.append(origo_camera)
        # camera_parameters = np.array(camera_parameters)

        # point_parameters = self.point_3d
        # for p2d in self.points_2d:
        #     point_parameters = np.hstack((point_parameters, p2d))
        # point_parameters = np.hstack((point_parameters, self.points_2d[0]))
        
        # cams = sba.Cameras.fromDylan(camera_parameters)
        # pts = sba.Points.fromDylan(point_parameters)
        # app_dir = os.path.dirname(os.path.realpath(__file__))
        # os.chdir("calib_data/extrinsic/sba_output")
        # cams.toTxt("cami.txt")
        # pts.toTxt("ptsi.txt")
        # os.chdir(app_dir)

        # newcams, newpts, info = sba.SparseBundleAdjust(cams, pts)
        # info.printResults()
        
        # app_dir = os.path.dirname(os.path.realpath(__file__))
        # os.chdir("calib_data/extrinsic/sba_output")
        # # newcams.toTxt("camo.txt")
        # # newpts.toTxt("ptso.txt")
        # new_camera_parameters = np.loadtxt("camo.txt")
        # new_point_3d = np.loadtxt("ptso.txt")
        # os.chdir(app_dir)

        # # Updating the parameters
        # for new_parameters, camera in zip(new_camera_parameters, self.cameras):
        #     [fx,cx,cy,ar,s,r2,r4,t1,t2,r6,q0,qi,qj,qk,tx,ty,tz] = new_parameters.tolist()
        #     camera.camera_matrix[0,0] = fx
        #     camera.camera_matrix[1,1] = fx / ar
        #     camera.camera_matrix[0,2] = cx
        #     camera.camera_matrix[1,2] = cy
        #     camera.camera_matrix[0,1] = s
        #     camera.dist_coeffs = np.array([[r2, r4, t1, t2, r6]]) 
        #     quaternion = sba.quaternions.Quaternion(q0, qi, qj, qk)
        #     print(camera.R)
        #     camera.R = quaternion.asRotationMatrix()
        #     print(camera.R)
        #     print(camera.t)
        #     camera.t = np.array([[tx], [ty], [tz]])
        #     print(camera.t)

        # M_matrices = [cam.M for cam in self.cameras]
        # point_3d = new_point_3d[:,:3]

        # for cam in range(4):
        #     print(f"Cam {cam}")
        #     self.calculate_reprojection_error(self.points[cam],
        #                                       point_3d,
        #                                       self.cameras[cam],
        #                                       self.cameras[cam].R,
        #                                       self.cameras[cam].t)
        
        # print(new_point_3d[:,:3].shape)
        # self.point_3d = new_point_3d[:,:3]
        # self.plot_calibration.emit()
        pass
    
    
    def start_light_tracking(self):
        self.radius = 5
        self.thres = 10
        self.prev_light_point_3d = None
        self.stopped = 0
        self.state = "light_tracking"
        self.ground_points = []

    def run_light_tracking(self):
        light_points = []
        for i, camera in enumerate(self.cameras):
            frame = camera.get_frame()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray = cv2.GaussianBlur(gray, (self.radius, self.radius), 0)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
            cv2.circle(frame, maxLoc, self.radius*2, (255, 0, 0), 5)
            light_points.append(np.array(maxLoc, dtype=float))
            self.last_frames[i]=frame
        
        # calculate errors
        distances = np.zeros((NUM_OF_CAMERAS-1, NUM_OF_CAMERAS), dtype=bool)
        for i, (cam1, cam2) in enumerate(combinations([i for i in range(NUM_OF_CAMERAS)],2)):
            points1 = light_points[cam1]
            points2 = light_points[cam2]
            F = self.fundamental_matrices[i]

            lines1 = cv2.computeCorrespondEpilines(points2.reshape(-1,1,2), 2, F)
            lines1 = lines1.reshape(-1,3)
            a = lines1[0,0]
            b = lines1[0,1]
            c = lines1[0,2]
            dist = abs(a*points1[0]+b*points1[1]+c)
            distances[cam1, cam2] = dist<self.thres

        # find pairs starting from the top, so 4,3,2
        points_norm = np.array([cv2.undistortPoints(light_points[i].reshape(-1,1,2),
                        self.cameras[i].camera_matrix, self.cameras[i].dist_coeffs)
                        for i in range(NUM_OF_CAMERAS)])
        M_matrices = np.array([cam.M for cam in self.cameras])
        # 4 cam, 3 cam
        for num in range(4,2,-1):
            for options in combinations([i for i in range(NUM_OF_CAMERAS)], num):
                combos = np.array(list(combinations(list(options),2)))
                if distances[combos[:,0], combos[:,1]].all():
                    self.light_point_3d = self.triangulate_nviews( M_matrices[list(options)], points_norm[list(options)])
                    # collecting 3d points
                    if not self.prev_light_point_3d is None:
                        if cv2.norm(self.prev_light_point_3d, self.light_point_3d, cv2.NORM_L2)<0.01:
                            self.stopped += 1
                            if 100 < self.stopped:
                                self.stopped = 0
                                self.ground_points.append(self.light_point_3d[0])
                                self.plot_ground.emit()
                                if 2<len(self.ground_points):
                                    self.back_to_default()
                                    app_dir = os.path.dirname(os.path.realpath(__file__))
                                    os.chdir("calib_data/extrinsic/extrinsic_parameters")
                                    np.savetxt('ground_points.txt', self.ground_points)
                                    os.chdir(app_dir)
                        else:
                            self.stopped = 0
                    self.prev_light_point_3d = self.light_point_3d
                    self.plot_light.emit()
                    return

    def rotate_to_ground(self):
        """
        [x]Step 1 calcualting plane equation
        [x]Step 2 find intersection (z=0)
        [x]Step 3 calculate the translation to the origo
        [x]Step 4 calculate rotation matrix
        [x]Step 5 apply the transformation for everything and check error
        """
        # loading points
        app_dir = os.path.dirname(os.path.realpath(__file__))
        os.chdir("calib_data/extrinsic/extrinsic_parameters")
        self.ground_points = np.loadtxt('ground_points.txt')
        os.chdir(app_dir)
        self.plot_ground.emit()
        # 3 points: P, Q, R
        pq = self.ground_points[1]-self.ground_points[0]
        pr = self.ground_points[2]-self.ground_points[0]
        n = np.cross(pq, pr)
        n = n/np.linalg.norm(n)
        # s1: a(x-x0)+b*(y-y0)+c*(z-z0)=0   n=(a,b,c)
        # s1: A*x+B*y+C*z=D  (A=a, B=b, C=c, D=a*x0+b*y0+c*z0)
        # s2: z=0
        #
        # e: A*x+B*y=D  (A=a, B=b, D=a*x0+b*y0+c*z0)
        # e: y=-A/B*x+D/B
        A = n[0]
        B = n[1]
        D = np.dot(n,self.ground_points[0])
        # line_points = np.array([(-100, A/B*100+D/B, 0),(100, -A/B*100+D/B, 0)])
        # line = gl.GLLinePlotItem(pos=line_points,
        #                         color=np.array([(1.0, 1.0, 0.0),(1.0, 1.0, 0.0)]),
        #                         width=2, antialias=True, mode='lines')
        # self.plotter.w.addItem(line)

        # t: e(x=0)-> y = D/B
        # t: (0, D/B, 0)
        translation = np.array([[0.0],[D/B],[0.0]])
        # n1: n=(a,b,c)
        # n2: k=(0,0,1)
        k = np.array([0.,0.,1.])
        rotation = self.rodrigues_formula(n, k)
        # transform: calib points, ground points, cameras
        for i in range(len(self.ground_points)):
            self.ground_points[i] -= translation.T[0]
            self.ground_points[i] = np.dot(rotation, self.ground_points[i])
            self.ground_points[i] += translation.T[0]
        self.plot_ground.emit()

        for i in range(len(self.point_3d)):
            self.point_3d[i] -= translation.T[0]
            self.point_3d[i] = np.dot(rotation, self.point_3d[i])
            self.point_3d[i] += translation.T[0]
        
        for cam in self.cameras:
            rot = cam.R.T
            trans = np.dot(-1*cam.R.T, cam.t)
            trans -= translation
            trans = np.dot(rotation, trans)
            trans += translation
            rot = np.dot(rotation, rot)

            cam.R = rot.T
            cam.t = np.dot(-1*rot.T, trans)
            cam.M = np.hstack((cam.R, cam.t))
            cam.P = np.dot(cam.camera_matrix, cam.M)
        
        print("--------------------- ROTATED POSE  ---------------------\n")      
        repojec_errors = []
        for i, cam in enumerate(self.cameras):
            print(f"Cam {i}")
            repojec_error = self.calculate_reprojection_error(self.points_2d_masked[i],
                                                    self.point_3d,
                                                    cam,
                                                    cam.R,
                                                    cam.t)
            cam.error_extrinsic = sum(repojec_error)/len(repojec_error)
            repojec_errors.append(repojec_error)
        avg_error = np.average(np.array(repojec_errors), axis=0)

        self.plot_calibration.emit()
        print("#########################################################")
    
    def rodrigues_formula(self, u, v):
        """
        rotation from u to v:
        v = M*u
        """
        cross = np.cross(u,v)
        dot = np.dot(u,v)

        c = dot
        s = np.linalg.norm(cross)
        axis = cross/s
        [ax, ay, az] = axis
        M = np.array([[ax*ax*(1-c)+c,    ax*ay*(1-c)-s*az, ax*az*(1-c)+s*ay],
                      [ax*ay*(1-c)+s*az, ay*ay*(1-c)+c,    ay*az*(1-c)-s*ax],
                      [ax*az*(1-c)-s*ay, ay*az*(1-c)+s*ax,    az*az*(1-c)+c]])
        return M
    
    def scale_scene(self, first_height):
        """
        Step 1: calculate scale from first camera height
        Step 2: scale everything
        """
        trans = np.dot(-1*self.cameras[0].R.T, self.cameras[0].t)
        scale = first_height/trans[-1]
        # transform: calib points, ground points, cameras
        self.ground_points *= scale
        self.plot_ground.emit()

        self.point_3d *= scale
        
        for i,cam in enumerate(self.cameras):
            cam.t *= scale
            print(f"cam {i} lengt:  {np.linalg.norm(cam.t)}")
            print(f"{np.dot(-1*cam.R.T, cam.t)}")
            cam.M = np.hstack((cam.R, cam.t))
            cam.P = np.dot(cam.camera_matrix, cam.M)
        
        print("--------------------- SCALED POSE  ---------------------\n")      
        repojec_errors = []
        for i, cam in enumerate(self.cameras):
            print(f"Cam {i}")
            repojec_error = self.calculate_reprojection_error(self.points_2d_masked[i],
                                                    self.point_3d,
                                                    cam,
                                                    cam.R,
                                                    cam.t)
            cam.error_extrinsic = sum(repojec_error)/len(repojec_error)
            repojec_errors.append(repojec_error)
        avg_error = np.average(np.array(repojec_errors), axis=0)

        self.plot_calibration.emit()
        print("#########################################################")
        
    
    def start_mocap(self, dist_thres, min_cam, pose_thres, kp_thres, model, track_thres, filter_alpha, video_recording):
        print(dist_thres, min_cam, pose_thres, kp_thres, model, track_thres, filter_alpha, video_recording)
        self.model = model
        if   model == "mobilenet_v1_353x481":
            self.engine = PoseEngine('posenet_models/posenet_mobilenet_v1_075_353_481_quant_decoder_edgetpu.tflite')
        elif model == "mobilenet_v1_481x641":
            self.engine = PoseEngine('posenet_models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')
        elif model == "mobilenet_v1_721x1281":
            self.engine = PoseEngine('posenet_models/posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite')
        self.dist_threshold = dist_thres
        self.min_cam = min_cam
        self.pose_threshold = pose_thres
        self.keypoint_threshold = kp_thres
        self.prev_humans = []
        self.tracked_humans = []
        self.tracking_threshold = track_thres
        self.filter_alpha = filter_alpha
        self.human_id = 0
        self.video_recording = video_recording
        if video_recording:
            width = self.frame_width*2
            height = self.frame_height*2
            self.video_out = cv2.VideoWriter('mocap_recording/out.avi',
                                             cv2.VideoWriter_fourcc('M','J','P','G'),
                                             self.frame_per_sec,
                                             (width, height))
        self.state = "mocap"
    
    def run_mocap(self):
        """
        [x]Step 1 estimate poses
        [x]Step 2 calculate errors between poses
        [x]Step 3 choose related poses
                c1,c2: 0-1 0-2 0-3 1-2 1-3 2-3
                cam1[2]: 11(0), 12(1)
                cam2[2]: 21(2), 22(3)
                cam3[1]: 31(4)
                cam4[3]: 41(5), 42(6), 43(7)
        [x]Step 4 calculate 3d
        [ ]Step 5 apply tracking
        [ ]Step 6 plot results
        """
        self.prev_humans = self.tracked_humans
        self.tracked_humans = []

        all_poses = []
        num_of_poses = []
        frames = []
        for i, camera in enumerate(self.cameras):
            frame = camera.get_frame()
            frames.append(frame.copy())

            if   self.model == "mobilenet_v1_353x481":
                frame = np.uint8(cv2.resize(frame, (481, 353)))
            elif self.model == "mobilenet_v1_481x641":
                frame = np.uint8(cv2.resize(frame, (641, 481)))
            elif self.model == "mobilenet_v1_721x1281":
                frame = np.uint8(cv2.resize(frame, (1281, 721)))
            
            poses, inference_time = self.engine.DetectPosesInImage(frame)

            # resizing keypoints
            for pose in poses:
                for kp in KEYPOINTS:
                    pose.keypoints[kp].yx[0]*=self.frame_height/frame.shape[0]
                    pose.keypoints[kp].yx[1]*=self.frame_width/frame.shape[1]

            # filtering poses
            for pose in poses:
                if pose.score < self.pose_threshold:
                    self.draw_pose(frames[-1], pose, RED)
                    poses.remove(pose)

            all_poses.extend(deepcopy(poses))
            num_of_poses.append(len(poses))
            # print('Inference time: %.fms' % inference_time)
        rolled_num = [sum(num_of_poses[:i]) for i in range(len(num_of_poses)+1)]
        human_colors = [YELLOW for i in range(rolled_num[-1])]

        # calculate errors
        distances = np.zeros((rolled_num[-1], rolled_num[-1]), dtype=bool)
        for i, (c1, c2) in enumerate(combinations([i for i in range(NUM_OF_CAMERAS)],2)):
            c1_indexes = np.arange(rolled_num[c1], rolled_num[c1+1]).tolist()
            c2_indexes = np.arange(rolled_num[c2], rolled_num[c2+1]).tolist()
            F = self.fundamental_matrices[i]

            for (p1, p2) in itertools.product(c1_indexes, c2_indexes):
                avg_dist = self.pose_pair_error(all_poses[p1], all_poses[p2], F)
                # print(p1, p2, avg_dist)
                distances[p1, p2] = avg_dist < self.dist_threshold

        # find pairs starting from the top, so 4,3,2
        for num in range(4, self.min_cam-1, -1):
            for cams in combinations([i for i in range(NUM_OF_CAMERAS)], num):
                """ choosing cameras """
                c_indexes = []
                for c in cams:
                    indexes = np.arange(rolled_num[c], rolled_num[c+1]).tolist()
                    c_indexes.append(indexes)
            
                for options in itertools.product(*c_indexes):
                    """ choosing poses from cams """
                    opt = list(options)
                    combos = np.array(list(combinations(opt, 2)))
                    if distances[combos[:,0], combos[:,1]].all():
                        """ checking every pair """
                        # print(opt)
                        human = self.triangulate_human([self.cameras[cam] for cam in cams], [all_poses[o] for o in opt])
                        # deleting all columns and rows
                        distances[opt, :] = False
                        distances[:, opt] = False
                        if not human: continue

                        ###### tracking the humans #####

                        # calculating tracking point from head points
                        head_points = []
                        for hp in HEADPOINTS:
                            if hp in human:
                                head_points.append(human[hp])
                        if not head_points: continue
                        tracking_point = np.mean(np.array(head_points),axis=0)[:-1]
                        print(tracking_point)

                        # storing distances
                        min_distance = 2*self.tracking_threshold
                        min_index = 0
                        for i,prev in enumerate(self.prev_humans):
                            tracking_distance = cv2.norm(prev['tracking_point'], tracking_point, cv2.NORM_L2)
                            if min_distance > tracking_distance:
                                min_distance = tracking_distance
                                min_index = i
                        
                        if min_distance < self.tracking_threshold:
                            # if the closeset is below the treshold then update
                            human['tracking_point'] = tracking_point
                            human['color'] = self.prev_humans[min_index]['color']
                            human['id'] = self.prev_humans[min_index]['id']
                            self.tracked_humans.append(human)
                            self.prev_humans.pop(min_index)
                        else:
                            # else add the new human to the list
                            human['tracking_point'] = tracking_point
                            human['color'] = HUMAN_COLORS_3D[self.human_id % 5]
                            human['id'] = self.human_id
                            self.human_id += 1
                            self.tracked_humans.append(human)
                        for o in opt:
                            human_colors[o] = HUMAN_COLORS_2D[human['id'] % 5]
                        print(self.tracked_humans)

        self.plot_humans.emit()

        for i, frame in enumerate(frames):
            for j in np.arange(rolled_num[i], rolled_num[i+1]).tolist():
                pose = all_poses[j]
                color = human_colors[j]
                # if pose.score < 0.5: continue
                self.draw_pose(frame, pose, color)
            
            self.last_frames[i]=frame
        
        if self.video_recording:
            frame_top = np.concatenate((self.last_frames[0], self.last_frames[1]), axis=1)
            frame_button = np.concatenate((self.last_frames[2], self.last_frames[3]), axis=1)
            frame = np.concatenate((frame_top, frame_button), axis=0)
            self.video_out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def draw_pose(self, frame, pose, color):
        xys = {}
        right_points = {}
        for label, keypoint in pose.keypoints.items():
            if keypoint.score < self.keypoint_threshold:
                circle_color = RED
            else:
                circle_color = color
                right_points[label] = 1
            xys[label] = (int(keypoint.yx[1]), int(keypoint.yx[0]))
            cv2.circle(frame, (int(keypoint.yx[1]), int(keypoint.yx[0])), 5,
                       circle_color, 5)
        for a, b in EDGES:
            if a not in right_points or b not in right_points:
                line_color = RED
            else:
                line_color = color
            ax, ay = xys[a]
            bx, by = xys[b]
            cv2.line(frame, (ax, ay), (bx, by), line_color, 2)
    
    def pose_pair_error(self, pose1, pose2, F):
        dists = []
        for kp in KEYPOINTS:
            # if one of the keypoints is bad, then we continue
            if pose1.keypoints[kp].score < self.keypoint_threshold or \
               pose2.keypoints[kp].score < self.keypoint_threshold: continue
            point1 = pose1.keypoints[kp].yx[::-1]
            point2 = pose2.keypoints[kp].yx[::-1]

            lines1 = cv2.computeCorrespondEpilines(point2.reshape(-1,1,2), 2, F)
            lines1 = lines1.reshape(-1,3)
            a = lines1[0,0]
            b = lines1[0,1]
            c = lines1[0,2]
            dist = abs(a*point1[0]+b*point1[1]+c)
            dists.append(dist)
        if dists: return sum(dists)/len(dists)
        else: return self.dist_threshold*2
    
    def triangulate_human(self, cams, poses):
        human = {}
        # M_matrices = np.array([cam.M for cam in cams])
        for kp in KEYPOINTS:
            points_norm = []
            M_matrices = []
            for pose, cam in zip(poses, cams):
                if pose.keypoints[kp].score < self.keypoint_threshold: continue
                point = pose.keypoints[kp].yx[::-1]
                points_norm.append(cv2.undistortPoints(point.reshape(-1,1,2),
                                                       cam.camera_matrix,
                                                       cam.dist_coeffs))
                M_matrices.append(cam.M)
            if 1<len(points_norm):
                # human.append(self.triangulate_nviews(M_matrices, points_norm)[0])
                human[kp] = self.triangulate_nviews(M_matrices, points_norm)[0]
        return human

    def close_thread(self):
        self.stop = True
        for camera in self.cameras:
            camera.stop()
        # print("thread closed")
        self.exit()

