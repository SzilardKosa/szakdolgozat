from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QFile, QTextStream
from style_sheets import breeze_resources
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import os
import sys
import cv2
import time
import numpy as np
from models import Camera, Plotter, matrix_template, HUMAN_COLORS_3D
from camera_thread import CameraThread
from config import NUM_OF_CAMERAS, ACTIVE_CAMERA_MODE, CAM_NUMS
from pose_engine import KEYPOINTS, EDGES

import main_window

class MainWindow(QtWidgets.QMainWindow, main_window.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        # adding image item to tab cam
        self.img_cam = pg.ImageItem()
        self.add_image_item(self.graphics_view_cam, self.img_cam)

        # adding image items to tab sys
        self.imgs_sys = [pg.ImageItem() for i in range(NUM_OF_CAMERAS)]
        self.graphics_view_sys = [self.graphics_view_sys_1, self.graphics_view_sys_2,
                                  self.graphics_view_sys_3, self.graphics_view_sys_4]
        for i, img in enumerate(self.imgs_sys):
            self.add_image_item(self.graphics_view_sys[i], img)
        
        # adding 3d plot to the placeholder widget
        self.plotter = Plotter()
        self.plotter.add_plot(self.grid_layout_plot3d)
        self.plotter.add_grid()
        # sp = gl.GLScatterPlotItem(pos=np.array([(1, 0, 0),(1, 1, 0)]), size=np.array([0.5, 0.5]), color=np.array([(0.0, 1.0, 0.0, 0.5),(1.0, 1.0, 0.0, 0.5)]), pxMode=False)
        # self.plotter.add_scatter_plot(sp)

        # extrinsic calib point indicator
        self.lb_point_indicator.setAlignment(QtCore.Qt.AlignCenter)
        self.reset_point_indicator()

        # initing the cameras
        self.cameras = []
        for cam_num in CAM_NUMS:
            camera = Camera(cam_num)
            self.cameras.append(camera)

        # adding camera thread
        self.camera_thread = CameraThread(self.cameras, self.plotter)
        self.camera_thread.start()

        if ACTIVE_CAMERA_MODE:
            # starting the image update timer
            self.update_timer = QtCore.QTimer()
            self.update_timer.timeout.connect(self.update_images)
            self.update_timer.start(30)

        # setting up camera choosing combo box
        for i in range(NUM_OF_CAMERAS):
            self.combo_box_camera.insertItem(i, f"cam {i+1}")


        # initing the intinsic parameters
        for i in range(NUM_OF_CAMERAS):
            self.load_intrinsic(cam_num=i)

        # Connections
        # settings
        self.push_button_qv4l2.clicked.connect(self.start_qv4l2)
        self.combo_box_video.currentTextChanged.connect(self.camera_thread.set_res_and_fps)
        self.camera_thread.update_fps_status.connect(self.update_status_bar)
        # intrinsic
        self.pb_calib_intrinsic.clicked.connect(self.start_intrinsic_calib)
        self.pb_save_intrinsic.clicked.connect(self.save_intrinsic)
        self.pb_load_intrinsic.pressed.connect(self.load_intrinsic)
        self.camera_thread.update_camera_labels.connect(self.update_camera_labels)
        self.combo_box_camera.currentIndexChanged.connect(self.update_camera_labels)
        # extrinsic
        self.pb_start_extrinsic.clicked.connect(self.start_extrinsic_collecting)
        self.pb_stop_extrinsic.clicked.connect(self.camera_thread.stop_extrinsic)
        self.pb_stop_extrinsic.clicked.connect(self.set_point_indicator)
        self.pb_save_points.clicked.connect(self.camera_thread.save_extrinsic_points)
        self.pb_load_points.clicked.connect(self.camera_thread.load_extrinsic_points)
        self.pb_load_points.clicked.connect(self.set_point_indicator)

        self.pb_calib_extrinsic.clicked.connect(self.start_extrinsic_calib)
        self.camera_thread.plot_calibration.connect(self.plot_calibration)

        self.pb_bundle_adjustment.clicked.connect(self.camera_thread.start_bundle_adjustment)

        self.pb_start_ground.clicked.connect(self.camera_thread.start_light_tracking)
        self.pb_stop_ground.clicked.connect(self.camera_thread.back_to_default)
        self.pb_rotate.clicked.connect(self.camera_thread.rotate_to_ground)
        self.camera_thread.plot_light.connect(self.plot_light)
        self.camera_thread.plot_ground.connect(self.plot_ground)

        self.pb_scale.clicked.connect(self.start_scale)

        # mocap
        self.pb_start_mocap.clicked.connect(self.start_mocap)
        self.pb_stop_mocap.clicked.connect(self.camera_thread.stop_extrinsic) # because of the recording
        self.camera_thread.plot_humans.connect(self.plot_humans)

        # plot
        self.pb_clear_all.clicked.connect(self.plotter.clear_all)
        self.pb_show_results.clicked.connect(self.plot_calibration)

    ################################ Connections ###################################
    # settings
    def start_qv4l2(self):
        current_camera = self.combo_box_camera.currentIndex()+1
        os.system(f"qv4l2 -d /dev/video{current_camera} &")

    def update_status_bar(self, real_fps):
        self.status_bar.showMessage("FPS: {0:.2f}".format(real_fps))

    # intrinsic
    def start_intrinsic_calib(self):
        delay = self.sb_delay_intrinsic.value()
        width = self.sb_grid_width.value()
        height = self.sb_grid_height.value()
        num_of_pics = self.sb_num_of_pics.value()
        cam_num = self.combo_box_camera.currentIndex()
        self.camera_thread.set_intrinsic(delay, width, height, num_of_pics, cam_num)
    
    def save_intrinsic(self):
        cam_num = self.combo_box_camera.currentIndex()
        camera = self.cameras[cam_num]
        app_dir = os.path.dirname(os.path.realpath(__file__))
        os.chdir("calib_data/intrinsic/intrinsic_parameters")
        np.savez(f'camera{cam_num}.npz', camera_matrix=camera.camera_matrix,
                                        dist_coeffs=camera.dist_coeffs,
                                        err=np.array(camera.error_intrinsic))
        os.chdir(app_dir)

    def load_intrinsic(self, cam_num=None):
        if cam_num is None:
            cam_num = self.combo_box_camera.currentIndex()
        camera = self.cameras[cam_num]
        app_dir = os.path.dirname(os.path.realpath(__file__))
        os.chdir("calib_data/intrinsic/intrinsic_parameters")
        data = np.load(f'camera{cam_num}.npz')
        camera.camera_matrix = data["camera_matrix"]
        camera.dist_coeffs = data["dist_coeffs"]
        camera.error_intrinsic = data["err"]
        os.chdir(app_dir)
        self.update_camera_labels()

    def update_camera_labels(self):
        cam_num = self.combo_box_camera.currentIndex()
        camera = self.cameras[cam_num]
        self.label_camera_matrix.setText(matrix_template(camera.camera_matrix))
        self.label_dist_coeffs.setText(matrix_template(camera.dist_coeffs))
        self.label_error_intrinsic.setText("{0:.3f}".format(camera.error_intrinsic))
        self.label_new_matrix.setText(matrix_template(camera.new_camera_matrix))
        self.label_rotation_matrix.setText(matrix_template(camera.R))
        self.label_translation_vect.setText(matrix_template(camera.t))
        self.label_error_extrinsic.setText("{0:.3f}".format(camera.error_extrinsic))
    
    # extrinsic
    def start_extrinsic_collecting(self):
        delay = self.sb_delay_extrinsic.value()
        radius = self.sb_light_radius.value()
        video_recoring = self.cb_video_recording.isChecked()
        self.camera_thread.start_extrinsic(delay, radius, video_recoring)
    
    def start_extrinsic_calib(self):
        method = self.combo_box_method.currentText()
        prob = self.dsp_prob.value()
        threshold = self.dsp_threshold.value()
        self.camera_thread.set_extrinsic(method, prob, threshold)
    
    def set_point_indicator(self):
        pixmap = QtGui.QPixmap(":/dark/radio_checked.svg")
        self.lb_point_indicator.setPixmap(pixmap)
    
    def reset_point_indicator(self):
        pixmap = QtGui.QPixmap(":/dark/radio_checked_disabled.svg")
        self.lb_point_indicator.setPixmap(pixmap)
    
    def start_scale(self):
        first_height = self.dsb_first_height.value()
        self.camera_thread.scale_scene(first_height)

    # mocap
    def start_mocap(self):
        # reconstruction parameters
        dist_thres = self.sb_dist_threshold.value()
        min_cam = self.sb_min_cam.value()
        # estimator parameters
        pose_thres = self.dsb_pose_threshold.value()
        kp_thres = self.dsb_keypoint_threshold.value()
        model = self.combo_box_models.currentText()
        # tracking parameters
        track_thres = self.dsb_tracking_threshold.value()
        filter_alpha = self.dsb_filter.value()
        video_recoring = self.cb_record_mocap.isChecked()
        self.camera_thread.start_mocap(dist_thres, min_cam, pose_thres, kp_thres, model, track_thres, filter_alpha, video_recoring)

    ################################ Plots ###################################

    def add_image_item(self, graphics_view, img):
        graphics_view.setAspectLocked()
        graphics_view.invertY()
        graphics_view.addItem(img)
    
    def update_images(self):
        current_tab = self.tab_widget.currentWidget().objectName()
        if current_tab == "tab_cam":
            current_camera = self.combo_box_camera.currentIndex()
            try:
                img = self.camera_thread.last_frames[current_camera]
                self.img_cam.setImage(img.transpose([1,0,2]))
            except:
                pass
        
        elif current_tab == "tab_sys":
            for current_camera in range(NUM_OF_CAMERAS):
                try:
                    img = self.camera_thread.last_frames[current_camera]
                    self.imgs_sys[current_camera].setImage(img.transpose([1,0,2]))
                except:
                    pass

        elif current_tab == "tab_plot":
            pass
    
    def plot_calibration(self):
        # Scatter plot
        point_3d = self.camera_thread.point_3d
        length = point_3d.shape[0]
        avg_error = self.camera_thread.avg_error
        max_error = max(avg_error)
        min_error = min(avg_error)
        avg = sum(avg_error)/len(avg_error)
        # print(max_error, min_error, avg)

        coloring_option = self.combo_box_color.currentText()
        if coloring_option == "time":
            color = np.array([(0.0, 1.0-i/length, 0.0, 0.5) for i in range(length)])
        elif coloring_option == "avg error":
            color = np.array([((e-min_error)/(max_error-min_error), 0.0, 0.0, 1.0) for e in avg_error])
        elif coloring_option == "error groups":
            group_color = [(0., 1., 0., 0.5), (1., 1., 0., 0.5), (1., 0., 0., 0.5)]
            intervalls = [5, 10]
            groups = np.searchsorted(intervalls, avg_error)
            color = np.array([group_color[g] for g in groups])
        
        if coloring_option != "empty":
            sp = gl.GLScatterPlotItem(pos=point_3d, size=np.array([0.03 for i in range(length)]),
                color=color, pxMode=False)
            self.plotter.add_scatter_plot(sp)


        # Cameras and axises
        self.plotter.remove_all_camera()
        self.plotter.remove_all_axis()

        # self.plotter.add_axis(scale=0.2)
        # self.plotter.add_camera(scale=0.2)

        for cam in self.cameras:
            R = cam.R
            t = cam.t
            self.plotter.add_camera(rot=R.T, trans=np.dot(-1*R.T, t).T, scale=0.2)
            self.plotter.add_axis(rot=R.T, trans=np.dot(-1*R.T, t).T, scale=0.2)
    
    def plot_light(self):
        light_point_3d = self.camera_thread.light_point_3d
        
        sp = gl.GLScatterPlotItem(pos=light_point_3d, size=np.array([0.05]),
             color=np.array([(0.0, 1.0, 1.0, 0.5)]), pxMode=False)
        self.plotter.add_light_plot(sp)
    
    def plot_ground(self):
        ground_points = np.array(self.camera_thread.ground_points)
        length = ground_points.shape[0]

        sp = gl.GLScatterPlotItem(pos=ground_points, size=np.array([0.1 for i in range(length)]),
             color=np.array([(1.0, 1.0, 0.0, 1.0) for i in range(length)]), pxMode=False)
        self.plotter.add_scatter_plot(sp)
    
    def plot_humans(self):
        humans = self.camera_thread.tracked_humans
        self.plotter.remove_all_scatter_plot()
        self.plotter.remove_all_line_plot()
        
        for c,human in enumerate(humans):
            xys = {}
            for kp in KEYPOINTS:
                if kp in human:
                    xys[kp]=1
                    # print(type(human[kp]), human[kp].shape)
                    sp = gl.GLScatterPlotItem(pos=human[kp].reshape(-1,3), size=np.array([0.1]),
                        color=np.array([human['color']]), pxMode=False)
                    self.plotter.add_scatter_plot(sp)
            for a, b in EDGES:
                if a not in xys or b not in xys: continue
                pa = human[a]
                pb = human[b]
                lp = gl.GLLinePlotItem(
                                pos=np.array([pa.reshape(-1,3),pb.reshape(-1,3)]),
                                color=np.array([human['color'],human['color']]),
                                width=2,
                                antialias=True,
                                mode='lines')
                self.plotter.add_line_plot(lp)



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    # set stylesheet
    file = QFile(":/dark.qss")
    file.open(QFile.ReadOnly | QFile.Text)
    stream = QTextStream(file)
    app.setStyleSheet(stream.readAll())

    form = MainWindow()
    form.show()
    if ACTIVE_CAMERA_MODE:
        app.aboutToQuit.connect(form.camera_thread.close_thread)
    sys.exit(app.exec_())