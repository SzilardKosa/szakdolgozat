import cv2
import time
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import math
from pose_engine import PoseEngine

RED   = (255,0,0)
GREEN = (0,255,0)
BLUE  = (0,0,255)
YELLOW = (255,255,0)
CYAN   = (0,255,255)
PINK   = (255,0,255)
WHITE  = (255,255,255)
HUMAN_COLORS_2D = [GREEN, BLUE, CYAN, PINK, WHITE]
RED_3D   =  (1.0,0.0,0.0,1.0)
GREEN_3D =  (0.0,1.0,0.0,1.0)
BLUE_3D  =  (0.0,0.0,1.0,1.0)
YELLOW_3D = (1.0,1.0,0.0,1.0)
CYAN_3D   = (0.0,1.0,1.0,1.0)
PINK_3D   = (1.0,0.0,1.0,1.0)
WHITE_3D   = (1.0,1.0,1.0,1.0)
HUMAN_COLORS_3D = [GREEN_3D, BLUE_3D, CYAN_3D, PINK_3D, WHITE_3D]


class Camera:
    def __init__(self, cam_num=0):
        self.cam_num = cam_num
        self.cap = None
        self.last_frame = None

        # Intrinsic parameters
        self.camera_matrix = np.zeros((3, 3))
        self.dist_coeffs = np.zeros((1, 5))
        self.error_intrinsic = 0
        self.new_camera_matrix = np.zeros((3, 3))
        self.undist_roi = None
        self.alpha = 1

        # Extrinsic parameters
        self.R = np.zeros((3, 3)) # rotation matrix
        self.t = np.zeros((3, 1)) # translation vect
        self.M = np.zeros((3, 4)) # extrinsic parameter matrix [R|t]
        self.P = np.zeros((3, 4)) # projection matrix         C[R|t]
        self.error_extrinsic = 0


    def set_intrinsic(self, mtx, dist, err, newcameramtx, undist_roi, alpha):
        self.camera_matrix = mtx
        self.dist_coeffs = dist
        self.error_intrinsic = err
        self.new_camera_matrix = newcameramtx
        self.undist_roi = undist_roi
        self.alpha = alpha
    
    def start(self):
        self.cap = cv2.VideoCapture(self.cam_num)

    def set_resolution(self, width, height):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def get_resolution(self):
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return (width, height)

    def set_fps(self, fps):
        self.cap.set(cv2.CAP_PROP_FPS, fps)
    
    def get_fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def stop(self):
        self.cap.release()

    def get_frame(self):
        _, self.frame = self.cap.read()
        self.last_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        return self.last_frame

    def __str__(self):
        return f'Camera {self.cam_num}'

class Plotter:
    def __init__(self):
        self.w = None
        self.scatter_plots = []
        self.line_plots = []
        self.light_plot = None
        self.axises = []
        self.cameras = []

    def add_plot(self, layout):
        self.w = gl.GLViewWidget()
        self.w.setCameraPosition(distance=10)
        layout.addWidget(self.w, 1, 1)
    
    def add_grid(self):
        grid3d = gl.GLGridItem()
        grid3d.setSize(10, 10)
        grid3d.setDepthValue(10)  # draw grid after surfaces since they may be translucent
        self.w.addItem(grid3d)
    
    def add_scatter_plot(self, sp):
        self.scatter_plots.append(sp)
        self.w.addItem(sp)
    
    def remove_scatter_plot(self):
        if 0 < len(self.scatter_plots):
            self.w.removeItem(self.scatter_plots[-1])
            del self.scatter_plots[-1]
    
    def remove_all_scatter_plot(self):
        if 0 < len(self.scatter_plots):
            for i in range(len(self.scatter_plots)):
                self.w.removeItem(self.scatter_plots[-1])
                del self.scatter_plots[-1]

    def add_line_plot(self, lp):
        self.line_plots.append(lp)
        self.w.addItem(lp)
    
    def remove_line_plot(self):
        if 0 < len(self.line_plots):
            self.w.removeItem(self.line_plots[-1])
            del self.line_plots[-1]
    
    def remove_all_line_plot(self):
        if 0 < len(self.line_plots):
            for i in range(len(self.line_plots)):
                self.w.removeItem(self.line_plots[-1])
                del self.line_plots[-1]
    
    def add_light_plot(self, sp):
        if not self.light_plot:
            self.light_plot = sp
            self.w.addItem(sp)
        else:
            self.light_plot.setData(pos=np.concatenate((self.light_plot.pos, sp.pos), axis=0),
                                    size=np.concatenate((self.light_plot.size, sp.size), axis=0),
                                    color=np.concatenate((self.light_plot.color, sp.color), axis=0))
    def remove_light_plot(self):
        if self.light_plot:
            self.w.removeItem(self.light_plot)
            self.light_plot = None
    
    def transform_points(self, points, rot=None, trans=None, scale=1):
        if rot is None:
            rot = np.array([(1, 0, 0),
                            (0, 1, 0),
                            (0, 0, 1)])
        if trans is None:
            trans = np.array([(0, 0, 0)])
        scale_mat = np.eye(3, dtype=float)*scale

        return np.dot(scale_mat, np.dot(rot, points.T)).T + trans
    
    def add_axis(self, rot=None, trans=None, scale=1):        
        default_pos = np.array([(0, 0, 0),(1, 0, 0),
                                 (0, 0, 0),(0, 1, 0),
                                 (0, 0, 0),(0, 0, 1)])
        new_pos = self.transform_points(default_pos, rot, trans, scale)
        axis = gl.GLLinePlotItem(
                                pos=new_pos,
                                color=np.array([(1.0, 0.0, 0.0),(1.0, 0.0, 0.0),
                                                (0.0, 1.0, 0.0),(0.0, 1.0, 0.0),
                                                (0.0, 0.0, 1.0),(0.0, 0.0, 1.0)]),
                                width=10*scale,
                                antialias=True,
                                mode='lines')
        self.axises.append(axis)
        self.w.addItem(axis)

    def remove_axis(self):
        if 0 < len(self.axises):
            self.w.removeItem(self.axises[-1])
            del self.axises[-1]
    
    def remove_all_axis(self):
        if 0 < len(self.axises):
            for i in range(len(self.axises)):
                self.w.removeItem(self.axises[-1])
                del self.axises[-1]

    def add_camera(self, rot=None, trans=None, scale=1):
        depth = scale
        width = depth*2*math.tan(math.radians(75/2))*4/5
        height = width*3/4
        default_pos = np.array([( width/2,  height/2, depth),( width/2, -height/2, depth),
                                ( width/2, -height/2, depth),(-width/2, -height/2, depth),
                                (-width/2, -height/2, depth),(-width/2,  height/2, depth),
                                (-width/2,  height/2, depth),( width/2,  height/2, depth),
                                ( width/2,  height/2, depth),( 0, 0, 0),
                                ( width/2, -height/2, depth),( 0, 0, 0),
                                (-width/2, -height/2, depth),( 0, 0, 0),
                                (-width/2,  height/2, depth),( 0, 0, 0)])
        new_pos = self.transform_points(default_pos, rot, trans)
        lp = gl.GLLinePlotItem(
                                pos=new_pos,
                                color=np.array([(1.0, 1.0, 0.0),(1.0, 1.0, 0.0),
                                                (1.0, 1.0, 0.0),(1.0, 1.0, 0.0),
                                                (1.0, 1.0, 0.0),(1.0, 1.0, 0.0),
                                                (1.0, 1.0, 0.0),(1.0, 1.0, 0.0),
                                                (1.0, 1.0, 0.0),(1.0, 1.0, 0.0),
                                                (1.0, 1.0, 0.0),(1.0, 1.0, 0.0),
                                                (1.0, 1.0, 0.0),(1.0, 1.0, 0.0),
                                                (1.0, 1.0, 0.0),(1.0, 1.0, 0.0)]),
                                width=10*depth,
                                antialias=True,
                                mode='lines')
        self.cameras.append(lp)
        self.w.addItem(lp)

    def remove_camera(self):
        if 0 < len(self.cameras):
            self.w.removeItem(self.cameras[-1])
            del self.cameras[-1]
    
    def remove_all_camera(self):
        if 0 < len(self.cameras):
            for i in range(len(self.cameras)):
                self.w.removeItem(self.cameras[-1])
                del self.cameras[-1]
    
    def clear_all(self):
        self.remove_all_axis()
        self.remove_all_camera()
        self.remove_all_scatter_plot()
        self.remove_light_plot()

EDGES = (
    ('nose', 'left eye'),
    ('nose', 'right eye'),
    ('nose', 'left ear'),
    ('nose', 'right ear'),
    ('left ear', 'left eye'),
    ('right ear', 'right eye'),
    ('left eye', 'right eye'),
    ('left shoulder', 'right shoulder'),
    ('left shoulder', 'left elbow'),
    ('left shoulder', 'left hip'),
    ('right shoulder', 'right elbow'),
    ('right shoulder', 'right hip'),
    ('left elbow', 'left wrist'),
    ('right elbow', 'right wrist'),
    ('left hip', 'right hip'),
    ('left hip', 'left knee'),
    ('right hip', 'right knee'),
    ('left knee', 'left ankle'),
    ('right knee', 'right ankle'),
)

class MocapThread(QtCore.QThread):

    def __init__(self, cameras):
        super().__init__()

        self.cameras = cameras
        self.frames = []

        self.last_frames = []
        # engine = PoseEngine('models/posenet_mobilenet_v1_075_353_481_quant_decoder_edgetpu.tflite')
        self.engine = PoseEngine('models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')
        # engine = PoseEngine('models/posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite')

        self.stop = False

        # real fps measurement variables
        self.time_start = time.time()
        self.time_stop = time.time()
        self.real_fps = 0

    def run(self):
        while not self.stop:

            self.time_start = time.time()

            for cam in self.cameras:
                self.frames = cam.last_frame

            for i, frame in enumerate(self.frames):
                # image = np.uint8(cv2.resize(frame, (481, 353)))
                image = np.uint8(cv2.resize(frame, (641, 481)))
                # frame = np.uint8(cv2.resize(frame, (1281, 721)))

                poses, inference_time = engine.DetectPosesInImage(frame)
                # print('Inference time: %.fms' % inference_time)

                for pose in poses:
                    draw_pose(frame, pose)

                self.last_frames[i] = frame


            self.time_stop = time.time()
            self.real_fps = self.real_fps*0.9 + (1/(self.time_stop-self.time_start))*0.1

    def draw_pose(self, frame, pose, threshold=0.2):
        xys = {}
        for label, keypoint in pose.keypoints.items():
            if keypoint.score < threshold: continue
            xys[label] = (int(keypoint.yx[1]), int(keypoint.yx[0]))
            cv2.circle(frame, (int(keypoint.yx[1]), int(keypoint.yx[0])), 5,
                        (255,255,255), 2)
        for a, b in EDGES:
            if a not in xys or b not in xys: continue
            ax, ay = xys[a]
            bx, by = xys[b]
            cv2.line(frame, (ax, ay), (bx, by), (255,255,0), 2)


def matrix_template(matrix):
    shape = matrix.shape

    matrix_richtext = ""+\
    "<html>"+\
        "<head>"+\
            "<style>"+\
            "td {text-align:right;}"+\
            "</style>"+\
        "</head>"+\
        "<body>"+\
            "<table align=\"center\">"
    
    for i in range(shape[0]):
        matrix_richtext += "<tr>"
        for j in range(shape[1]):
            matrix_richtext += "<td>{0:.3f}&nbsp;</td>".format(matrix[i,j])
        matrix_richtext += "</tr>"

    matrix_richtext += ""+\
            "</table>"+\
        "</body>"+\
    "</html>"\
    
    return matrix_richtext

if __name__ == '__main__':
    cam = Camera()
    print(cam)
    cam.start()
    frame = cam.get_frame()
    cv2.imshow('frame',frame)
    cv2.waitKey(0)
    cam.stop()
    cv2.destroyAllWindows()

