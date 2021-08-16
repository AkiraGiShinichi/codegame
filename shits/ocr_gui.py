# 131072 x 80 = 10485760
import argparse
import sys
import os
import copy
import time
import datetime
from threading import Thread
import socket

import cv2
import numpy as np
import torch
import random             

from PySide6.QtWidgets import (
    QApplication, QLineEdit, QMainWindow, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QFrame, QWidget, QFileDialog, QCheckBox)
from PySide6.QtGui import QPixmap, QImage, QAction
from PySide6.QtCore import Qt, Signal, QThread
from PySide6 import QtCore


style_sheet = """
    QLabel#ImageLabel{
        color: darkgrey;
        border: 2px solid #000000;
        qproperty-alignment: AlignCenter
    }
"""

DEFAULT_BUFLEN = 256
EXTERNAL_CAMERA = 1
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_PATH = os.path.join(PARENT_DIR, 'best.pt')
# (2592, 1944) (2048, 1536) (1600, 1200) (1920, 1080) (1280, 1024) (1280, 960) (1280, 720) (1024, 768) (640, 480)
RESOLUTION = (640, 480)


class ServerWorkerThread(QThread):
    connection_requested = Signal(socket.socket)
    disconnection_requested = Signal(socket.socket)
    get_detection_requested = Signal(socket.socket)
    get_image_requested = Signal(socket.socket)
    get_chars_requested = Signal(socket.socket)
    invalid_requested = Signal(socket.socket)

    def __init__(self, parent) -> None:
        super().__init__()
        self.parent = parent
        self.params = {}
        self.params['config'] = {}

    def initialize(self, ip, port):
        self.params['config']['ip'] = ip
        self.params['config']['port'] = port
        self.listen_socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        try:
            self.listen_socket.bind(
                (self.params['config']['ip'],
                 self.params['config']['port']))
        except socket.error as msg:
            print(f'\nBind failed. Error code: {msg} - Message ')
        print(f'\nSocket bind complete.')

    def run(self):
        self.listen_socket.listen(0)
        print(f'\nSocket listening...')

        while 1:
            conn, addr = self.listen_socket.accept()
            print(f'Connected {addr[0]}:{addr[1]}')

            while 1:
                try:
                    data = conn.recv(DEFAULT_BUFLEN)
                    req = data.decode()
                    if req == 'connect':
                        print(f'req connect')
                        self.connection_requested.emit(conn)
                    elif req == 'disconnect':
                        print(f'req disconnect')
                        self.disconnection_requested.emit(conn)
                        break
                    elif req == 'get image':
                        print(f'req get image')
                        self.get_image_requested.emit(conn)
                    elif req == 'get chars':
                        print(f'req get chars')
                        self.get_chars_requested.emit(conn)
                    elif req == 'get detection':
                        print(f'req get detection')
                        self.get_detection_requested.emit(conn)
                    else:
                        print(f'req invalid')
                        print(f'Req {req}')
                        res = f"System hasn't support '{req}'."
                        # self.invalid_requested.emit()
                        break
                except:
                    break

    def on_connectionResolved(self, conn):
        res = 'ACK'
        print(
            f'***********************Server responded connect: {res}')
        try:
            conn.send(res.encode())
        finally:
            pass

    def on_disconnectionResolved(self, conn):
        res = 'ACK'
        print(
            f'***********************Server responded disconnect: {res}')
        try:
            conn.send(res.encode())
            conn.close()
        finally:
            pass

    def on_getImageResolved(self, conn, image):
        res = 'ACK'
        print(
            f'***********************Server responded getImage: {res}')
        try:
            conn.send(res.encode())
        finally:
            pass

    def on_getCharsResolved(self, conn, chars):
        res = chars + 'ACK'
        print(
            f'***********************Server responded getChars: {res}')
        try:
            conn.send(res.encode())
        finally:
            pass
        print(f'Completed responding getChars')

    def on_getDetectionResolved(self, conn, result):
        if result['chars'] == '':
            res = 'Error,'
        else:
            confidences = result['boxes'][:, 4]
            labels = result['chars']
            res = [f'{labels[i]}_{confidences[i]:.1f},'
                   for i in range(len(labels))]

            res = ''.join(res)
        print(
            f'***********************Server responded getChars: {res}')
        try:
            conn.send(res.encode())
        finally:
            pass
        print(f'Completed responding getResult')


class VideoWorkerThread(QThread):
    frame_data_updated = Signal(np.ndarray)
    frame_data_invalid = Signal()

    def __init__(
            self, parent, video_file, fps=24, frame_size=(640, 480)) -> None:
        super().__init__()
        self.parent = parent
        self.video_file = video_file
        self.fps = fps
        self.frame_size = frame_size
        self.delay = int(1000 / self.fps)

        self.setup_capture()

        print(f'\n  VideoWorkerThread - initialized')

    def setup_capture(self):
        print(f'\n  VideoWorkerThread - setup_capture')
        if self.video_file == 0:
            pass
        elif self.video_file == 1:
            self.video_capture = cv2.VideoCapture(
                EXTERNAL_CAMERA + cv2.CAP_DSHOW)
            self.video_capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.video_capture.set(cv2.CAP_PROP_FOCUS, 521)
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
            self.video_capture.set(
                cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
            self.video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(
                'M', 'J', 'P', 'G'))
        else:
            self.video_capture = cv2.VideoCapture(self.video_file)

    def initializeRecorder(self, file_path):
        print(f'\n  VideoWorkerThread - initializeRecorder')
        self.video_writer = cv2.VideoWriter(
            file_path, cv2.VideoWriter.fourcc(
                'M', 'J', 'P', 'G'), self.fps, self.frame_size)

    def executeRecording(self):
        if hasattr(self, 'video_writer'):
            if self.video_writer.isOpened():
                self.video_writer.write(self.frame)
            else:
                print(
                    f'\n  VideoWorkerThread - run / video_is_recording: Error - recorder is not initialized yet.')

    def stopRecording(self):
        if hasattr(self, 'video_writer'):
            if self.video_writer.isOpened():
                self.video_writer.release()
                print(
                    f'\n  VideoWorkerThread - run / not video_is_recording: stop recording.')

    def run(self):
        print(f'\n  VideoWorkerThread - run: continuously capture frame and emit result')
        if not self.video_capture.isOpened():
            self.frame_data_invalid.emit()
        else:
            while self.parent.params['state']['video_thread_is_running']:
                if self.parent.params['state']['video_is_pausing']:
                    continue
                else:
                    ret_val, self.frame = self.video_capture.read()  # Read a frame from camera
                    if not ret_val:  # If couldn't get new valid frame
                        print(
                            f'\n  VideoWorkerThread - run: Error or reached the end of the video')
                        self.frame_data_invalid.emit()
                        break
                    else:  # If got new valid frame
                        self.frame_data_updated.emit(self.frame)
                        if self.parent.params['state']['video_is_recording']:
                            self.executeRecording()
                        else:
                            self.stopRecording()
                    cv2.waitKey(self.delay)

    def stopThread(self):
        print(f'\n  VideoWorkerThread - stopThread')
        self.wait()

        self.releaseVideoTools()

        QApplication.processEvents()

    def releaseVideoTools(self):
        print(f'\n  VideoWorkerThread - releaseVideoTools')
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
        if hasattr(self, 'video_capture'):
            self.video_capture.release()


class ImageDisplay(QMainWindow):
    connection_resolved = Signal(socket.socket)
    disconnection_resolved = Signal(socket.socket)
    get_detection_resolved = Signal(socket.socket, dict)
    get_image_resolved = Signal(socket.socket, np.ndarray)
    get_chars_resolved = Signal(socket.socket, str)

    def __init__(self) -> None:
        super().__init__()
        self.verbose = False
        self.params = {}
        self.params['state'] = {}
        self.params['fps'] = 25
        # (2592, 1944) (2048, 1536) (1600, 1200) (1920, 1080) (1280, 1024) (1280, 960) (1280, 720) (1024, 768)
        self.params['frame_size'] = RESOLUTION
        box_size = np.array(
            [1720 * self.params['frame_size'][0] / 1920, 300 * self.params
             ['frame_size'][1] / 1080])
        xs, ys, xe, ye = self.params['frame_size'][0] / 2 - box_size[0] / 2, self.params['frame_size'][1] / 2 - box_size[1] / \
            2, self.params['frame_size'][0] / 2 + box_size[0] / 2, self.params['frame_size'][1] / 2 + box_size[1] / 2  # 1720 300
        self.crop_list = (
            np.array([[xs, ys, xe, ye]])).astype(
            np.int)
        self.detected_results = {}
        self.detected_results['chars'] = ''
        self.detected_results['boxes'] = []
        self.detected_results['image'] = None

        # ------------------------------ New properties ------------------------------ #
        self.recognized = {}
        self.recognized['boxes'] = None
        self.recognized['confidences'] = None
        self.recognized['labels'] = None
        self.recognized['chars'] = ''
        self.recognized['colors'] = None
        self.recognized['image'] = None
        # ---------------------------------------------------------------------------- #

        self.initializeUI()
        Thread(target=self.initializeFunctions).start()
        self.setupServer()
        if self.verbose:
            print(f'\nImageDisplay - initialized')

    # ------------------------------- Initialize UI ------------------------------ #
    def initializeUI(self):
        if self.verbose:
            print(f'\nImageDisplay - initializeUI')
        self.setMinimumSize(800, 600)
        self.setWindowTitle('OCR')

        self.setupWindow()
        self.setupMenu()
        self.show()

    def setupWindow(self):
        if self.verbose:
            print(f'\nImageDisplay - setupWindow')
        # Create image display + button to open an image
        self.image_label = QLabel()
        self.image_label.setObjectName('ImageLabel')

        self.display_file_path_line = QLineEdit()
        self.display_file_path_line.setPlaceholderText(
            'Select video or use webcam')

        open_file_button = QPushButton('&Open Image/Video')
        open_file_button.clicked.connect(self.openAnImageOrAVideoFile)

        self.pause_resume_playing_button = QPushButton('Pause Video')
        self.pause_resume_playing_button.clicked.connect(
            self.pauseOrResumePlayingVideo)
        self.pause_resume_playing_button.setEnabled(False)
        self.params['state']['video_is_pausing'] = False

        self.start_stop_playing_button = QPushButton('Start Video')
        self.start_stop_playing_button.clicked.connect(
            self.startOrStopPlayingVideo)
        self.params['state']['video_thread_is_running'] = False

        self.enable_detection_checkbox = QCheckBox('Enable Detection')
        self.enable_detection_checkbox.stateChanged.connect(
            self.changeDetectionState)
        self.detection_is_enabled = False
        self.enable_detection_checkbox.setEnabled(False)

        self.display_record_directory_line = QLineEdit()
        self.display_record_directory_line.setPlaceholderText(
            'Select saving folder of recorded videos')

        open_saving_folder_button = QPushButton('&Save into Folder')
        open_saving_folder_button.clicked.connect(
            self.openAFolderToSaveRecordedVideo)

        self.start_stop_recording_button = QPushButton('Start &Recording')
        self.start_stop_recording_button.clicked.connect(
            self.startOrStopRecording)
        self.params['state']['video_is_recording'] = False

        save_image_button = QPushButton('Save Image')
        save_image_button.clicked.connect(self.saveImage)

        self.randomly_save_image_checkbox = QCheckBox('Randomly Save Image')
        self.randomly_save_image_checkbox.setChecked(False)

        config_camera_button = QPushButton('Config')
        config_camera_button.clicked.connect(self.configCamera)

        # Layout items
        side_panel_v_box = QVBoxLayout()
        side_panel_v_box.setAlignment(Qt.AlignTop)
        side_panel_v_box.addWidget(self.display_file_path_line)
        side_panel_v_box.addWidget(open_file_button)
        side_panel_v_box.addWidget(self.start_stop_playing_button)
        side_panel_v_box.addWidget(self.enable_detection_checkbox)
        side_panel_v_box.addWidget(self.pause_resume_playing_button)
        side_panel_v_box.addWidget(self.display_record_directory_line)
        side_panel_v_box.addWidget(open_saving_folder_button)
        side_panel_v_box.addWidget(self.start_stop_recording_button)
        side_panel_v_box.addWidget(save_image_button)
        side_panel_v_box.addWidget(self.randomly_save_image_checkbox)
        side_panel_v_box.addWidget(config_camera_button)

        side_panel_frame = QFrame()
        side_panel_frame.setFrameStyle(QFrame.WinPanel)
        side_panel_frame.setMinimumWidth(200)
        side_panel_frame.setLayout(side_panel_v_box)

        main_h_box = QHBoxLayout()
        # ? 1 means auto-stretch full horizontal?
        main_h_box.addWidget(self.image_label, 1)
        main_h_box.addWidget(side_panel_frame)

        # Create container widget and set main window's widget
        container = QWidget()
        container.setLayout(main_h_box)
        self.setCentralWidget(container)

    def setupMenu(self):
        # Create Actions for File menu
        # ! If there isn't self, action can not visible inside menu
        open_file_action = QAction('Open Image/Video', self)
        open_file_action.setShortcut('Ctrl+O')
        open_file_action.triggered.connect(self.openAnImageOrAVideoFile)

        save_directory_action = QAction('Save into..', self)
        save_directory_action.setShortcut('Ctrl+S')
        save_directory_action.triggered.connect(
            self.openAFolderToSaveRecordedVideo)

        record_action = QAction('Start/Stop Record', self)
        record_action.setShortcut('Ctrl+R')
        record_action.triggered.connect(self.startOrStopRecording)

        # Create menu Bar
        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)  # ? What is this?

        # Create File menu and add actions
        file_menu = menu_bar.addMenu('File')
        file_menu.addAction(open_file_action)
        file_menu.addAction(save_directory_action)
        file_menu.addAction(record_action)
    # ---------------------------------------------------------------------------- #

    # --------------------------- Initialize functions --------------------------- #
    def initializeFunctions(self):
        if self.verbose:
            print(f'\nImageDisplay - initializeFunctions')
        self.weight_path = WEIGHT_PATH
        self.setupDetectionModule()

    def setupDetectionModule(self):
        if self.verbose:
            print(f'\nImageDisplay - setupDetectionModule')
        if hasattr(self, 'weight_path'):
            self.detector = torch.hub.load(
                'ultralytics/yolov5', 'custom', path=self.weight_path)
            self.detector = self.detector.autoshape()
            self.detector.conf = 0.6  # confidence threshold (0-1)
            self.detector.iou = 0.8  # NMS IoU threshold (0-1)
        else:
            print(f'\nImageDisplay - setupDetectionModule: Error. Please define path to weight of detection network.')

    def setupServer(self):
        self.server_worker_thread = ServerWorkerThread(parent=self)

        self.server_worker_thread.connection_requested.connect(
            self.resolveConnectionRequest)
        self.connection_resolved.connect(
            self.server_worker_thread.on_connectionResolved)

        self.server_worker_thread.disconnection_requested.connect(
            self.resolveDisconnectionRequest)
        self.disconnection_resolved.connect(
            self.server_worker_thread.on_disconnectionResolved)

        self.server_worker_thread.get_image_requested.connect(
            self.resolveGetImageRequest)
        self.get_image_resolved.connect(
            self.server_worker_thread.on_getImageResolved)

        self.server_worker_thread.get_chars_requested.connect(
            self.resolveGetCharsRequest)
        self.get_chars_resolved.connect(
            self.server_worker_thread.on_getCharsResolved)

        self.server_worker_thread.get_detection_requested.connect(
            self.resolveGetDetectionRequest)
        self.get_detection_resolved.connect(
            self.server_worker_thread.on_getDetectionResolved)

        self.server_worker_thread.initialize('', 6666)
        self.server_worker_thread.start()
    # ---------------------------------------------------------------------------- #

    # -------------------------- Execute client request -------------------------- #
    def resolveConnectionRequest(self, conn):
        if self.verbose:
            print(f'\nImageDisplay - startCameraAndDetection')
        self.startVideoPlaying()
        time.sleep(0.01)
        self.changeDetectionState(state=Qt.Checked)

        self.connection_resolved.emit(conn)

    def resolveDisconnectionRequest(self, conn):
        if self.verbose:
            print(f'\nImageDisplay - stopCameraAndDetection')
        self.changeDetectionState(state=Qt.Unchecked)
        time.sleep(0.01)
        self.stopCurrentVideoPlaying()

        self.disconnection_resolved.emit(conn)

    def resolveGetImageRequest(self, conn):
        if self.verbose:
            print(f'\nImageDisplay - sendClientImage')

        # TODO save image with detected result into path: "C:\QR\Download Server\error.jpg"
        file_path = f'C:/QR/Download Server/error.jpg'
        mat_image_1 = np.copy(self.mat_image)
        xs, ys, xe, ye = self.crop_list[0]
        xs, ys, xe, ye = xs - 20, ys - 10, xe + 20, ye + 10
        mat_image_1 = mat_image_1[ys:ye, xs:xe]
        cv2.imwrite(file_path, mat_image_1)

        self.get_image_resolved.emit(conn, mat_image_1)

    def resolveGetCharsRequest(self, conn):
        if self.verbose:
            print(f'\nImageDisplay - sendClientDetectedChars')

        self.get_chars_resolved.emit(conn, self.detected_results['chars'])

    def resolveGetDetectionRequest(self, conn):
        if self.verbose:
            print(f'\nImageDisplay - resolveGetDetectionRequest')
        file_path = f'C:/QR/Download Server/error.jpg'
        cv2.imwrite(file_path, self.detected_results['image'])
        self.get_detection_resolved.emit(conn, self.detected_results)

    # ---------------------------------------------------------------------------- #

    # ------------------------------ Image functions ----------------------------- #
    def openAnImageOrAVideoFile(self):
        if self.verbose:
            print(f'\nImageDisplay - openFile')
        self.stopCurrentVideoPlaying()

        selected_file_path, selected_filter = QFileDialog.getOpenFileName(
            parent=self, caption='Open Video',
            # dir=os.getenv('HOME'),
            # dir='F:/WORKSPACES/Qr/21054ocr/bin',
            dir=f'{os.getcwd()}',
            filter='Images (*.png *.jpeg *.jpg *.bmp);;Videos (*.mp4 *.avi)')

        if selected_file_path:
            self.display_file_path_line.setText(selected_file_path)
            if selected_filter == 'Images (*.png *.jpeg *.jpg *.bmp)':
                self.mat_image = cv2.imread(selected_file_path)
                self.original_mat_image = copy.deepcopy(self.mat_image)
                self.displayImage()
            elif selected_filter == 'Videos (*.mp4)':
                self.startVideoPlaying()

    def displayImage(self):
        mat_image_1 = np.copy(self.mat_image)
        if hasattr(self, 'crop_list'):
            xs, ys, xe, ye = self.crop_list[0]
            xs, ys, xe, ye = xs - 20, ys - 10, xe + 20, ye + 10
            # cv2.rectangle(mat_image_1, (xs, ys), (xe, ye), (255, 0, 0), 1)
        self.Qt_image = self.convertCVImageIntoQImage(image=mat_image_1)
        if self.verbose:
            print(f'\nImageDisplay - displayImage')
        if hasattr(self, 'Qt_image'):
            self.image_label.setPixmap(
                QPixmap.fromImage(self.Qt_image).scaled(
                    self.image_label.width(),
                    self.image_label.height(),
                    aspectMode=Qt.KeepAspectRatio))
    # ---------------------------------------------------------------------------- #

    # ------------------------------ Video functions ----------------------------- #
    def startOrStopPlayingVideo(self):
        if self.verbose:
            print(f'\nImageDisplay - startOrStopPlayingVideo')
        self.changeDetectionState(state=Qt.Unchecked)
        # video is Stopping -> Start playing video
        if self.params['state']['video_thread_is_running']:
            self.stopCurrentVideoPlaying()
        else:  # video is Playing -> Stop playing video
            self.startVideoPlaying()

    def invalidVideoFrame(self):
        if self.verbose:
            print(f'\nImageDisplay - invalidVideoFrame')
        self.stopCurrentVideoPlaying()

    def closeEvent(self, event):  # ! This function happens when UI is closed
        if self.verbose:
            print(f'\nImageDisplay - closeEvent')
        if self.params['state']['video_thread_is_running']:
            self.video_worker_thread.quit()
            self.params['state']['video_thread_is_running'] = False

    def startVideoPlaying(self):
        if self.verbose:
            print(f'\nImageDisplay - startVideoPlaying')
        if not self.params['state']['video_thread_is_running']:
            display_file_path = self.display_file_path_line.text()
            _, extension = os.path.splitext(display_file_path)

            if display_file_path == '':
                self.video_worker_thread = VideoWorkerThread(
                    parent=self, video_file=1, fps=self.params['fps'],
                    frame_size=self.params['frame_size'])
            elif display_file_path != '' and extension in set(['.mp4', '.avi']):
                video_file = display_file_path
                self.video_worker_thread = VideoWorkerThread(
                    parent=self, video_file=video_file)
            else:  # ! If the file path is image or other file types, do NOT Execute
                print(
                    f'Can NOT start playing video, file type is not a video. Please select 1 video.')
                return

            self.params['state']['video_thread_is_running'] = True

            self.video_worker_thread.frame_data_updated.connect(
                self.updateVideoFrame)
            self.video_worker_thread.frame_data_invalid.connect(
                self.invalidVideoFrame)
            self.video_worker_thread.start()

            self.start_stop_playing_button.setText('Stop Video')

            self.pause_resume_playing_button.setEnabled(True)

            self.start_stop_recording_button.setEnabled(True)

            self.enable_detection_checkbox.setEnabled(True)

    def stopCurrentVideoPlaying(self):
        if self.verbose:
            print(f'\nImageDisplay - stopCurrentVideoPlaying')
        if self.params['state']['video_thread_is_running']:
            self.params['state']['video_thread_is_running'] = False
            self.video_worker_thread.stopThread()

            self.start_stop_playing_button.setText('Start Video')

            self.pause_resume_playing_button.setEnabled(False)
            self.resumeCurrentVideo()  # reset to pause-able state

            self.start_stop_recording_button.setEnabled(False)
            self.stopCurrentRecording()  # reset to start-able recording process

            self.enable_detection_checkbox.setEnabled(False)

    def updateVideoFrame(self, frame):
        if self.verbose:
            print(f'\nImageDisplay - updateVideoFrame')
        self.mat_image = frame
        self.original_mat_image = copy.deepcopy(self.mat_image)

        if self.detection_is_enabled:
            self.detectAndLabelCharacters()

        if self.randomly_save_image_checkbox.isChecked():
            self.randomlySaveImage(ratio=1)

        # self.draw_crop_list(self.mat_image)

        self.displayImage()
    # ---------------------------------------------------------------------------- #

    # --------------------------------- Detection -------------------------------- #
    def changeDetectionState(self, state):
        if state == Qt.Checked:
            self.detection_is_enabled = True
            self.enable_detection_checkbox.setChecked(True)
            if self.image_label.pixmap() != None:
                self.detectAndLabelCharacters()
        else:
            self.detection_is_enabled = False
            self.enable_detection_checkbox.setChecked(False)

    def detectAndLabelCharacters(self):
        xs, ys, xe, ye = self.crop_list[0]
        roi = self.mat_image[ys:ye, xs:xe]

        results = self.detector(roi, size=320)
        results_sorted = results.pandas().xyxy[0].sort_values('xmin')
        result_labels = results_sorted['name'].to_numpy()
        result_label_indices = results_sorted['class'].to_numpy()
        result_chars = ''.join(result_labels)
        result_boxes = results_sorted[[
            'xmin', 'ymin', 'xmax', 'ymax']].to_numpy()
        result_confidences = results_sorted['confidence'].to_numpy()

        if self.recognized['colors'] == None:
            self.recognized['colors'] = [
                [random.randint(0, 255) for _ in range(3)]
                for _ in range(len(results.names))]
        boxes = results_sorted[['xmin', 'ymin', 'xmax',
                                'ymax', 'confidence', 'class']].to_numpy()
        self.detected_results['boxes'] = boxes

        boxes[:, [0, 2]] += xs
        boxes[:, [1, 3]] += ys
        # Adjust coordinates of boxes to absolute values
        result_boxes[:, [0, 2]] += xs
        result_boxes[:, [1, 3]] += ys

        detected_chars = result_chars
        self.detected_results['chars'] = detected_chars  # TODO

        img = self.mat_image.copy()
        cv2.putText(
            img, text=detected_chars, org=(10, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
            color=(255, 255, 255),
            thickness=2)

        self.draw_cropping_area(img, self.crop_list[0])

        for i in range(len(result_labels)):
            self.plot_one_box(
                result_boxes[i, :4],
                img,
                label=f'{result_labels[i]}',
                confidence=result_confidences[i],
                color=self.recognized['colors'][result_label_indices[i]],
                line_thickness=1)

        self.mat_image = img  # TODO
        self.detected_results['image'] = img[ys:ye, xs:xe]  # TODO

        # Backup recognized result
        self.recognized['boxes'] = result_boxes
        self.recognized['confidences'] = result_confidences
        self.recognized['labels'] = result_labels
        self.recognized['chars'] = result_chars
        self.recognized['image'] = img

    @staticmethod
    def draw_cropping_area(image, crop_list):
        cv2.rectangle(image, (crop_list[0], crop_list[1]),
                      (crop_list[2], crop_list[3]), (255, 0, 0), 2)

    @staticmethod
    def plot_one_box(
            x, img, color=None, label=None, confidence=None,
            line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(
            0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        if confidence:
            t_size = cv2.getTextSize(
                f'{confidence:.2f}', 0, fontScale=tl / 3, thickness=tf)[0]
            c3 = c1[0] + t_size[0], c2[1] + t_size[1] + 3
            cv2.rectangle(img, (c1[0], c2[1]), c3,
                          color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                img, f'{confidence:.2f}', (c1[0] + 1,
                                           c2[1] + t_size[1] + 1),
                0, tl / 3, [0, 0, 0],
                thickness=tf, lineType=cv2.LINE_AA)
        if label:
            t_size = cv2.getTextSize(
                label, 0, fontScale=tl / 3, thickness=tf)[0]
            c4 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c4, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                img, label, (c1[0],
                             c1[1] - 2),
                0, tl / 3, [0, 0, 0],
                thickness=tf, lineType=cv2.LINE_AA)
    # ---------------------------------------------------------------------------- #

    @staticmethod
    def convertCVImageIntoQImage(image, verbose=False):
        if verbose:
            print(f'\nImageDisplay - convertCVImageIntoQImage')
        # Convert OpenCV BGR image to RGB image
        bgr_image = image  # Defaultly, OpenCV load image as BGR image
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # Convert Mat(numpy)-Image into QImage
        # # Get shape of image
        height, width, channels = rgb_image.shape
        # # Get number of bytes required by image pixels in a row
        bytes_per_line = width * channels
        # # Perform converting
        Qt_image = QImage(rgb_image, width, height,
                          bytes_per_line, QImage.Format_RGB888)
        return Qt_image

    @staticmethod
    def create_folder(path):
        if not os.path.exists(path):
            os.makedirs(path)

    # ------------------------ Pause/Resume playing video ------------------------ #
    def pauseOrResumePlayingVideo(self):
        if self.verbose:
            print(f'\nImageDisplay - pauseOrResumePlayingVideo')
        if self.params['state']['video_thread_is_running']:
            if self.params['state']['video_is_pausing']:
                self.resumeCurrentVideo()
            else:
                self.pauseCurrentVideo()
        else:
            print(
                f'Can NOT pause/resume video, there is no video that is playing yet. Please open & play a video.')

    def pauseCurrentVideo(self):
        if self.verbose:
            print(f'\nImageDisplay - pauseCurrentVideo')
        self.params['state']['video_is_pausing'] = True
        self.pause_resume_playing_button.setText('Resume Video')

    def resumeCurrentVideo(self):
        if self.verbose:
            print(f'\nImageDisplay - resumeCurrentVideo')
        self.params['state']['video_is_pausing'] = False
        self.pause_resume_playing_button.setText('Pause Video')
    # ---------------------------------------------------------------------------- #

    # ---------------------------------- Record ---------------------------------- #
    def openAFolderToSaveRecordedVideo(self):
        if self.verbose:
            print(f'\nImageDisplay - openAFolderToSaveRecordedVideo')

        folder_path = QFileDialog.getExistingDirectory(
            self, caption='Select a Folder', dir=f'{os.getcwd()}')
        self.display_record_directory_line.setText(folder_path)

    def startOrStopRecording(self):
        if self.verbose:
            print(f'\nImageDisplay - startOrStopRecording')
        if self.params['state']['video_thread_is_running']:
            # If it is recording, stop it.
            if self.params['state']['video_is_recording']:
                self.stopCurrentRecording()
            else:  # If it hasn't recorded yet, start recording
                if self.display_record_directory_line.text() == '':  # If there isn't saving folder yet
                    print(
                        f'Can NOT start/stop recording video. Please select saving folder')
                else:
                    self.startRecording()
        else:
            print(
                f'Can NOT start/stop recording video, there is no video that is playing yet. Please open & play a video.')

    def startRecording(self):
        if self.verbose:
            print(f'\nImageDisplay - startRecording')
        directory = self.display_record_directory_line.text()
        self.create_folder(path=directory)
        current_time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = f'{directory}/{current_time_str}.avi'

        self.video_worker_thread.initializeRecorder(file_path=file_path)
        self.params['state']['video_is_recording'] = True
        self.start_stop_recording_button.setText('Stop Recording')

    def stopCurrentRecording(self):
        if self.verbose:
            print(f'\nImageDisplay - stopCurrentRecording')
        self.params['state']['video_is_recording'] = False
        self.start_stop_recording_button.setText('Start Recording')
    # ---------------------------------------------------------------------------- #

    # ------------------ Save Image for Creating trainning data ------------------ #
    def saveImage(self):
        if hasattr(self, 'original_mat_image'):
            current_time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            file_path = f'{self.display_record_directory_line.text()}/{current_time_str}.jpg'
            # cv2.imwrite(file_path, self.mat_image)
            mat_image_1 = np.copy(self.original_mat_image)
            xs, ys, xe, ye = self.crop_list[0]
            # xs, ys, xe, ye = xs - 20, ys - 10, xe + 20, ye + 10
            mat_image_1 = mat_image_1[ys:ye, xs:xe]
            cv2.imwrite(file_path, mat_image_1)

    def randomlySaveImage(self, ratio=0.1):
        if np.random.rand() < ratio:
            self.saveImage()
    # ---------------------------------------------------------------------------- #

    def configCamera(self):
        self.video_worker_thread.video_capture.set(cv2.CAP_PROP_SETTINGS, 1)
        pass


# ----------------------------------- Main ----------------------------------- #
def _main_(args):
    print(f'This is {args.desc}')
    app = QApplication(sys.argv)
    app.setStyleSheet(style_sheet)
    window = ImageDisplay()
    sys.exit(app.exec())
# ---------------------------------------------------------------------------- #


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='OCR')
    argparser.add_argument(
        '-d', '--desc', help='Description', default='Quadrep OCR')
    args = argparser.parse_args()

    _main_(args)
