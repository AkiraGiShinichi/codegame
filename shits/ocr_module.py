import os
import sys
import argparse
import time
import copy
import random

import numpy as np
import cv2

import socket

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import (QObject, QRunnable, QThread,
                            QThreadPool, Signal)

# from video_worker_thread import VideoWorkerThread
# from server_worker_thread import ServerWorkerThread
import torch


DEFAULT_BUFLEN = 256
EXTERNAL_CAMERA = 0 + cv2.CAP_DSHOW
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_PATH = os.path.join(PARENT_DIR, 'best.pt')


class ServerWorkerThread(QThread):
    connection_requested = Signal(socket.socket)
    disconnection_requested = Signal(socket.socket)
    get_detection_requested = Signal(socket.socket, float)
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
                    print(f'req {req}')
                    if req == 'connect':
                        print(f'req connect')
                        self.connection_requested.emit(conn)
                    elif req == 'disconnect':
                        print(f'req disconnect')
                        self.disconnection_requested.emit(conn)
                        break
                    elif 'get detection' in req:
                        smt = req.split(',')
                        conf = float(smt[-1])
                        conf = min(1, max(0, conf))
                        print(f'req get detection {conf}')
                        self.get_detection_requested.emit(conn, conf)
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

    def on_getDetectionResolved(self, conn, result):
        if result['chars'] == '':
            res = 'Error,'
        else:
            confidences = result['confidences']
            labels = result['labels']
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
    frame_data_invalid = Signal(str)
    is_running = False

    def __init__(self, fps=25, frame_size=(640, 480)) -> None:
        """Initialize video worker

        :param fps: frame per second, defaults to 25
        :type fps: int, optional
        :param frame_size: (width, height) of frame, defaults to (640, 480)
        :type frame_size: tuple(int, int), optional
        """
        super().__init__()
        print(f'\n  VideoWorkerThread - initializing..')
        self.fps = fps
        self.frame_size = frame_size
        self.delay = int(1000 / self.fps)

        self.initialize_capture()

    def initialize_capture(self):
        """Initialize video capture
        """
        print(f'\n  VideoWorkerThread - initialize_capture')
        self.video_capture = cv2.VideoCapture(EXTERNAL_CAMERA)
        self.video_capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
        self.video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(
            'M', 'J', 'P', 'G'))

    def capture_frame(self):
        """Capture a frame
        """
        if self.is_running:
            if self.video_capture.isOpened():
                ret_val, self.frame = self.video_capture.read()
                if ret_val:
                    self.frame_data_updated.emit(self.frame)
                    return True
                else:
                    self.frame_data_invalid.emit('Can not capture frame.')
            else:
                self.frame_data_invalid.emit('Video capture is not opened yet.')
        else:
            self.frame_data_invalid.emit('Worker is not running yet.')
        return False

    def run(self):
        """Run workers
        """
        print(f'\n  VideoWorkerThread - run')
        self.is_running = True
        while self.is_running:
            ret_val = self.capture_frame()
            if not ret_val:
                self.is_running = False
                print('\n  VideoWorkerThread - run: Error occured. Worker stop running.')
                break
            cv2.waitKey(self.delay)

    def stop_thread(self):
        """Stop worker running & worker thread
        """
        print(f'\n  VideoWorkerThread - stop_thread')

        self.is_running = False
        self.wait()
        self.video_capture.release()

        QApplication.processEvents()


class Signals(QObject):
    connection_resolved = Signal(socket.socket)
    disconnection_resolved = Signal(socket.socket)
    get_detection_resolved = Signal(socket.socket, dict)


class OcrRunner(QRunnable):
    server_worker_thread = None
    video_worker_thread = None
    detector = None

    def __init__(self) -> None:
        super().__init__()
        self.verbose = False

        self.configs = {}
        self.configs['fps'] = 2
        # (2592, 1944) (2048, 1536) (1600, 1200) (1920, 1080) (1280, 1024) (1280, 960) (1280, 720)
        self.configs['frame_size'] = (1920, 1080)
        self.configs['weight_path'] = WEIGHT_PATH

        self.recognition = {}
        self.recognition['crop_list'] = self.calculate_crop_area(
            self.configs['frame_size'])
        self.recognition['is_enabled'] = False
        self.recognition['results'] = {}
        self.recognition['colors'] = None

        self.setup_tcp_server()
        self.server_worker_thread.start()
        self.start_device()

    @staticmethod
    def calculate_crop_area(frame_size):
        """Calculate crop area when know the size of frame

        :param frame_size: size of frame
        :type frame_size: list(int, int)
        :return: x_start, y_start, x_end, y_end
        :rtype: list(int, int, int, int) 
        """
        # (1720, 300) * ratio
        width, height = frame_size
        box_size = np.array(
            [1720 * width / 1920, 300 * height / 1080])
        x_start = width / 2 - box_size[0] / 2
        y_start = height / 2 - box_size[1] / 2
        x_end = width / 2 + box_size[0] / 2
        y_end = height / 2 + box_size[1] / 2
        return int(x_start), int(y_start), int(x_end), int(y_end)

    def run(self):
        count = 0
        app = QApplication.instance()
        while 1:
            count = (count + 1) % 10
            print(f'\n {count}')
            time.sleep(1)
        app.quit()

    def start_device(self):
        """Start video worker & detector
        """
        if self.video_worker_thread is None:
            self.setup_video_worker()
            self.video_worker_thread.start()
        if self.detector is None:
            self.setup_detection_module()
            self.recognition['is_enabled'] = True

    def stop_device(self):
        """Stop detection & video worker & display
        """
        self.recognition['is_enabled'] = False
        self.detector = None

        self.video_worker_thread.stop_thread()
        self.video_worker_thread = None

        cv2.destroyAllWindows()

    # ------------------------------- Video worker ------------------------------- #
    def setup_video_worker(self):
        """Setup video worker
        """
        self.video_worker_thread = VideoWorkerThread(
            fps=self.configs['fps'],
            frame_size=self.configs['frame_size'])

        self.video_worker_thread.frame_data_updated.connect(
            self.update_video_frame)
        self.video_worker_thread.frame_data_invalid.connect(
            self.stop_device)

    def update_video_frame(self, frame):
        """Update video frame

        :param frame: new coming frame
        :type frame: ndarray
        """
        self.mat_image = frame
        self.original_mat_image = copy.deepcopy(self.mat_image)

        if self.recognition['is_enabled']:
            self.detect_and_label_characters()

        self.display_image()

    def display_image(self):
        if self.verbose:
            print(f'\n OCR - display_image')
        if cv2.getWindowProperty('OCR', 0) < 0:
            cv2.namedWindow('OCR', cv2.WINDOW_NORMAL)
            # cv2.namedWindow('OCR', cv2.WND_PROP_FULLSCREEN)
            # cv2.setWindowProperty(
            #     'OCR', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('OCR', self.mat_image)
        # Catch key input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.stop_device()
    # ---------------------------------------------------------------------------- #

    # -------------------------------- TCP server -------------------------------- #
    def setup_tcp_server(self):
        self.signals = Signals()
        self.server_worker_thread = ServerWorkerThread(parent=self)

        self.server_worker_thread.connection_requested.connect(
            self.resolveConnectionRequest)
        self.signals.connection_resolved.connect(
            self.server_worker_thread.on_connectionResolved)

        self.server_worker_thread.disconnection_requested.connect(
            self.resolveDisconnectionRequest)
        self.signals.disconnection_resolved.connect(
            self.server_worker_thread.on_disconnectionResolved)

        self.server_worker_thread.get_detection_requested.connect(
            self.resolveGetDetectionRequest)
        self.signals.get_detection_resolved.connect(
            self.server_worker_thread.on_getDetectionResolved)

        self.server_worker_thread.initialize('', 6666)

    def resolveConnectionRequest(self, conn):
        print(f'\nImageDisplay - resolveConnectionRequest')
        self.start_device()
        time.sleep(0.01)

        self.signals.connection_resolved.emit(conn)

    def resolveDisconnectionRequest(self, conn):
        print(f'\nImageDisplay - resolveDisconnectionRequest')
        time.sleep(0.01)
        self.stop_device()

        self.signals.disconnection_resolved.emit(conn)

    def push_error_image_to_server(self):
        """Save image into place of server, where client can download
        """
        file_path = f'C:/QR/Download Server/error.jpg'
        cv2.imwrite(file_path, self.recognition['results']['image'])

    def update_model_confidence_threshold(self, conf):
        if self.detector.conf != conf:
            self.detector.conf = conf

    def resolveGetDetectionRequest(self, conn, conf):
        print(f'\nImageDisplay - resolveGetDetectionRequest')
        # self.detector.conf != conf
        # update model conf-threshold if neccessary
        self.update_model_confidence_threshold(conf)
        # file_path = f'C:/QR/Download Server/error.jpg'
        # cv2.imwrite(file_path, self.recognition['results']['image'])
        self.push_error_image_to_server()  # push image to server for client to download
        self.signals.get_detection_resolved.emit(
            conn, self.recognition['results'])  # trigger server worker
    # ---------------------------------------------------------------------------- #

    # -------------------------- Character recognization ------------------------- #
    def setup_detection_module(self):
        """Setup detection module
        """
        self.detector = torch.hub.load(
            'ultralytics/yolov5', 'custom', path=self.configs
            ['weight_path'])
        self.detector = self.detector.autoshape()  # for PIL/cv2/np inputs and NMS
        self.detector.conf = 0.6
        self.detector.iou = 0.8

    def create_color_codes_from_names(self, names):
        """Create color codes to label recognized characters

        :param names: labels
        :type names: list(string)
        """
        if self.recognition['colors'] is None:
            self.recognition['colors'] = [
                [random.randint(0, 255) for _ in range(3)]
                for _ in range(len(names))]

    def extract_inferred_result(self, results):
        """Extract bounding boxes, confidence value, labels,... from model output

        :param results: output from torch hub
        :type results: [type]
        :return: result_boxes, result_confidences, result_labels, result_label_indices, result_chars
        :rtype: list(ndarray)
        """
        results_sorted = results.pandas().xyxy[0].sort_values('xmin')
        results_sorted = results_sorted[results_sorted.name != 'line']
        result_labels = results_sorted['name'].to_numpy()
        result_label_indices = results_sorted['class'].to_numpy()
        result_chars = ''.join(result_labels)
        result_boxes = results_sorted[[
            'xmin', 'ymin', 'xmax', 'ymax']].to_numpy()
        result_confidences = results_sorted['confidence'].to_numpy()
        return result_boxes, result_confidences, result_labels, result_label_indices, result_chars

    def detect_and_label_characters(self):
        """Detect and label characters
        """
        self.recognition['results'] = {}
        # Crop the detecting-area
        x_start, y_start, x_end, y_end = self.recognition['crop_list']
        roi = self.mat_image[y_start:y_end, x_start:x_end]

        # Detect & Extract result attributes
        results = self.detector(roi, size=640)

        boxes, confidences, labels, label_indices, chars = self.extract_inferred_result(
            results)

        # Adjust coordinates of boxes to absolute values
        boxes[:, [0, 2]] += x_start
        boxes[:, [1, 3]] += y_start

        self.create_color_codes_from_names(results.names)
        # Mark results on image
        image = self.mat_image.copy()
        cv2.putText(
            image, text=chars, org=(10, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
            color=(0, 0, 255),
            thickness=1)
        self.draw_region_of_interest(image, self.recognition['crop_list'])
        for i in range(len(labels)):
            self.plot_one_box(
                boxes[i, :4],
                image,
                label=f'{labels[i]}',
                confidence=confidences[i],
                color=self.recognition['colors'][label_indices[i]],
                line_thickness=1)
        self.mat_image = image  # Update image after marking

        self.backup_recognized_result(boxes, confidences, labels, chars, image)

    def backup_recognized_result(
            self, boxes, confidences, labels, chars, image):
        """Backup recognized results

        :param boxes: recognized bounding boxes
        :type boxes: ndarray 2d
        :param confidences: confidences of recognized bounding boxes
        :type confidences: ndarray
        :param labels: labels of recognized bounding boxes
        :type labels: ndarray
        :param chars: all characters(all labels)
        :type chars: string
        :param image: image of result
        :type image: ndarray 3d
        """
        self.recognition['results']['boxes'] = boxes
        self.recognition['results']['confidences'] = confidences
        self.recognition['results']['labels'] = labels
        self.recognition['results']['chars'] = chars
        # self.recognition['results']['image'] = image[y_start:y_end, x_start:x_end]
        self.recognition['results']['image'] = image

    @staticmethod
    def draw_region_of_interest(image, crop_list):
        """Draw working area of OCR

        :param image: target image
        :type image: ndarray
        :param crop_list: (x_start, y_start, x_end, y_end) is top-left & bottom-right pixels of working area
        :type crop_list: tuple(int, int, int, int)
        """
        cv2.rectangle(image, (crop_list[0], crop_list[1]),
                      (crop_list[2], crop_list[3]), (255, 0, 0), 1)

    @staticmethod
    def plot_one_box(
            box, image, color=None, label=None, confidence=None,
            line_thickness=None):
        # Plots one bounding box on image image
        tl = line_thickness or round(
            0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        if confidence:
            t_size = cv2.getTextSize(
                f'{confidence:.2f}', 0, fontScale=tl / 3, thickness=tf)[0]
            c3 = c1[0] + t_size[0], c2[1] + t_size[1] + 3
            cv2.rectangle(image, (c1[0], c2[1]), c3,
                          color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                image, f'{confidence:.2f}', (c1[0] + 1,
                                             c2[1] + t_size[1] + 1),
                0, tl / 3, [0, 0, 0],
                thickness=tf, lineType=cv2.LINE_AA)
        if label:
            t_size = cv2.getTextSize(
                label, 0, fontScale=tl / 3, thickness=tf)[0]
            c4 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c4, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                image, label, (c1[0],
                               c1[1] - 2),
                0, tl / 3, [0, 0, 0],
                thickness=tf, lineType=cv2.LINE_AA)
    # ---------------------------------------------------------------------------- #


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='OCR')
    argparser.add_argument(
        '-d', '--desc', help='Description', default='Quadrep OCR')
    args = argparser.parse_args()

    app = QApplication([])
    ocr_runner = OcrRunner()
    QThreadPool.globalInstance().start(ocr_runner)
    sys.exit(app.exec())
