import os
import json
import cv2

import pyrealsense2 as rs

from PySide6.QtCore import (QThread)

from list_camera_names import list_available_cameras

CAMERAS = os.path.join(os.path.dirname(__file__), 'cameras.json')


class CameraWorker(QThread):
    def __init__(self, device_name, resolution=(640, 480),
                 fps=24, supported_cameras=CAMERAS) -> None:
        super().__init__()
        self.device_name = device_name
        self.supported_camera_file = supported_cameras
        self.resolution = resolution
        self.fps = fps
        self.delay = int(1000 / self.fps)

        self.isOpened = self.setup_capture(self.device_name)

    def setup_capture(self, device_name):
        self.camera_names = list_available_cameras()
        self.supported_cameras = load_supported_cameras(
            self.supported_camera_file)
        if 'Realsense' in device_name:
            # TODO check resolution valid?
            pass
        else:
            if device_name in self.supported_cameras:
                self.device_index = self.camera_names.index(device_name)
                self.video_capture = self.setup_normal_camera()

    def setup_normal_camera(self):
        width, height = self.resolution
        video_capture = cv2.VideoCapture(self.device_index + cv2.CAP_DSHOW)
        video_capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(
            'M', 'J', 'P', 'G'))
        return video_capture


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls.__instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(
                *args, **kwargs)
        return cls._instances[cls]


class RealsenseCapture():
    _isOpened = False

    def __init__(self, resolution, fps=25) -> None:
        self.resolution = resolution
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        width, height = self.resolution
        self.config.enable_stream(
            rs.stream.color, width, height, rs.format.rgb8, fps)
        try:
            pipeline_profile = self.config.resolve(
                rs.pipeline_wrapper(self.pipeline))

            self.profile = self.pipeline.start(self.config)
            # TODO add sensor config

            print(f'\n    RealsenseCapture - initialized')
        except:
            print(f'\n    RealsenseCapture - initialized not success')

    # TODO add read
    # TODO add isOpened
    # TODO add release?


def load_supported_cameras(file_name):
    json_data = load_json_file(file_name)
    camera_names = list(json_data.keys())
    return camera_names


def load_json_file(file_name):
    with open(file_name) as f:
        data = json.load(f)
    return data


if __name__ == '__main__':
    camera = CameraWorker(device_name='Intel(R) RealSense(TM) 515')
    exit(0)
