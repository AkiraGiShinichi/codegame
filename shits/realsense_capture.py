import numpy as np
import pyrealsense2 as rs
import cv2


class RealsenseCapture():
    camera_is_open = False

    def __init__(
            self, depth_size=(640, 480),
            color_size=(640, 480),
            fps=30) -> None:
        self.depth_size = depth_size
        self.color_size = color_size
        # Create a pipeline
        self.pipeline = rs.pipeline()
        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        self.config = rs.config()
        color_width, color_height = self.color_size
        depth_width, depth_height = self.depth_size
        self.config.enable_stream(
            rs.stream.color, color_width, color_height, rs.format.rgb8, fps)
        self.config.enable_stream(
            rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)
        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        try:
            # ! If there isn't any Realsense camera, this function is broken immediately -> time saving
            pipeline_profile = self.config.resolve(
                rs.pipeline_wrapper(self.pipeline))

            self.profile = self.pipeline.start(self.config)

            # sensor = self.profile.get_device().query_sensors()[1]
            # sensor.set_option(rs.option.exposure, 4)
            # sensor.set_option(rs.option.gain, 80)  # 84

            self.camera_is_open = True
            print(f'\n    RealsenseCapture - initialized')
        except:
            print(f'\n    RealsenseCapture - initialized not success')

    def read(self, return_depth=False):
        """Read BGR image from Realsense camera

        Returns:
            [bool]: able to capture frame or not
            [ndarray]: frame
        """
        try:
            frames = self.pipeline.wait_for_frames()
            # Align the depth frame to color frame
            frames = self.align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            color_image = np.asarray(color_frame.get_data())
            depth_image = np.asarray(depth_frame.get_data())

            if return_depth:
                return True, color_image[:, :, ::-1], depth_image
            else:
                return True, color_image[:, :, ::-1]
        except:
            self.camera_is_open = False
            print(f'\n    RealsenseCapture - read: error')
            return False, None

    def isOpened(self):
        return self.camera_is_open

    def release(self):
        print(f'\n    RealsenseCapture - release')
        try:
            self.pipeline.stop()
        except:
            print(
                f'\n    RealsenseCapture - release: error. Camera is not initialized yet.')


if __name__ == '__main__':
    realsense_capture = RealsenseCapture(
        depth_size=(640, 480),
        color_size=(960, 540),
        fps=30)  # L515
    # realsense_capture = RealsenseCapture(depth_size=(640, 480), color_size=(640, 480), fps=30) # D435i
    while 1:
        if realsense_capture.isOpened():
            status, color_image, depth_image = realsense_capture.read(
                return_depth=True)
            if status:
                cv2.imshow('Test', color_image)
                cv2.waitKey(100)
    print('Byebye!')
