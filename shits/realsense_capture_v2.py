import numpy as np
import pyrealsense2 as rs
import cv2
from enum import Enum


# ----------------------------- Helper functions ----------------------------- #
class Device:
    def __init__(self, pipeline, pipeline_profile, align, product_line):
        self.pipeline = pipeline
        self.pipeline_profile = pipeline_profile
        self.align = align
        self.product_line = product_line


def enumerate_connected_devices(context):
    """Enumerate the connected Intel Realsense devices

    :param context: The context created for using the realsense library
    :type context: rs.context()
    :return: List of (serial-number, product-line) tuples of devices which are connected to the PC
    :rtype: list
    """
    connect_device = []

    for d in context.devices:
        if d.get_info(rs.camera_info.name).lower() != 'platform camera':
            device_serial = d.get_info(rs.camera_info.serial_number)
            product_line = d.get_info(rs.camera_info.product_line)
            device_info = (device_serial, product_line)
            connect_device.append(device_info)
    return connect_device


def post_process_depth_frame(
        depth_frame, decimation_magnitude=1.0, spatial_magnitude=2.0,
        spatial_smooth_alpha=0.5, spatial_smooth_delta=20,
        temporal_smooth_alpha=0.4, temporal_smooth_delta=20):
    """
    Filter the depth frame acquired using the Intel RealSense device
    Parameters:
    -----------
    depth_frame          : rs.frame()
                           The depth frame to be post-processed
    decimation_magnitude : double
                           The magnitude of the decimation filter
    spatial_magnitude    : double
                           The magnitude of the spatial filter
    spatial_smooth_alpha : double
                           The alpha value for spatial filter based smoothening
    spatial_smooth_delta : double
                           The delta value for spatial filter based smoothening
    temporal_smooth_alpha: double
                           The alpha value for temporal filter based smoothening
    temporal_smooth_delta: double
                           The delta value for temporal filter based smoothening
    Return:
    ----------
    filtered_frame : rs.frame()
                     The post-processed depth frame
    """
    # Post processing possible only on the depth_frame
    assert (depth_frame.is_depth_frame())

    # Available filters and control options for the filters
    decimation_filter = rs.decimation_filter()
    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()

    filter_magnitude = rs.option.filter_magnitude
    filter_smooth_alpha = rs.option.filter_smooth_alpha
    filter_smooth_delta = rs.option.filter_smooth_delta

    # Apply the control parameters for the filter
    decimation_filter.set_option(filter_magnitude, decimation_magnitude)
    spatial_filter.set_option(filter_magnitude, spatial_magnitude)
    spatial_filter.set_option(filter_smooth_alpha, spatial_smooth_alpha)
    spatial_filter.set_option(filter_smooth_delta, spatial_smooth_delta)
    temporal_filter.set_option(filter_smooth_alpha, temporal_smooth_alpha)
    temporal_filter.set_option(filter_smooth_delta, temporal_smooth_delta)

    # Apply the filters
    filtered_frame = decimation_filter.process(depth_frame)
    filtered_frame = spatial_filter.process(filtered_frame)
    filtered_frame = temporal_filter.process(filtered_frame)

    return filtered_frame


# class SingleInstanceMetaClass(type):
#     def __init__(self, name, bases, dic):
#         self.__single_instance = None
#         super().__init__(name, bases, dic)

#     def __call__(cls, *args, **kwargs):
#         if cls.__single_instance:
#             return cls.__single_instance
#         single_obj = cls.__new__(cls)
#         single_obj.__init__(*args, **kwargs)
#         cls.__single_instance = single_obj
#         return single_obj
class SingleInstanceMetaClass(type):
    def __call__(cls, *args, **kwargs):
        try:
            return cls.__instance
        except AttributeError:
            cls.__instance = super(
                SingleInstanceMetaClass, cls).__call__(
                *args, **kwargs)
            return cls.__instance
# ---------------------------------------------------------------------------- #


# ------------------------------- Main content ------------------------------- #
class DataType(Enum):
    FRAMES = 1
    COLOR_FRAME = 2
    DEPTH_FRAME = 3
    COLOR_IMAGE = 4
    DEPTH_IMAGE = 5
    IMAGES = 6


class RealsenseCapture(metaclass=SingleInstanceMetaClass):

    def __init__(
            self, depth_size=(640, 480),
            color_size=(640, 480),
            fps=30) -> None:
        self._depth_size = depth_size
        self._color_size = color_size
        self._fps = fps

        self._context = rs.context()
        _available_devices = enumerate_connected_devices(self._context)
        self._device_serial, self._product_line = _available_devices[0]
        self._enabled_device = None

        color_width, color_height = self._color_size
        depth_width, depth_height = self._depth_size
        self._config = rs.config()
        self._config.enable_stream(
            rs.stream.color, color_width, color_height, rs.format.rgb8, fps)
        self._config.enable_stream(
            rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)

        self._camera_is_open = False
        self._frames = None

    def enable_device(self, enable_ir_emitter=False):
        """Enable an Intel Realsense device

        :param device_info: Serial number and product line of the Realsense device
        :type device_info: tuple(str, str)
        :param enable_ir_emitter: Enable/Disable the IR-Emitter of the device
        :type enable_ir_emitter: bool
        """
        try:
            pipeline = rs.pipeline()

            self._config.enable_device(self._device_serial)
            pipeline_profile = pipeline.start(self._config)

            # Set the acquisition parameters
            sensor = pipeline_profile.get_device().first_depth_sensor()
            if sensor.supports(rs.option.emitter_enabled):
                sensor.set_option(rs.option.emitter_enabled, 1
                                  if enable_ir_emitter else 0)
            # Create an align object
            # rs.align allows us to perform alignment of depth frames to others frames
            # The "align_to" is the stream type to which we plan to align depth frames.
            align_to = rs.stream.color
            align = rs.align(align_to)

            self._enabled_device = Device(
                pipeline, pipeline_profile, align, self._product_line)

            self._camera_is_open = True
            print(f'\n    RealsenseCapture - initialized')
        except:
            print(f'\n    RealsenseCapture - initialized not success')

    def warm_up(self, dispose_frames_for_stablisation):
        """Dispose some frames for stablisation

        :param dispose_frames_for_stablisation: Number of disposing frames
        :type dispose_frames_for_stablisation: int
        """
        for _ in range(dispose_frames_for_stablisation):
            frames = self.read()

    def read(self, return_depth=False, depth_filter=None):
        """Read BGR image from Realsense camera

        Returns:
            [bool]: able to capture frame or not
            [ndarray]: frame
        """
        try:
            frames = self._enabled_device.pipeline.wait_for_frames()
            # Align the depth frame to color frame
            self._frames = self._enabled_device.align.process(frames)

            color_frame = self._frames.get_color_frame()
            depth_frame = self._frames.get_depth_frame()
            if depth_filter is not None:
                depth_frame = depth_filter(depth_frame)

            color_image = np.asarray(color_frame.get_data())
            depth_image = np.asarray(depth_frame.get_data())

            if return_depth:
                return True, [color_image[:, :, ::-1], depth_image]
            else:
                return True, color_image[:, :, ::-1]
        except:
            self._camera_is_open = False
            print(f'\n    RealsenseCapture - read: error')
            return False, None

    def isOpened(self):
        return self._camera_is_open

    def release(self):
        print(f'\n    RealsenseCapture - release')
        self._config.disable_all_streams()

    def get_intrinsics(self, type: DataType = DataType.COLOR_FRAME):
        assert type == DataType.COLOR_FRAME or type == DataType.DEPTH_FRAME

        if type == DataType.COLOR_FRAME:
            frame = self.get_data_according_type(DataType.COLOR_FRAME)
        elif type == DataType.DEPTH_FRAME:
            frame = self.get_data_according_type(DataType.DEPTH_FRAME)

        intrinsics = frame.get_profile().as_video_stream_profile().get_intrinsics()
        return intrinsics

    def get_depth_to_color_extrinsics(self):
        color_frame = self.get_data_according_type(DataType.COLOR_FRAME)
        depth_frame = self.get_data_according_type(DataType.DEPTH_FRAME)
        extrinsics = depth_frame.get_profile().as_video_stream_profile(
        ).get_extrinsics_to(color_frame.get_profile())
        return extrinsics

    def get_data_according_type(self, type: DataType = DataType.FRAMES):
        if type == DataType.FRAMES:
            return self._frames
        elif type == DataType.COLOR_FRAME:
            return self._frames.get_color_frame()
        elif type == DataType.DEPTH_FRAME:
            return self._frames.get_depth_frame()
        elif type == DataType.COLOR_IMAGE:
            return np.asarray(self._frames.get_color_frame().get_data())
        elif type == DataType.DEPTH_IMAGE:
            return np.asarray(self._frames.get_depth_frame().get_data())
        elif type == DataType.IMAGES:
            color_image = np.asarray(self._frames.get_color_frame().get_data())
            depth_image = np.asarray(self._frames.get_color_frame().get_data())
            return (color_image, depth_image)
# ---------------------------------------------------------------------------- #


# ---------------------------------- Testing --------------------------------- #
class Observation(Enum):
    COLOR = 1
    DEPTH = 2


if __name__ == '__main__':
    # Initialize capture
    realsense_capture = RealsenseCapture(
        depth_size=(640, 480),
        color_size=(960, 540),
        fps=30)  # L515
    print(realsense_capture)
    # realsense_capture = RealsenseCapture(depth_size=(640, 480), color_size=(640, 480), fps=30) # D435i
    realsense_capture.enable_device()

    # Observe image
    observe = Observation.COLOR
    while 1:
        if realsense_capture.isOpened():
            status, images = realsense_capture.read(
                return_depth=True)  # , depth_filter=post_process_depth_frame
            if status:
                color_image, depth_image = images
                if observe == Observation.COLOR:
                    cv2.imshow('Test', color_image)
                else:
                    cv2.imshow('Test', depth_image)
                key = cv2.waitKey(100)
                if key & 0xFF == ord('q'):
                    break
                elif key & 0xFF == ord('1'):
                    observe = Observation.COLOR
                elif key & 0xFF == ord('2'):
                    observe = Observation.DEPTH
        else:
            break

    # Release capture
    realsense_capture.release()
    print('Byebye!')
# ---------------------------------------------------------------------------- #
