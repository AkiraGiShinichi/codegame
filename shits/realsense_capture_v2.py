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
        """Class to manage the Intel Realsense capture

        :param depth_size: Size of depth frame, defaults to (640, 480)
        :type depth_size: tuple, optional
        :param color_size: Size of color frame, defaults to (640, 480)
        :type color_size: tuple, optional
        :param fps: FPS of capture, defaults to 30
        :type fps: int, optional
        """
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
            _ = self.read()

    def read(self, return_depth=False, depth_filter=None):
        """Read data from camera

        :param return_depth: Whether return depth image or not, defaults to False
        :type return_depth: bool, optional
        :param depth_filter: [description], defaults to None
        :type depth_filter: [type], optional
        :return: Whether having data, and data
        :rtype: tuple(bool, array or list of array)
        """
        try:
            frames = self._enabled_device.pipeline.wait_for_frames()
            # Align the depth frame to color frame
            self._frames = self._enabled_device.align.process(frames)

            if return_depth:  # Return RGB image and Depth image
                return True, self.get_data_according_type(
                    DataType.IMAGES, depth_filter)
            else:  # Return RGB image only
                return True, self.get_data_according_type(DataType.COLOR_IMAGE)
        except:
            self._camera_is_open = False
            print(f'\n    RealsenseCapture - read: error')
            return False, None

    def isOpened(self):
        """Check whether the camera is open(ready to use)

        :return: Is open or not
        :rtype: bool
        """
        return self._camera_is_open

    def release(self):
        """Release/Disable cameras
        """
        print(f'\n    RealsenseCapture - release')
        self._config.disable_all_streams()

    def get_intrinsics(self, frame_type: DataType = DataType.COLOR_FRAME):
        """Get intrinsics of a frame(depth ? color)
        :In this case, after alignment, intrinsics of depth and color frames
        :are the same

        :param frame_type: Type of frame, defaults to DataType.COLOR_FRAME
        :type frame_type: DataType, optional
        :return: intrinsics
        :rtype: rs.intrinsics
        """
        assert frame_type == DataType.COLOR_FRAME or frame_type == DataType.DEPTH_FRAME

        if frame_type == DataType.COLOR_FRAME:
            frame = self.get_data_according_type(DataType.COLOR_FRAME)
        elif frame_type == DataType.DEPTH_FRAME:
            frame = self.get_data_according_type(DataType.DEPTH_FRAME)

        if frame is None:
            return None

        intrinsics = frame.get_profile().as_video_stream_profile().get_intrinsics()
        return intrinsics

    def get_depth_to_color_extrinsics(self):
        """Get extrinsics from depth frame to color frame

        :return: Extrinsics
        :rtype: rs.extrinsics
        """
        color_frame = self.get_data_according_type(DataType.COLOR_FRAME)
        depth_frame = self.get_data_according_type(DataType.DEPTH_FRAME)

        if color_frame is None or depth_frame is None:
            return None

        extrinsics = depth_frame.get_profile().as_video_stream_profile(
        ).get_extrinsics_to(color_frame.get_profile())
        return extrinsics

    def get_depth_frame(self, depth_filter=None):
        """Get depth frame

        :param depth_filter: Function to filter depth frame, defaults to None
        :type depth_filter: object, optional
        :return: Depth frame after filtered
        :rtype: rs.depth_frame
        """
        if self._frames is None:
            return None

        depth_frame = self._frames.get_depth_frame()
        if depth_filter is not None:
            depth_frame = depth_filter(depth_frame)
        return depth_frame

    def get_data_according_type(self, data_type: DataType = DataType.FRAMES,
                                depth_filter=None):
        """Get data according to type

        :param data_type: Expected type of data, defaults to DataType.FRAMES
        :type data_type: DataType, optional
        :param depth_filter: Function to filter depth frame, defaults to None
        :type depth_filter: object, optional
        :return: Data
        :rtype: frame or array
        """
        if self._frames is None:
            return None

        if data_type == DataType.FRAMES:
            return self._frames
        elif data_type == DataType.COLOR_FRAME:
            return self._frames.get_color_frame()
        elif data_type == DataType.DEPTH_FRAME:
            return self.get_depth_frame(depth_filter)
        elif data_type == DataType.COLOR_IMAGE:
            return np.asarray(self._frames.get_color_frame().get_data())
        elif data_type == DataType.DEPTH_IMAGE:
            return np.asarray(self.get_depth_frame(depth_filter).get_data())
        elif data_type == DataType.IMAGES:
            color_image = np.asarray(self._frames.get_color_frame().get_data())
            depth_image = np.asarray(self.get_depth_frame(depth_filter)
                                     .get_data())
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
            # Capture image
            status, images = realsense_capture.read(
                return_depth=True)  # , depth_filter=post_process_depth_frame
            # Display image
            if status:
                color_image, depth_image = images
                if observe == Observation.COLOR:
                    cv2.imshow('Test', cv2.cvtColor(
                        color_image, cv2.COLOR_RGB2BGR))
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

    # Other tests
    intrinsics = realsense_capture.get_intrinsics()
    extrinsics = realsense_capture.get_depth_to_color_extrinsics()

    # Release capture
    realsense_capture.release()
    print('Byebye!')
# ---------------------------------------------------------------------------- #
