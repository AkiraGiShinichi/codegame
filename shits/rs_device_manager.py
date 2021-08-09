import pyrealsense2 as rs
import numpy as np


# ----------------------------- Helper functions ----------------------------- #
class Device:
    def __init__(self, pipeline, pipeline_profile, product_line):
        self.pipeline = pipeline
        self.pipeline_profile = pipeline_profile
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
class DeviceManager(metaclass=SingleInstanceMetaClass):
    def __init__(self, context, D400_pipeline_configuration,
                 L500_pipeline_configuration=rs.config()):
        """Class to manage the Intel Realsense devices

        :param context: The context created for using the realsense library
        :type context: rs.context()
        :param D400_pipeline_configuration: The Realsense library configuration to be used for the application when D400 product is attached
        :type D400_pipeline_configuration: rs.config()
        :param L500_pipeline_configuration: The Realsense library configuration to be used for the application when L500 product is attached, defaults to rs.config()
        :type L500_pipeline_configuration: rs.config(), optional
        """
        assert isinstance(context, type(rs.context()))
        assert isinstance(D400_pipeline_configuration, type(rs.config()))
        assert isinstance(L500_pipeline_configuration, type(rs.config()))

        self._context = context
        self._available_devices = enumerate_connected_devices(context)
        self._enabled_devices = {}
        self.D400_config = D400_pipeline_configuration
        self.L500_config = L500_pipeline_configuration
        self._frame_counter = 0

    def enable_device(self, device_info, enable_ir_emitter=False):
        """Enable an Intel Realsense device

        :param device_info: Serial number and product line of the Realsense device
        :type device_info: tuple(str, str)
        :param enable_ir_emitter: Enable/Disable the IR-Emitter of the device
        :type enable_ir_emitter: bool
        """
        pipeline = rs.pipeline()

        device_serial, product_line = device_info

        if product_line == 'L500':  # Enable L515 device
            self.L500_config.enable_device(device_serial)
            pipeline_profile = pipeline.start(self.L500_config)
        else:  # Enable D400 device
            self.D400_config.enable_device(device_serial)
            pipeline_profile = pipeline.start(self.D400_config)

        # Set the acquisition parameters
        sensor = pipeline_profile.get_device().first_depth_sensor()
        if sensor.supports(rs.option.emitter_enabled):
            sensor.set_option(rs.option.emitter_enabled, 1
                              if enable_ir_emitter else 0)
        self._enabled_devices[device_serial] = (
            Device(pipeline, pipeline_profile, product_line))

    def enable_device_according_serial(self, serial):
        """Enable device according serial number

        :param serial: Serial number of the device
        :type serial: str
        :return: Whether device is enabled successfully
        :rtype: bool
        """
        device_info = self.get_device_info_from_serial(serial)
        if device_info is None:
            return False
        else:
            self.enable_device(device_info)
            return True

    def enable_all_devices(self, enable_ir_emitter=False):
        """Enable all the Intel Realsense devices which are connected to the PC

        :param enable_ir_emitter: Enable/Disable the IR-Emitter of the devices, defaults to False
        :type enable_ir_emitter: bool, optional
        """
        print(str(len(self._available_devices)) + " devices have been found")

        for device_info in self._available_devices:
            self.enable_device(device_info, enable_ir_emitter)

    def enable_emitter(self, enable_ir_emitter=True):
        pass

    def load_settings_json(self, path_to_settings_file):
        """Load the settings stored in the JSON file

        :param path_to_settings_file: path to the json file of settings
        :type path_to_settings_file: str
        """
        with open(path_to_settings_file, 'r') as file:
            json_text = file.read().strip()

        for (device_serial, device) in self._enabled_devices.items():
            if device.product_line == 'L500':
                continue
            # Get the active profile and load the json file which contains settings readable by the Realsense
            device = device.pipeline_profile.get_device()
            advanced_mode = rs.rs400_advanced_mode(device)
            advanced_mode.load_json(json_text)

    def poll_frames(self):
        """Poll for frames from the enabled Intel Realsense devices. This will return at least one frame from each device.
        :If temporal post processing is enabled, the depth stream is averaged over a certain amount of frames

        :return: Set of frames of all stream types of all enabled-devices 
        :rtype: Dict
        """
        frames = {}
        while len(frames) < len(self._enabled_devices.items()):
            for (serial, device) in self._enabled_devices.items():
                streams = device.pipeline_profile.get_streams()
                # frameset will be a pyrealsense2.composite_frame object
                frameset = device.pipeline.poll_for_frames()
                # frameset = device.pipeline.wait_for_frames()
                # ! Move outside by Akira
                # ! End of modification by Akira
                if frameset.size() == len(streams):
                    device_info = (serial, device.product_line)
                    frames[device_info] = {}
                    for stream in streams:
                        if (rs.stream.infrared == stream.stream_type()):
                            frame = frameset.get_infrared_frame(
                                stream.stream_index())
                            key_ = (stream.stream_type(), stream.stream_index())
                        else:
                            frame = frameset.first_or_default(
                                stream.stream_type())
                            key_ = stream.stream_type()
                        frames[device_info][key_] = frame
                # else:
                #     frames[device_info] = None

        return frames

    def get_depth_shape(self):
        """Returns width and height of the depth stream for one arbitrary device

        :return: (width, height) of depth stream
        :rtype: list(int, int)
        """
        width, height = -1, -1

        for (serial, device) in self._enabled_devices.items():
            for stream in device.pipeline_profile.get_streams():
                if (rs.stream.depth == stream.stream_type()):
                    width = stream.as_video_stream_profile().width()
                    height = stream.as_video_stream_profile().height()
                    return width, height

    def get_device_intrinsics(self, frames):
        pass

    def get_depth_to_color_extrinsics(self, frames):
        """Get the extrinsics between the depth imager 1 and the color imager using its frame delivered by the Realsense device

        :param frames: The frame grabbed from the imager inside the Intel Realsense for which the intrinsics is needed
        :type frames: rs::frame
        :return: Diction of {keys: values} in which key_: serial number of the device, value_: extrinsic of the corresponding device
        :rtype: dict
        """
        device_extrinsics = {}

        for (device_info, frameset) in frames.items():
            serial = device_info[0]
            device_extrinsics[serial] = frameset[rs.stream.depth].get_profile(
            ).as_video_stream_profile().get_extrinsics_to(frameset[rs.stream.color].get_profile())
        return device_extrinsics

    def get_device_info_from_serial(self, serial):
        """Get device information from its serial number

        :param serial: Serial number of expected device
        :type serial: str
        :return: Device info of (serial_number, product_line)
        :rtype: tuple
        """
        for device_info in self._available_devices:
            if serial in device_info:
                return device_info
        return None

    def extract_frames_of_device(self, frames, serial):
        """Extract frames of a device from a set of frames of multiple devices

        :param frames: Set of frames grabbed from the imager inside the Intel Realsense
        :type frames: dict
        :param serial: Serial number of expected device
        :type serial: str
        :return: Frames of expected device
        :rtype: dict
        """
        device_info = self.get_device_info_from_serial(serial)
        if device_info is None:
            return None
        else:
            return frames[device_info]

    def extract_frame_according_type(self, frames, frame_type, frame_index=0):
        """Extract frame of type from a set of frames

        :param frames: Set of frames grabbed from the imager inside the Intel Realsense
        :type frames: dict
        :param frame_type: Expected frame type
        :type frame_type: [rs.stream.depth, rs.stream.color, rs.stream.infrared]
        :param frame_index: Index of infrared frame, defaults to 0
        :type frame_index: int, optional
        :return: Frame of expected type
        :rtype: rs::frame
        """
        for key, value in frames.items():
            if isinstance(key, tuple):
                _type, _index = key
                if _type == frame_type and _index == frame_index:
                    return value
            else:
                _type = key
                if _type == frame_type:
                    return value
        return None

    def convert_frame_to_array(self, frame):
        """Convert Realsense frame to array

        :param frame: Realsense frame
        :type frame: rs::frame
        :return: Array-data of frame
        :rtype: array
        """
        return np.asarray(frame.get_data())

    def warm_up(self, dispose_frames_for_stablisation):
        """Dispose some frames for stablisation

        :param dispose_frames_for_stablisation: Number of disposing frames
        :type dispose_frames_for_stablisation: int
        """
        for _ in range(dispose_frames_for_stablisation):
            frames = self.poll_frames()

    def disable_all_streams(self):
        """Disable all streams
        """
        self.D400_config.disable_all_streams()
        self.L500_config.disable_all_streams()


# ---------------------------------------------------------------------------- #


# ---------------------------------- Testing --------------------------------- #
def test_1():
    try:
        w, h = 1280, 720
        c = rs.config()
        c.enable_stream(rs.stream.depth, w, h, rs.format.z16, 6)
        c.enable_stream(rs.stream.infrared, 1, w, h, rs.format.y8, 6)
        c.enable_stream(rs.stream.infrared, 2, w, h, rs.format.y8, 6)
        c.enable_stream(rs.stream.color, w, h, rs.format.rgb8, 6)
        device_manager = DeviceManager(rs.context(), c)
        device_manager.enable_all_devices()

        targeted_device = enumerate_connected_devices()[0]
        epochs = 15

        import cv2
        for k in range(epochs):
            frames = device_manager.poll_frames()
            frames_of_device = frames[(targeted_device[0], targeted_device[1])]
            print(frames_of_device)
            # cv2.imshow('A', np.zeros((100, 100, 3)))
            # cv2.waitKey(0)
        # device_manager.enable_emitter(True)
        device_extrinsics = device_manager.get_depth_to_color_extrinsics(frames)
        print(device_extrinsics)
    except Exception as e:
        print(e)
    finally:
        device_manager.disable_all_streams()


def create_L515_config():
    L515_resolution_width, L515_resolution_height = 1024, 768
    L515_frame_rate = 30

    resolution_width, resolution_height = 1920, 1080  # 1920, 1080 ; 1280, 720
    frame_rate = 15

    L515_rs_config = rs.config()
    L515_rs_config.enable_stream(rs.stream.depth,
                                 L515_resolution_width, L515_resolution_height,
                                 rs.format.z16, L515_frame_rate)
    L515_rs_config.enable_stream(rs.stream.infrared, 0,
                                 L515_resolution_width, L515_resolution_height,
                                 rs.format.y8, L515_frame_rate)
    L515_rs_config.enable_stream(rs.stream.color,
                                 resolution_width, resolution_height,
                                 rs.format.bgr8, L515_frame_rate)
    return L515_rs_config


def create_D400_config():
    L515_resolution_width, L515_resolution_height = 1024, 768
    L515_frame_rate = 30

    resolution_width, resolution_height = 1280, 720
    frame_rate = 15

    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth,
                            resolution_width, resolution_height,
                            rs.format.z16, frame_rate)
    rs_config.enable_stream(rs.stream.infrared, 1,
                            resolution_width, resolution_height,
                            rs.format.y8, frame_rate)
    rs_config.enable_stream(rs.stream.color,
                            resolution_width, resolution_height,
                            rs.format.bgr8, frame_rate)
    return rs_config


def test_2():
    dispose_frames_for_stablisation = 30
    targeted_device = enumerate_connected_devices(rs.context())[0][0]

    L515_rs_config = create_L515_config()

    rs_config = create_D400_config()

    # Create device manager
    device_manager = DeviceManager(
        rs.context(),
        rs_config, L515_rs_config)  # , L515_rs_config
    print(device_manager)
    # device_manager.enable_all_devices()
    status = device_manager.enable_device_according_serial(targeted_device)
    # Warm-up
    device_manager.warm_up(dispose_frames_for_stablisation)

    import cv2
    while 1:
        frames = device_manager.poll_frames()
        frames_of_device = device_manager.extract_frames_of_device(
            frames, targeted_device)
        if frames_of_device is not None:
            color_frame = device_manager.extract_frame_according_type(
                frames_of_device, rs.stream.color)
            depth_frame = device_manager.extract_frame_according_type(
                frames_of_device, rs.stream.depth)

            color_image = device_manager.convert_frame_to_array(color_frame)
            depth_image = device_manager.convert_frame_to_array(depth_frame)

            cv2.imshow('a', color_image)
            cv2.waitKey(100)

        print(frames_of_device)

    # except Exception as e:
    #     print(f'Error: {e}')
    # finally:
    #     device_manager.disable_all_streams()


if __name__ == '__main__':
    # test_1()
    test_2()
# ---------------------------------------------------------------------------- #
