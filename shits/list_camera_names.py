from pygrabber.dshow_graph import FilterGraph
from timing_operation import measure


@measure
def list_available_cameras():
    """List all available cameras

    :return: list of cameras
    :rtype: list
    """
    graph = FilterGraph()
    device_names = graph.get_input_devices()
    return device_names


if __name__ == '__main__':
    print(list_available_cameras())
    exit(0)
