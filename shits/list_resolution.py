from timing_operation import measure
from timeit import timeit
import os
import pandas as pd
import cv2


def get_cur_file_dir():
    """Get directory of current file

    :return: directory path
    :rtype: str
    """
    return os.path.dirname(os.path.realpath(__file__))


def download_possible_resolution(saving_path='./resolutions.json'):
    """Download from internet list of possible resolutions of cameras


    :param saving_path: path to saving config, defaults to './resolutions.json'
    :type saving_path: str, optional
    :return: list of possible resolutions
    :rtype: list
    """
    # 1. Retrieve infor from internet
    url = "https://en.wikipedia.org/wiki/List_of_common_resolutions"
    table = pd.read_html(url)[0]
    # 2. Extract resolutions
    table.columns = table.columns.droplevel()
    resolution_list = table[["W", "H"]]
    # 3. Save resolutions into json file to use later
    if saving_path is not None:
        # data = resolution_list.to_json(orient='index')
        data = resolution_list.to_json(saving_path, orient='index')
    return resolution_list.to_numpy().tolist()


possible_resolutions = download_possible_resolution(saving_path=os.path.join(
    get_cur_file_dir(), 'resolutions.json'))


@measure
def retrieve_valid_resolutions(cap, resolutions, verbose=False):
    """Check valid resolutions

    :param cap: video capture
    :type cap: cv2.VideoCapture
    :param resolutions: list of possible resolutions
    :type resolutions: ndarray 2D
    :return: list of valid resolutions
    :rtype: list
    """
    valid_resolutions = set()
    for w, h in resolutions:
        if verbose:
            print(w, h)
        # 1. Try to config each resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        # 2. Update valid resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        valid_resolutions.add((width, height))
    return list(valid_resolutions)


cap = cv2.VideoCapture(1)
valid_resolutions = retrieve_valid_resolutions(
    cap, possible_resolutions, verbose=True)
print(valid_resolutions)
