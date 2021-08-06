# Import dependancies
import cv2

max_scale_up = 100
scale_factor = 1
window_name = "Resize Image"
trackbar_value = "Scale"

vid_capture = cv2.VideoCapture(0)

# read the image
# ret, image = vid_capture.read()

# Create a window to display results and  set the flag to Autosize
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

# Callback functions


def change_scale_factor(*args):
    """Change image scale factor
    """
    global scale_factor
    # Get the scale factor from the trackbar
    scale_factor = 1 + args[0]/100.0


# Create trackbar and associate a callback function
cv2.createTrackbar(trackbar_value, window_name,
                   scale_factor, max_scale_up, change_scale_factor)

while 1:
    # read the image
    ret, image = vid_capture.read()
    # image = cv2.imread("../Input/sample.jpg")

    # Resize the image
    scaledImage = cv2.resize(
        image, None, fx=scale_factor, fy=scale_factor,
        interpolation=cv2.INTER_LINEAR)

    # Display the image
    cv2.imshow(window_name, scaledImage)
    key = cv2.waitKey(1)

cv2.destroyAllWindows()
