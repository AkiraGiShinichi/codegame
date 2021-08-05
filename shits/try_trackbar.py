# Import dependancies
import cv2

maxScaleUp = 100
scaleFactor = 1
windowName = "Resize Image"
trackbarValue = "Scale"

vid_capture = cv2.VideoCapture(0)

# read the image
# ret, image = vid_capture.read()
# image = cv2.imread("../Input/sample.jpg")

# Create a window to display results and  set the flag to Autosize
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)

# Callback functions


def scaleImage(*args):
    global scaleFactor
    # Get the scale factor from the trackbar
    scaleFactor = 1 + args[0]/100.0
    # # Resize the image
    # scaledImage = cv2.resize(
    #     image, None, fx=scaleFactor, fy=scaleFactor,
    #     interpolation=cv2.INTER_LINEAR)
    # cv2.imshow(windowName, scaledImage)


# Create trackbar and associate a callback function
cv2.createTrackbar(trackbarValue, windowName,
                   scaleFactor, maxScaleUp, scaleImage)

while 1:
    # read the image
    ret, image = vid_capture.read()
    # Resize the image
    scaledImage = cv2.resize(
        image, None, fx=scaleFactor, fy=scaleFactor,
        interpolation=cv2.INTER_LINEAR)
    # Display the image
    cv2.imshow(windowName, scaledImage)
    key = cv2.waitKey(1)
cv2.destroyAllWindows()
