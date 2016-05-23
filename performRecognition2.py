# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np

dico = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# Load the classifier
clf = joblib.load("digits_cls2.pkl")

# Read the input image 
im = cv2.imread("IMG_1148.jpg")

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 120, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((5,5),np.uint8)
im_th = cv2.morphologyEx(im_th, cv2.MORPH_DILATE, kernel)
im_th = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, kernel)


cv2.imshow("Pre-processed image", im_th)
cv2.waitKey()

# Find contours in the image
ctrs, hierarchy = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
for rect in rects:
    # Draw the rectangles
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
    # Make the rectangular region around the digit
    center = (rect[0]+rect[2]//2,rect[1] + rect[3]//2)
    half_width  = min(rotated_img.shape[1]-center[0],rect[2]//2)
    half_height = min(rotated_img.shape[0]-center[1],rect[3]//2)
    half_size = max(half_width,half_height)
    cropped = np.zeros((2*half_size,2*half_size))
    for i in range(-half_height,half_height):
        for j in range(-half_width,half_width):
            cropped[half_size+i,half_size+j] = rotated_img[center[1]+i,center[0]+j]
    leng = int(rect[3] * 1.6)
    if leng <= 1:
        continue
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
    # Resize the image
    if len(roi) <= 2:
        continue
    try:
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        cv2.putText(im, str(dico[int(nbr[0])-1]), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    except:
        continue

cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()