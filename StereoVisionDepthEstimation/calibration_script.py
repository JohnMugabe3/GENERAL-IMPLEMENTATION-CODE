import numpy as np
import cv2
import glob
import os

# Termination criteria for cornerSubPix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0)....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane for left camera.
imgpointsR = [] # 2d points in image plane for right camera.

# Directories containing the images
left_image_dir = r"C:\Users\johnm\Desktop\THESIS 19.08.2024\PyBULLET 1\ComputerVision\StereoVisionDepthEstimation\images\IMAGES LEFT"
right_image_dir = r"C:\Users\johnm\Desktop\THESIS 19.08.2024\PyBULLET 1\ComputerVision\StereoVisionDepthEstimation\images\IMAGES RIGHT"

# Check if directories exist
if not os.path.isdir(left_image_dir) or not os.path.isdir(right_image_dir):
    print(f"One or both directories '{left_image_dir}' and '{right_image_dir}' do not exist.")
    sys.exit()

# Images
images_left = glob.glob(os.path.join(left_image_dir, '*.png'))
images_right = glob.glob(os.path.join(right_image_dir, '*.png'))

# Check if images are found
if not images_left or not images_right:
    print(f"No images found in '{left_image_dir}' or '{right_image_dir}'.")
    sys.exit()

# Read the first image to get the shape
imgL = cv2.imread(images_left[0])
grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

for img_left, img_right in zip(images_left, images_right):
    imgL = cv2.imread(img_left)
    imgR = cv2.imread(img_right)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv2.findChessboardCorners(grayL, (9,6), None)
    retR, cornersR = cv2.findChessboardCorners(grayR, (9,6), None)

    # If found, add object points, image points (after refining them)
    if retL == True and retR == True:
        objpoints.append(objp)

        corners2L = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        imgpointsL.append(corners2L)

        corners2R = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsR.append(corners2R)

# Calibration for left camera
retL, camera_matrix1, dist_coeffs1, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)

# Calibration for right camera
retR, camera_matrix2, dist_coeffs2, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, grayL.shape[::-1], None, None)

# Stereo calibration
retS, camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR, camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, grayL.shape[::-1],
    criteria = criteria, flags = cv2.CALIB_FIX_INTRINSIC)

# Save the calibration data
output_filename = 'calibration_data.npz'
np.savez(output_filename, K1=camera_matrix1, D1=dist_coeffs1, K2=camera_matrix2, D2=dist_coeffs2, R=R, T=T)

print(f"Calibration data saved to {output_filename}")
