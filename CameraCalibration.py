import cv2
import numpy as np
import glob

# find the chessboard corners
frameSize = (256, 256)
chessboardSize = (1, 1)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare the object points
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1,2)

# arrays to store object points and image points from all the images

objPoints = [] # for 3d point in the real world space
imgPoints = [] # for the 2d points in image plane

# define the path of all the images
images = glob.glob("/home/junaid/alifya/Bean/dataset3D/*.jpg")

# looping
for image in images:
    print(image)

    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # find the chess board corners
    ret, corners = cv2.findCirclesGrid(gray, chessboardSize, None, cv2.CALIB_CB_SYMMETRIC_GRID)

    # if found, add object points, image points (after refining them)
    if ret == True:
        objPoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # draw and display the corners
        output_dir = "/home/junaid/alifya/Bean/corners_output"
        corners_output = cv2.drawChessboardCorners(img, chessboardSize, corners2, ret)
        corners_save = os.path.join(output_dir, os.path.splitext(img)[0] + '_corners.jpg')
        cv2.imwrite(corners_save, corners_output)


# then, perform camera calibration
ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, frameSize, None, None)

print("camera calibrated: ", ret)
print("\ncamera matrix: \n", cameraMatrix)
print("\ndistortion parameter: \n", dist)
print("\nrotation vectors: \n", rvecs)
print("\ntranslation vectors: \n", tvecs)
'''
# perform undistortion to check if the corners are correctly aligned.
img = cv2.imread("/home/junaid/alifya/Bean/corners_output/defect (1297)_corners.jpg")
h, w = img.shape[:2]
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))

# undistort
dst = cv2.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite("calibrationResult.jpg", dst)

# undistort with remapping
mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# crop the images
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite("calibrationResult2.jpg", dst)

# reprojection error
mean_error = 0

for i in range(len(objPoints)):
    imgPoints2, _ = cv2.projectPoints(objPoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv2.norm(imgPoints[i], imgPoints2, cv2.NORM_L2)/len(imgPoints2)
    mean_error+=error

print("\ntotal error: {}".format(mean_error/len(objPoints)))
print("\n\n\n")
'''

