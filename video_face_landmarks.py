'''
This file is to detect human face in video and match landmark on it.
Usage:
python video_face_landmarks.py
--shape-predictor shape_predictor_68_face_landmarks.dat
--videofile ${images/video.mp4}
'''
from imutils.video import FileVideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
from scipy.linalg import logm
from numpy import linalg as LA

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--videofile",
	help="whether or not the video file path should be used")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# read the video
vs = FileVideoStream(args["videofile"]).start()
time.sleep(2.0)

lastShapeCov = []

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	for rect in rects:
		shape = predictor(gray, rect)
		# length of shape is 68 presenting the 68 coordinates of landmarks
		shape = face_utils.shape_to_np(shape)
		# compute C = XX' and distance using LERM between 2 convariance matrics
		if (len(lastShapeCov) !=0 ):
			shapeCov = np.cov(shape.T)
			# Log-Euclidean Riemannian metric (LERM) between two covariance matrices.
			# True geodesic distance induced by Riemannian geometry
			# dis_LERM = dist_LERM(lastShapeCov, shapeCov)
			temp = np.subtract(logm(lastShapeCov),logm(shapeCov))
			dist_LERM = LA.norm(temp, 'fro')
			print (dist_LERM)
			lastShapeCov = shapeCov
		else:
			lastShapeCov = np.cov(shape.T)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), 5)

		# show the frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

# do a bit of cleanup
cv2.destroyAllWindoqs()
vs.stop()