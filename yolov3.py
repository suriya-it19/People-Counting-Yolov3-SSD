import argparse
import os
import time
from itertools import zip_longest
from pathlib import Path

import cv2
import dlib
import imutils
import numpy as np
import schedule
from imutils.video import FPS, VideoStream
from numpy import random

from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-w", "--weights", required=True,
	help="path to weights file")
ap.add_argument("-f", "--config", required=True,
	help="path to config file")
ap.add_argument("-z", "--size", required=True, type=int,
	help="image size")
ap.add_argument("-n", "--names", default='weights/coco.names',
	help="coco names file")
ap.add_argument("-c", "--confidence", type=float, default=0.6,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
	help="# of skip frames between detections")
args = vars(ap.parse_args())

writer = None

W = None
H = None

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

totalFrames = 0
totalDown = 0
totalUp = 0
x = []
empty=[]
empty1=[]

labelsPath = args['names']
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

weightsPath = args['weights']
configPath = args['config']
print("[INFO] loading YOLO from disk...")

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

if not args.get("input", False):
	url = ''
	print("[INFO] Starting the live stream..")
	vs = VideoStream(url).start()
	time.sleep(2.0)

else:
	print("[INFO] Starting the video..")
	vs = cv2.VideoCapture(args["input"])

#writer = None
#(W, H) = (None, None)
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

while True:
	(grabbed, frame) = vs.read()
	if not grabbed:
		break
	#frame = frame[1] if args.get("input", False) else frame
	if args["input"] is not None and frame is None:
		break

	frame = imutils.resize(frame, height= 500,width = 500)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	print(H, W)

	status = "Waiting"
	rects = []

	if totalFrames % args["skip_frames"] == 0:
		status = "Detecting"
		trackers = []

		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (args['size'], args['size']),
			swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()

		boxes = []
		confidences = []
		classIDs = []

		for output in layerOutputs:
			for detection in output:

				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				if confidence > args["confidence"]:
					if LABELS[classID] != 'person':
						continue

					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

					idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
						args["threshold"])
					if len(idxs) > 0:
						for i in idxs.flatten():
							(x, y) = (boxes[i][0], boxes[i][1])
							(w, h) = (boxes[i][2], boxes[i][3])

					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(x, y, x + w, y + h)
					tracker.start_track(rgb, rect)

					trackers.append(tracker)
	else:
		for tracker in trackers:
			status = "Tracking"

			tracker.update(rgb)
			pos = tracker.get_position()

			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			rects.append((startX, startY, endX, endY))

	cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
	cv2.putText(frame, "-Prediction border - Entrance-", (0, H - ((20 * 20) + 200)),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

	objects = ct.update(rects)

	for (objectID, centroid) in objects.items():
		to = trackableObjects.get(objectID, None)

		if to is None:
			to = TrackableObject(objectID, centroid)

		else:
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)

			if not to.counted:
				if direction < 0 and centroid[1] < H // 2:
					totalUp += 1
					empty.append(totalUp)
					to.counted = True

				elif direction > 0 and centroid[1] > H // 2:
					totalDown += 1
					empty1.append(totalDown)
					x = []
					x.append(len(empty1)-len(empty))

					to.counted = True


		trackableObjects[objectID] = to

		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

	info = [
	("Exit", totalUp),
	("Enter", totalDown),
	("Status", status),
	]

	info2 = [
	("Total people inside", x),
	]

	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

	#for (i, (k, v)) in enumerate(info2):
	#	text = "{}: {}".format(k, v)
	#	cv2.putText(frame, text, (180, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])
	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

	totalFrames += 1
cv2.destroyAllWindows()

print("[INFO] cleaning up...")
vs.release()
