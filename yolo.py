import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os

FLAGS = []

def show_image(img):
    cv.imshow("Image", img)
    cv.waitKey(0)

def draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels):
    # If there are any detections
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Get the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            
            # Get the unique color for this class
            color = [int(c) for c in COLORS[classids[i]]]

            # Draw the bounding box rectangle and label on the image
            cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
            text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
            cv.putText(img, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img



def generate_boxes_confidences_classids(outs, height, width):
    boxes = []
    confidences = []
    classids = []

    for out in outs:
        for detection in out:
            print (detection)
            
            # Get the scores, classid, and the confidence of the prediction
            scores = detection[:5]
            classid = np.argmax(scores)
            confidence = scores[classid]
            
            # Consider only the predictions that are above a certain confidence level
            if confidence > FLAGS.confidence:
                # TODO Check detection
                box = detection[0:4] * np.array([height, width, height, width])
                (centerX, centerY, bwidth, bheight) = box.astype('int')

                # Using the center x, y coordinates to derive the top
                # and the left corner of the bounding box
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))

                # Append to list
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classids.append(classid)

    return boxes, confidences, classids
                

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-m', '--model-path',
		type=str,
		default='./yolov3-coco/',
		help='The directory where the model weights and \
			  configuration files are.')

	parser.add_argument('-w', '--weights',
		type=str,
		default='./yolov3-coco/yolov3.weights',
		help='Path to the file which contains the weights \
			 	for YOLOv3.')

	parser.add_argument('-cfg', '--config',
		type=str,
		default='./yolov3-coco/yolov3.cfg',
		help='Path to the configuration file for the YOLOv3 model.')

	parser.add_argument('-i', '--image-path',
		type=str,
		help='The path to the image file')

	parser.add_argument('-l', '--labels',
		type=str,
		default='./yolov3-coco/coco-labels',
		help='Path to the file having the \
					labels in a new-line seperated way.')

	parser.add_argument('-c', '--confidence',
		type=float,
		default=0.5,
		help='The model will reject boundaries which has a \
				probabiity less than the confidence value. \
				default: 0.5')

	parser.add_argument('-t', '--threshold',
		type=float,
		default=0.3,
		help='The threshold to use when applying the \
				Non-Max Suppresion')

	parser.add_argument('--download-model',
		type=bool,
		default=False,
		help='Set to True, if the model weights and configurations \
				are not present on your local machine.')

        parser.add_argument('-t', '--show-time',
                type=bool,
                default=False,
                help='Show the time taken to infer each image.')

	FLAGS, unparsed = parser.parse_known_args()

	# Download the YOLOv3 models if needed
	if FLAGS.download_model:
		subprocess.call(['./yolov3-coco/get_model.sh'])

	# Get the labels
	labels = open(FLAGS.labels).read().strip().split('\n')

	# Intializing colors to represent each label uniquely
	colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

	# Load the weights and configutation to form the pretrained YOLOv3 model
	net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

	# Get the output layer names of the model
	layer_names = net.getLayerNames()
	layer_names = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# Do inference with given image
	if not FLAGS.image_path == None:
		# Read the image
		try:
			img = cv.imread(FLAGS.image_path)
                        height, width = img.shape[:2]
		except:
			raise 'Image cannot be loaded!\n\
                               Please check the path provided!'

                finally:
                    # Contructing a blob from the input image
                    blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), 
                                    swapRB=True, crop=False)

                    # Perform a forward pass of the YOLO object detector
                    net.setInput(blob)

                    # Getting the outputs from the output layers
                    start = time.time()
                    outs = net.forward(layer_names)
                    end = time.time()

                    if FLAGS.show_time:
                        print ("[INFO] YOLOv3 took {:6f} seconds".format(end - start))

                    
                    # Generate the boxes, confidences, and classIDs
                    boxes, confidences, classids = generate_boxes_confidences_classids(outs, height, width)
                    
                    # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
                    idxs = cv.dnn.NMSBoxes(boxes, confidence, FLAGS.confidence, FLAGS.threshold)

                    # Draw labels and boxes on the image
                    img = draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors)

                    # show the image
                    show_image(img)

	else:
		pass
