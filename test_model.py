import matplotlib.pyplot as plt
import sys
import os
import PIL
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
caffe_root= '/home/tesfa/sources_tree/caffe/'
os.environ['GLOG_minloglevel'] = '2'
sys.path.insert(0,caffe_root + 'pthon')
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()

def face_detection(imgList):
	cap = cv2.VideoCapture(0)
	img_count = 0
	plt.ion()
	plt.show()
	while(True):
		total_boxes = []
		ret, frame = cap.read()
		#cv2.resize(frame,(160,120))
		cv2.imwrite('tmp.jpg',frame)
		img = Image.open('tmp.jpg')
		net_full_conv = caffe.Net('F8_deploy.prototxt','G3XT_B.caffemodel',caffe.TEST)
		transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})
		im = caffe.io.load_image("tmp.jpg")
		transformer.set_transpose('data', (2, 0, 1))
		transformer.set_raw_scale('data', 255)
		transformer.set_channel_swap('data', (2,1,0))
		net_full_conv.blobs['data'].reshape(50, 3, 160,120)
		transformed_image = transformer.preprocess('data',im)
		net_full_conv.blobs['data'].data[...] = transformed_image
		out = net_full_conv.forward()
				#plt.subplot(1, 3, 1)
				#plt.imshow(transformer.deprocess('data', net_full_conv.blobs['data'].data[0]))
				#plt.subplot(1,3,2)
				#plt.imshow(out['fpool'][0,1])
				#plt.subplot(1,3,3)
				#plt.imshow(out['fpool'][0,0])
				#plt.show()
		boxes = generateBoundingBox(out['fpool'][0, 1], 0.25)
				#print("11")
		plt.subplot(1, 2, 1)
		plt.imshow(transformer.deprocess('data', net_full_conv.blobs['data'].data[0]))
		plt.subplot(1, 2, 2)
		plt.imshow(out['fpool'][0,1])
		plt.draw()
		#plt.show(block=False)
		plt.pause(0.000001)
		print boxes
		if (boxes):
			total_boxes.extend(boxes)

			# boxes_nms = np.array(total_boxes)
			# true_boxes = nms(boxes_nms, overlapThresh=0.3)
			# #display the nmx bounding box in  image.
			# draw = ImageDraw.Draw(scale_img)
			# for box in true_boxes:
			#     draw.rectangle((box[0], box[1], box[2], box[3]) )
			# scale_img.show()

			# nms
		boxes_nms = np.array(total_boxes)
		true_boxes1 = nms_max(boxes_nms, overlapThresh=0.3)
		true_boxes = nms_average(np.array(true_boxes1), overlapThresh=0.07)
		#display the nmx bounding box in  image.
		#draw = ImageDraw.Draw(frame)
		for box in true_boxes:
			if len(box) != 0:
				print("box " + str(box))
				cv2.rectangle(frame,(int(box[0]), int(box[1])), (int(box[2]), int(box[3])),(255,255,255))
				font = cv2.FONT_HERSHEY_SIMPLEX	
				cv2.putText(frame,"frame",(int(box[2]), int(box[3])),font,0.3,(255,255,255) )
		cv2.imwrite("result/" + str(img_count) + ".jpg",frame)
		img_count += 1
		cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

def nms_average(boxes, overlapThresh=0.2):
    result_boxes = []
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(boxes[:, 4])

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # area of i.
        area_i = np.maximum(0, x2[i] - x1[i] + 1) * np.maximum(0, y2[i] - y1[i] + 1)
        area_array = np.zeros(len(idxs) - 1)
        area_array.fill(area_i)
        # compute the ratio of overlap
        # overlap = (w * h) / (area[idxs[:last]]  - w * h + area_array)

        overlap = (w * h) / (area[idxs[:last]])
        delete_idxs = np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        xmin = 10000
        ymin = 10000
        xmax = 0
        ymax = 0
        ave_prob = 0
        width = x2[i] - x1[i] + 1
        height = y2[i] - y1[i] + 1
        for idx in delete_idxs:
            ave_prob += boxes[idxs[idx]][4]
            if (boxes[idxs[idx]][0] < xmin):
                xmin = boxes[idxs[idx]][0]
            if (boxes[idxs[idx]][1] < ymin):
                ymin = boxes[idxs[idx]][1]
            if (boxes[idxs[idx]][2] > xmax):
                xmax = boxes[idxs[idx]][2]
            if (boxes[idxs[idx]][3] > ymax):
                ymax = boxes[idxs[idx]][3]
        if (x1[i] - xmin > 0.1 * width):
            xmin = x1[i] - 0.1 * width
        if (y1[i] - ymin > 0.1 * height):
            ymin = y1[i] - 0.1 * height
        if (xmax - x2[i] > 0.1 * width):
            xmax = x2[i] + 0.1 * width
        if (ymax - y2[i] > 0.1 * height):
            ymax = y2[i] + 0.1 * height
        result_boxes.append([xmin, ymin, xmax, ymax, ave_prob / len(delete_idxs)])
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, delete_idxs)

    # return only the bounding boxes that were picked using the
    # integer data type
    # result = np.delete(boxes[pick],np.where(boxes[pick][:, 4] < 0.9)[0],  axis=0)
    # print boxes[pick]
    return result_boxes


def nms_max(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(boxes[:, 4])

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # area of i.
        area_i = np.maximum(0, x2[i] - x1[i] + 1) * np.maximum(0, y2[i] - y1[i] + 1)
        area_array = np.zeros(len(idxs) - 1)
        area_array.fill(area_i)
        # compute the ratio of overlap
        overlap = (w * h) / (area[idxs[:last]] - w * h + area_array)
        # overlap = (w * h) / (area[idxs[:last]])
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    # result = np.delete(boxes[pick],np.where(boxes[pick][:, 4] < 0.9)[0],  axis=0)
    # print boxes[pick]
    return boxes[pick]

def generateBoundingBox(featureMap, scale):
    boundingBox = []
    stride = 1
    cellSize = 32
    # 227 x 227 cell, stride=32
    for (x, y), prob in np.ndenumerate(featureMap):
        if (prob >= 0.85):
            boundingBox.append(
                [float(stride * y) / scale, float(x * stride) / scale, float(stride * y + cellSize - 1) / scale,
                 float(stride * x + cellSize - 1) / scale, prob])
    # sort by prob, from max to min.
    # boxes = np.array(boundingBox)
    return boundingBox

if __name__ == "__main__":
	face_detection("lfw.txt")
