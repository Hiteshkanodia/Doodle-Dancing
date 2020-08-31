import torch as t
import torchvision as tv
import numpy as np
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
from torchvision import datasets,transforms
from torch.autograd import Variable
cv.namedWindow("image", cv.WINDOW_NORMAL)


model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()

def get_prediction(img,threshold):
	# img = Image.open(img_path)
	transform = transforms.Compose([transforms.ToTensor()])
	img = transform(img)
	pred = model([img])
	pred_score = list(pred[0]['scores'].detach().numpy())
	pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
	# print(len(pred[0]['keypoints']))
	keypoints = pred[0]['keypoints'].squeeze().detach().cpu().numpy()
	pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
	pred_boxes = pred_boxes[:pred_t+1]
	if len(pred[0]['keypoints'])==1:
		keypoints1 = []
		keypoints1.append(keypoints)
		return pred_boxes,keypoints1
	keypoints = keypoints[:pred_t+1]
	return pred_boxes,keypoints

def draw_line(x1,y1,x2,y2,img):
	img = cv.line(img,(x1,y1),(x2,y2),(0,0,0),3)
	return img

def mark_keypoints(keypoints,img1):
	a = np.full(img1.shape,255,dtype=np.uint8)
	img = Image.fromarray(a,"RGB")
	img = np.array(img)
	midx = int((keypoints[5][0] + keypoints[6][0])/2)
	midy = int((keypoints[5][1] + keypoints[6][1])/2)
	img = draw_line(keypoints[0][0],keypoints[0][1],keypoints[1][0],keypoints[1][1],img)
	img = draw_line(keypoints[0][0],keypoints[0][1],keypoints[2][0],keypoints[2][1],img)
	img = draw_line(keypoints[0][0],keypoints[0][1],midx,midy,img)
	img = draw_line(keypoints[5][0],keypoints[5][1],keypoints[6][0],keypoints[6][1],img)
	img = draw_line(keypoints[5][0],keypoints[5][1],keypoints[7][0],keypoints[7][1],img)
	img = draw_line(keypoints[7][0],keypoints[7][1],keypoints[9][0],keypoints[9][1],img)
	img = draw_line(keypoints[6][0],keypoints[6][1],keypoints[8][0],keypoints[8][1],img)
	img = draw_line(keypoints[8][0],keypoints[8][1],keypoints[10][0],keypoints[10][1],img)
	img = draw_line(keypoints[11][0],keypoints[11][1],keypoints[5][0],keypoints[5][1],img)
	img = draw_line(keypoints[12][0],keypoints[12][1],keypoints[6][0],keypoints[6][1],img)
	img = draw_line(keypoints[11][0],keypoints[11][1],keypoints[13][0],keypoints[13][1],img)
	img = draw_line(keypoints[12][0],keypoints[12][1],keypoints[14][0],keypoints[14][1],img)
	img = draw_line(keypoints[13][0],keypoints[13][1],keypoints[15][0],keypoints[15][1],img)
	img = draw_line(keypoints[14][0],keypoints[14][1],keypoints[16][0],keypoints[16][1],img)
	# img = draw_line(keypoints[i][0],keypoints[i][1],keypoints[i][0],keypoints[i][1],img)
	# for i in range(len(keypoints)):
	# 	if keypoints[i][2]==1:
	# 		x = keypoints[i][0]
	# 		y = keypoints[i][1]
	# 		cv.circle(img,(x,y),radius=3,color=(0,0,0),thickness=-1)
	return img

def instance_segmentation_api(img,rect_th=2):
	# cv.imshow('image',img)
	# cv.waitKey(0)
	boxes,keypoints = get_prediction(img,0.5)
	# print(boxes)
	# print(keypoints)
	# img  = cv.imread(img_path)
	for i in range(len(boxes)):
		img = mark_keypoints(keypoints[i],img)
		cv.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
		# cv.resizeWindow('image', 900,900)
		# cv.imshow('image',img)
		# cv.waitKey(0)
	return img

# instance_segmentation_api('a2.jpg')

def load_vedio():
	cap = cv.VideoCapture('outpy.avi')
	if (cap.isOpened()== False):
		print("Error opening video stream or file")
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	out = cv.VideoWriter('outpy1.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
	while(cap.isOpened()):
		ret,frame = cap.read()
		if ret==True:
			im1 = instance_segmentation_api(frame)
			out.write(im1)
			cv.imshow('image',im1)
			if cv.waitKey(25) & 0xFF == ord('q'):
				break
		else:
			break
	cap.release()
	cv.destroyAllWindows()

load_vedio()