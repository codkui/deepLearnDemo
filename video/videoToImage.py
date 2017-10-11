# -*- coding:utf-8 -*-
#!/usr/bin/env python
import cv2
vc=cv2.VideoCapture("ceshi.mp4")
c=1
if vc.isOpened():
	rval,frame=vc.read()
else:
	rval=False
while rval:
	rval,frame=vc.read()
	cv2.imwrite('image/ceshi_'+str(c)+'.jpg',frame)
	c=c+1
	cv2.waitKey(1)
vc.release()