#!/usr/bin/env python
# hand_project.py - Python script demonstrates hand signs detection
# and hand tracking. Hopefully this script can help in research and 
# development of assistive technology for Autistic and Deaf-mute 
# people.
#
# Principle: Hand geometry model rather than image set trained one, 
# is projected and correlated with sample Image. Natively 
# accelerated by graphics card without direct programming.
#
# Depends on: OpenCV, OpenGL, PyGame and NumPy. Requires PC machine
# with camera installed.
#
# Author: Milan Neskovic, 2020, neskomi@gmail.com
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
import numpy as np
import cv2
from math import *
from imutils import rotate_bound
import itertools
from copy import deepcopy
import sys, time

class Hand:
	def __init__(self, size, lenRatio = 1.1, widthRatio = 0.85, thickRatio = 0.95, color = [150,100,120]):
		self.size = size
		self.lenRatio = lenRatio
		self.widthRatio = widthRatio
		self.thickRatio = thickRatio
		self.colors = [color,color,color,color,color,color]
		
		self.fingerNames = {'ALL':0, 'THUMB':1, 'POINTER':2, 'MIDDLE':3, 'RING':4, 'LITTLE':5}
		self.find=1500
		self.flist={}
		self.plist={}
		self.listFlag=0
		
	def DrawCylinder(self, radius, height, num_slices, rt=1):
		r = radius
		h = height
		n = float(num_slices)
			
		glTranslatef(0.0, 0.0, -h/2.0)
		glPushMatrix()
		qobj = gluNewQuadric()
		gluQuadricOrientation(qobj, GLU_OUTSIDE )
		gluQuadricNormals(qobj, GLU_SMOOTH )
		glTranslatef(0.0, 0.0, -h/2.0)
		gluCylinder(qobj, r*rt, r, h, 20, 20);
		glPopMatrix()

		curColor =glGetFloatv(GL_CURRENT_COLOR);
		glPushMatrix()

		glTranslatef(0.0, 0.0, -h/2.0)
		gluSphere(qobj, r*rt*1.02, 20, 20)
		glPopMatrix()
		glPushMatrix()
		glTranslatef(0.0, 0.0, h/2.0)
		glPushAttrib(GL_CURRENT_BIT)
		gluSphere(qobj, r*1.03, 20, 20)
		glPopAttrib()
		glPopMatrix()
		
	def DrawComplex(self, w1, h1, w2, h2, length, num_slices):
		r = h1/2
		rd=h2/2
		w = (w1-2*r)/2
		wd=(w2-2*rd)/2
		n = float(num_slices)

		dpts = []
		for i in range(int(n)+1):
			angle = 2 * pi * (i/n)
			x = r * cos(angle) 
			y = r * sin(angle) + (w if i<=int(n/2) else -w)
			xd=rd * cos(angle) 
			yd = rd * sin(angle) + (wd if i<=int(n/2) else -wd)
			pt = (x, y, xd, yd)
			dpts.append(pt)
		dpts.append(dpts[0])
		
		glPushMatrix()
		glBegin(GL_TRIANGLE_STRIP)

		for (x, y, xd, yd) in dpts:
			z = length/2.0
			glVertex(x, y, -z)
			glVertex(xd, yd, z)
		glEnd()
		glPopMatrix()
		
	def DrawFinger(self, bend, ang, len, baseRad, thickTatio):
		rad=thickTatio*baseRad/2.0
		if self.listFlag:
			glPushMatrix()
			qobj = gluNewQuadric()
			gluQuadricOrientation(qobj, GLU_OUTSIDE )
			gluQuadricNormals(qobj, GLU_SMOOTH )
			gluSphere(qobj, baseRad/2, 20, 20)

			glRotatef(90+ang,1.0, 0.0, 0.0);
			glRotatef(bend[0]+bend[1],0.0, 10.0, 0.0);
			self.DrawCylinder(rad, len*0.398, 20,thickTatio)
			glTranslatef(0.0, 0.0, -len*0.398/2.0)
			glRotatef(bend[0]+bend[2],0.0, 10.0, 0.0);

			rad*=thickTatio
			self.DrawCylinder(rad, len*0.224, 20,thickTatio)
			glTranslatef(0.0, 0.0, -len*0.224*0.8/2.0)
			glRotatef(bend[0]+bend[3],0.0, 10.0, 0.0);

			rad*=thickTatio
			self.DrawCylinder(rad, len*0.158, 20, thickTatio)
			glPopMatrix();
			return

		key=str(bend)+str(ang)+str(len)+str(rad)
		if key in self.flist:
			glCallList(self.flist[key][1])
		else:
			self.find+=1
			gList = glGenLists(self.find)
			glNewList(gList, GL_COMPILE)
			glPushMatrix()
			qobj = gluNewQuadric()
			gluQuadricOrientation(qobj, GLU_OUTSIDE )
			gluQuadricNormals(qobj, GLU_SMOOTH )
			gluSphere(qobj, baseRad/2, 20, 20)

			glRotatef(90+ang,1.0, 0.0, 0.0);
			glRotatef(bend[0]+bend[1],0.0, 10.0, 0.0);
			self.DrawCylinder(rad, len*0.398, 20,thickTatio)
			glTranslatef(0.0, 0.0, -len*0.398/2.0)
			glRotatef(bend[0]+bend[2],0.0, 10.0, 0.0);

			rad*=thickTatio
			self.DrawCylinder(rad, len*0.224, 20,thickTatio)
			glTranslatef(0.0, 0.0, -len*0.224*0.8/2.0)
			glRotatef(bend[0]+bend[3],0.0, 10.0, 0.0);

			rad*=thickTatio
			self.DrawCylinder(rad, len*0.158, 20, thickTatio)
			glPopMatrix();
			glEndList()
			self.flist[key]=(self.find,gList)
			glCallList(gList)

	def DrawPalm(self, len,width):
		glPushMatrix()
		glRotatef(90,1.0, 0.0, 0);
		glTranslatef(0.0, -0.5*width, 0.4*len)
		w1=width
		h1 = 0.25*width
		w2=1.0*width
		h2=1.25*h1

		w = (w1-h1)/2
		wd=(w2-h2)/2
		plen=len*0.8
		self.DrawComplex(w1, h1, w2, h2, plen, 20)
		glPushMatrix()
		glRotatef(90,1.0, 0.0, 0.0);
		glTranslatef(0.0, -plen/2, w)
		glPopMatrix()

		rlen=len-plen
		glTranslatef(0.0, 0.0, plen/2+rlen/2)
		self.DrawComplex(w2, h2, w2*0.6, h2*1.3, rlen, 20)

		glPopMatrix()
		
	def Draw(self, gest, size = None, lenRatio=None, widthRatio=None, thickRatio=None, colors=None):
		ang=gest[0][0]
		bend=gest[1][0][0]
		bgest=np.array(gest[1])
		lRatio=self.lenRatio if lenRatio is None else lenRatio
		sz=self.size if size is None else size
		plen=self.size/(lRatio+1)
		flen=lRatio*plen
		wRatio=self.widthRatio if widthRatio is None else widthRatio
		pwidth=wRatio*plen
		thRatio=self.thickRatio if thickRatio is None else thickRatio
		frad=pwidth/4.0
		fdist = pwidth/4.0
		partColor=np.array(self.colors)/255 if thickRatio is None else np.array(colors)/255
		glColor(partColor[0])
		if self.listFlag:
			self.DrawPalm(plen,pwidth)
		else:
			key=str(plen)+str(frad)
			if key in self.plist:
				glCallList(self.plist[key][1])
			else:
				self.find+=1
				gList = glGenLists(self.find)
				glNewList(gList, GL_COMPILE)
				self.DrawPalm(plen,pwidth)
				glEndList()
				self.plist[key]=(self.find,gList)
				glCallList(gList)
				
		glPushMatrix()
		glTranslatef(0.0, 0.0, -fdist/2)
		glColor(partColor[2])
		self.DrawFinger(bgest[0]+bgest[2], ang+gest[0][2], flen*0.95, frad*0.9, thRatio)
		glTranslatef(0.0, 0.0, -fdist)
		glColor(partColor[3])
		self.DrawFinger(bgest[0]+bgest[3], ang/3+gest[0][3], flen*1.1, frad, thRatio)
		glTranslatef(0.0, 0.0, -fdist)
		glColor(partColor[4])
		self.DrawFinger(bgest[0]+bgest[4], -ang/3+gest[0][4], flen*1.05, frad*0.9, thRatio)
		glTranslatef(0.0, 0.0, -fdist)
		glColor(partColor[5])
		self.DrawFinger(bgest[0]+bgest[5], -ang+gest[0][5], flen*0.88, frad*0.8, thRatio)
		glPopMatrix()
		glPushMatrix()

		glTranslatef(-frad/2.0, -(plen*0.85), -frad)
		glRotatef(-70,0.0, 1.0, 0);
		glRotatef(30,1.0, 0.0, 0.0);
		glRotatef(-60,0.0, 0.0, 1.0);
		glColor(partColor[1])
		self.DrawFinger(bgest[0]+bgest[1], ang+gest[0][1], flen*1.1, frad*1.1, thRatio)
		glPopMatrix()
		
	def GetList(self, gest, ind=1):
		genList = glGenLists(ind)
		glNewList(genList, GL_COMPILE)
		self.listFlag=1
		self.Draw(gest)
		self.listFlag=0
		glEndList()
		return genList 

	def RecogniseGesture(self, gest):
		bend = gest[1][:,0] #+ gest[1][0,0]
		#bend[1:] = bend[1:] + gest[1][0,0] # add offset for all
		f=self.fingerNames
		print("bend ",bend)
		if bend[f['THUMB']]<30 and bend[f['POINTER']]<20 and bend[f['MIDDLE']]<20 and bend[f['RING']]<20 and bend[f['LITTLE']]<20:
			return 'Five'
		elif bend[f['THUMB']]<30 and bend[f['POINTER']]<20 and bend[f['MIDDLE']]<20 and bend[f['RING']]>30 and bend[f['LITTLE']]>30:
			return 'Three'
		elif bend[f['THUMB']]<30 and bend[f['POINTER']]<20 and bend[f['MIDDLE']]>30 and bend[f['RING']]>30 and bend[f['LITTLE']]>30:
			return 'Peace'
		elif bend[f['THUMB']]>12 and bend[f['POINTER']]>30 and bend[f['MIDDLE']]>30 and bend[f['RING']]>30 and bend[f['LITTLE']]>30:
			return 'Closed'
		else:
			return ''
	
def GetColor(im, mask):
	histColor=np.array([0,0,0],np.uint8)
	for i in range(0,3):
		hist = cv2.calcHist([im],[i],mask,[256],[1,255])
		histColor[i]=np.argmax(hist)
	return histColor
	
def GetMeanColor(im, mask):
	masked = np.ma.masked_where(mask, im)
	return masked.mean()
	
def ConvertYCrCb2BGR(color):
	ycrcb=np.array([[color,color],[color,color]])#np.array((2,2,3),np.uint8)
	bgr=cv2.cvtColor(ycrcb,cv2.COLOR_YCrCb2BGR)
	return bgr[0,0]
	
def GetGradient(graysrc):
    # gradient X
    gradx = cv2.Sobel(graysrc, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0)
    gradx = cv2.convertScaleAbs(gradx)
    # gradient Y
    grady = cv2.Sobel(graysrc, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0)
    grady = cv2.convertScaleAbs(grady)

    return cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
	
def OverlayImage(background, overlay, x, y, transp=0.1):
	rows,cols,channels = overlay.shape
	bckg=background[int(y):int(y+rows), int(x):int(x+cols)]
	try:
		rows,cols,channels = bckg.shape
	except:
		rows,cols = bckg.shape
	ovrl=overlay[0:rows,0:cols]
	overlay = cv2.addWeighted(bckg,1.0-transp,ovrl,transp,0)
	bckgWidth = background.shape[1]
	bckgHeight = background.shape[0]

	if overlay is None: return
	if x >= bckgWidth or y >= bckgHeight:
		return background
	h, w = overlay.shape[0], overlay.shape[1]
	if x + w > bckgWidth:
		w = bckgWidth - x
		overlay = overlay[:, :w]
	if y + h > bckgHeight:
		h = bckgHeight - y
		overlay = overlay[:h]
	if overlay.shape[2] < 4:
		overlay = np.concatenate(
			[ 	overlay,
				np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255 ],
			axis = 2)

	overlay_image = overlay[..., :3]
	mask = overlay[..., 3:] / 255.0
	background[int(y):int(y+h), int(x):int(x+w)] = (1.0 - mask) * background[int(y):int(y+h), int(x):int(x+w)] + mask * overlay_image

	return background
		
def RemoveBackground(im, rect):
	(xg,yg,wg,hg)=rect
	if wg<4 or hg<4: return im
	sourceImage=im[int(yg):int(yg+hg),int(xg):int(xg+wg)]
	imageYCrCb = cv2.cvtColor(sourceImage,cv2.COLOR_BGR2YCR_CB)
	color=imageYCrCb[int(hg/2),int(wg/2)]
	avg_color_per_row = np.average(imageYCrCb[int(3*hg/7):int(4*hg/7),int(3*wg/7):int(4*wg/7)], axis=0)
	avg_color = np.average(avg_color_per_row, axis=0)
	min_YCrCb = avg_color-np.array([15,12,10],np.int16)#np.array([0,133,77],np.uint8)
	max_YCrCb = avg_color+np.array([15,12,10],np.int16)#np.array([255,173,127],np.uint8)
	skinRegion = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
	skin = cv2.bitwise_and(sourceImage, sourceImage, mask = skinRegion)
	skinf = cv2.GaussianBlur(skin, (9, 9), 0)
	ret = np.zeros(im.shape,dtype=np.uint8)
	ret[int(yg):int(yg+hg),int(xg):int(xg+wg)]=skinf
	return ret

def RemoveBackground2(sourceImage,srcIm,col):
	min_YCrCb = col-np.array([20,4,8],np.int16)#np.array([0,133,77],np.uint8)
	max_YCrCb = col+np.array([20,4,8],np.int16)#np.array([255,173,127],np.uint8)
	skinRegion = cv2.inRange(srcIm,min_YCrCb,max_YCrCb)
	skin = cv2.bitwise_and(sourceImage, sourceImage, mask = skinRegion)
	return skin

def GetAverageColor(im, rect):
	(xg,yg,wg,hg)=rect
	sourceImage=im[int(yg):int(yg+hg),int(xg):int(xg+wg)]
	#imageYCrCb = cv2.cvtColor(sourceImage,cv2.COLOR_BGR2YCR_CB)
	#color=imageYCrCb[int(hg/2),int(wg/2)]
	h1=int(3*hg/7)
	h2=int(4*hg/7)
	w1=int(3*wg/7)
	w2=int(4*wg/7)
	reg = sourceImage[h1:h2,w1:w2]
	h,w,c=reg.shape
	if h>0 and w>0:
		avg_color_per_row = np.average(reg, axis=0)
		avg_color = np.flip(np.average(avg_color_per_row, axis=0))
	else:
		avg_color = np.array([0,0,0])
	reg = sourceImage[0:2,:]
	h,w,c=reg.shape
	if h>0 and w>0:
		bck_color_per_row = np.average(reg, axis=0)
		bck_color = np.flip(np.average(bck_color_per_row, axis=0))
	else:
		bck_color = np.array([0,0,0])

	return avg_color, bck_color
	
def GetBounds(im,tresh):
	rt,thr = cv2.threshold(im,tresh,255,cv2.THRESH_BINARY)
	#ctrs, hrr = cv2.findContours(thr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	p = np.nonzero(thr)
	if len(p[0])==0 or len(p[1])==0: return (0,0,0,0)
	x = p[1].min()
	y = p[0].min()
	w = p[1].max()-x
	h = p[0].max()-y
	return np.array([x,y,w,h])#cv2.boundingRect(ctrs[0])
	
def GetReshaped(im, rect, d):
	x,y,w,h=rect
	hg,wg=im.shape[0:2]
	(xr1,yr1,xr2,yr2)=(x-(d-1)*w, y-(d-1)*h, x+d*w, y+d*h)
	y1=max(0,int(round(yr1)))
	x1=max(0,int(round(xr1)))
	y2=min(int(round(yr2)),hg)
	x2=min(int(round(xr2)),wg)
	roi=im[y1:y2,x1:x2]
	return roi, np.array([x1,y1,x2-x1,y2-y1])
	
def GetGradPyramid(im, rec, d, scale=2):
	grad=GetGradient(im)

	grad = cv2.GaussianBlur(grad, (11, 11), 0)
	roi,rect = GetReshaped(grad, rec, d)
	
	temp = cv2.pyrUp(roi)
	#im = cv2.resize(im, (64, 64)) 
	pyrs=[temp]
	pyrs.append(roi)
	temp = roi
	for i in range(scale):
		temp = cv2.pyrDown(temp)
		pyrs.append(temp)

	return pyrs, rect*2
	
def GetPyramid(im, rec, d, scale=2):
	roi,rect = GetReshaped(im, rec, d)
	roi= cv2.cvtColor(roi,cv2.COLOR_BGR2YCR_CB)
	temp = cv2.pyrUp(roi)
	pyrs=[temp]
	pyrs.append(roi)
	temp = roi
	for i in range(scale):
		temp = cv2.pyrDown(temp)
		pyrs.append(temp)
		
	return pyrs, rect*2
	
def SpaceTransform(space, width, height):
	(rx,ry,rz,sz)=space
	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	gluPerspective(30, float(width)/height, 1, 500)#gluPerspective(90,1,0.01,1000)
	gluLookAt(0,0,sz,0,0,0,0,1,0)

	glEnable(GL_DEPTH_TEST)
	#glEnable(GL_LIGHTING)
	glEnable(GL_LIGHT0)
	glEnable(GL_COLOR_MATERIAL)
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE )

	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

	glMatrixMode(GL_MODELVIEW)
	glClearColor(0.0, 0.0, 0.0, 1.0)
	#glColor(handColor[2]/255, handColor[1]/255, handColor[0]/255)
	glLoadIdentity()

	#glLightfv(GL_LIGHT0,GL_POSITION,[10,0,0]);
	#glLight(GL_LIGHT0, GL_POSITION,  (0, 0, -5, 1)) # point light from the left, top, front
	glLightfv(GL_LIGHT0, GL_AMBIENT, (0.9, 0.9, 0.9, 1))
	#glLightfv(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))
	glRotate(ry, 0, 1, 0)#NOTE: this is applied BEFORE the translation due to OpenGL multiply order
	glRotate(rx, 1, 0, 0)
	glRotate(rz, 0, 0, 1)
	
class Projector:
	def __init__(self, hand, width, height, scale=2):
		self.width = width
		self.height = height
		self.bckColor=[0,0,0]
		self.hand = hand
		self.handList=[]
		self.mn=0
		self.mna=0
		self.maxi=0
		self.scale=scale
		
		for a in range(0,45):
			blist=[]
			for b in range(0,60):
				gest=(np.zeros(6),np.zeros((6,4)))
				gest[0][0]=a
				gest[1][0][0]=b
				blist.append(self.hand.GetList(gest,a*60+b+1))
			self.handList.append(blist)
			
		start = time.time()
		self.prj,self.ims,self.refPoints=self.GetProjections(self.handList)
		self.state=self.prj[0]
		self.ml=(0,0)
		end = time.time()
		print('Build time:',end-start,len(self.prj))
	
	def GetProjection(self, hlist,space,gest=None, ax=0):
		if hlist:
			SpaceTransform(space, self.width, self.height)
			glCallList(hlist)
		else:
			glClearColor(0.0, 0.0, 0.0, 1.0 )
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			self.hand.Draw(gest)

		buffer = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
		im = Image.frombuffer(mode="RGB", size=(self.width, self.height), data=buffer)
		pix = np.asarray(im)
		gray = cv2.cvtColor(pix, cv2.COLOR_RGB2GRAY)
		rt,thr = cv2.threshold(gray,2,255,cv2.THRESH_BINARY)

		pix2=np.copy(pix)
		pix2[thr==0]=self.bckColor
		gray[thr>0]=255

		bnds = GetBounds(gray, 2)
		if bnds[2]==0: return None, None, gray, None
		pyrs, rect = GetGradPyramid(gray, bnds, 1.05, self.scale)
		cpyrs, rect = GetPyramid(pix2, bnds, 1.05, self.scale)
		
		if ax:
			rm = GetBounds(pix[:,:,1], 250) * 2
		else:
			rm = rect

		return pyrs, cpyrs, gray, rm#pix[int(y/d):int((y+h)*d),int(x/d):int((x+w)*d)].copy()
		
	def GetTemplateSet(self, prjSet):
		ret=[]
		(xmin,ymin,xmax,ymax)=(self.width, self.height,0,0)
		rd=[]
		for prj in prjSet:
			if isinstance(prj[0][2], tuple):
				f=prj[0][2][0]
			else:
				f=prj[0][2]
			gest=prj[1][1]
			glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
			self.hand.Draw(gest)
			buffer = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
			pix = np.asarray(Image.frombuffer(mode="RGB", size=(self.width, self.height), data=buffer))

			gray = cv2.cvtColor(pix, cv2.COLOR_RGB2GRAY)
			rt,thr = cv2.threshold(gray,2,255,cv2.THRESH_BINARY)
			pix2=pix.copy()
			finger=pix.copy()
			fmask=(pix[:,:,0]>0)
			finger[fmask]=[0,0,0]
			gray[thr>0]=255

			pix2[thr==0]=np.flip(self.bckColor)
			p = np.nonzero(thr)
			if len(p[0])>0 and len(p[1])>0: 
				(x1,y1,x2,y2) = (p[1].min(),p[0].min(),p[1].max(),p[0].max())
				if x1<xmin: xmin=x1
				if y1<ymin: ymin=y1
				if x2>xmax: xmax=x2
				if y2>ymax: ymax=y2
			rd.append([prj,gray,pix2, finger])
		if xmax==0 or ymax==0: return []

		for r in rd:
			r[1], rect = GetGradPyramid(r[1], (xmin,ymin,xmax-xmin,ymax-ymin), d=1.05, scale=self.scale)
			r[2], rect = GetPyramid(r[2], (xmin,ymin,xmax-xmin,ymax-ymin), d=1.05, scale=self.scale)
			r[3],frect = GetReshaped(r[3], (xmin,ymin,xmax-xmin,ymax-ymin), d=1.05)
			r.append(rect)

		return rd
		
	def GetTemplateSetSpace(self, prjSet):
		ret=[]
		(xmin,ymin,xmax,ymax)=(self.width, self.height,0,0)
		rd=[]
		hlist=self.hand.GetList(prjSet[0][1][1],ind=1)
		for prj in prjSet:
			gest=prj[1][1]
			space=prj[1][0]

			SpaceTransform(space, self.width, self.height)
			glCallList(hlist)
			buffer = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
			pix = np.asarray(Image.frombuffer(mode="RGB", size=(self.width, self.height), data=buffer))
			gray = cv2.cvtColor(pix, cv2.COLOR_RGB2GRAY)
			rt,thr = cv2.threshold(gray,2,255,cv2.THRESH_BINARY)
			pix2=pix.copy()
			finger=pix.copy()
			fmask=(pix[:,:,0]>0)
			finger[fmask]=[0,0,0]
			gray[thr>0]=255
			pix2[thr==0]=np.flip(self.bckColor)
			p = np.nonzero(thr)
			if len(p[0])>0 and len(p[1])>0: 
				(x1,y1,x2,y2) = (p[1].min(),p[0].min(),p[1].max(),p[0].max())
				if x1<xmin: xmin=x1
				if y1<ymin: ymin=y1
				if x2>xmax: xmax=x2
				if y2>ymax: ymax=y2
			rd.append([prj,gray,pix2, finger])
		if xmax==0 or ymax==0: return []

		for r in rd:
			r[1], rect = GetGradPyramid(r[1], (xmin,ymin,xmax-xmin,ymax-ymin), d=1.05, scale=self.scale)
			r[2], rect = GetPyramid(r[2], (xmin,ymin,xmax-xmin,ymax-ymin), d=1.05, scale=self.scale)
			r[3],frect = GetReshaped(r[3], (xmin,ymin,xmax-xmin,ymax-ymin), d=1.05)
			r.append(rect)

		glDeleteLists(hlist,1)
		return rd
		
	def GetBaseProjections(self, hlist):
		prj=[]
		ims=[]
		im=np.zeros((900,1800,3), np.uint8)
		roty=[-45,-90,-120,-135]#[-90,-45]#[-90,-45,-135]#[-90,-22,-45,-68,-135]#
		rotx=[-45,0,-22,45]#[37,10,64]#
		rotz=[0]
		sclz=[-300]#,-500,-550,-600,-750,-650,-700,-800,-850,-950,-1100,-900,-1000,-1200,-1400]
		strf=[(0,-90,0,-260),(0,-45,0,-260)]
		ang=[0,20]
		bend=[0,40]
		i=0
		for a,b in itertools.product(ang,bend):
			self.plist=hlist[a][b]
			j=0
			for (rx,ry,rz,sz) in strf:
				start = time.time()
				pyrs,cpyrs,cim,m=GetProjection(self.plist,(rx,ry,rz,sz))
				end = time.time()
				#print('prj build time:',end-start, len(prj))
				prj.append([rx,ry,rz,sz,a,b,0])
				ims.append(pyrs)
				OverlayImage(im,cv2.cvtColor(pyrs[0], cv2.COLOR_GRAY2BGR),i*300,j*300, 0.9)
				j+=1
			i+=1
		return prj,ims,im
		
	def GetProjections(self, hlist):
		prj=[]
		ims=[]
		refPoints=[]
		roty=[-90,-100,-80]#,-70,-45,-20,-110,-135]#[-90,-45]#[-90,-45,-135]#[-90,-22,-45,-68,-135]#
		rotx=[45,34-23,-15,-10,0,10,15,23,34,45,64]#[37,10,64]#
		rotz=[0]
		sclz=[-180,-220,-210,-250,-300,-360,-430]#,-500,-550,-600,-750,-650,-700,-800,-850,-950,-1100,-900,-1000,-1200,-1400]
		ang=[5,0,10,20]
		bend=[0,20]#,45]
		gest=(np.zeros(6),np.zeros((6,4)))
		for a,b in itertools.product(ang,bend):
			plist=hlist[a][b]
			for ry,rz,sz in itertools.product(roty,rotz,sclz):#for rx,ry,rz,sz in zip(rotx,roty,rotz,sclz):
				gest[0][0]=a
				gest[1][0][0]=b
				for rx in rotx:
					start = time.time()
					pyrs,cpyrs,cim,m=self.GetProjection(plist,(rx,ry,rz,sz))
					end = time.time()
					im0 = rotate_bound(pyrs[0], rx-rotx[0])
					im1 = im0
					im2 = rotate_bound(pyrs[2], rx-rotx[0])
					ims.append(pyrs)#ims.append([im0,im1,im2])
					prj.append(deepcopy([(rx,ry,rz,sz),gest]))
					refPoints.append(deepcopy(m))
		return prj,ims,refPoints
		
	def Project(self, im, templ, pos,scale,ext=1.5,ptype=cv2.TM_CCOEFF_NORMED):
		if pos is None:
			return 0, None, None,None, None

		if len(templ[scale].shape) > 2:
			h,w,c=templ[scale].shape[0:3]
		else:
			h,w=templ[scale].shape[0:2]
			c=1

		roi,(xr,yr,wr,hr) = GetReshaped(im[scale], (pos[0],pos[1],w,h), ext)
		if hr<h or wr<w:#if (y2-y1)<=h or (x2-x1)<=w:
			return 0, None, None,None, None
		res = cv2.matchTemplate(roi,templ[scale],ptype)
		minv, maxv, minl, maxl = cv2.minMaxLoc(res)
		max_loc=(maxl[0]+xr,maxl[1]+yr)
		ret = np.zeros((hr,res.shape[1]+wr,3), 'uint8')
		if c==1:
			rim=cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
			ret[0:hr,0:wr]=rim
		else:
			ret[0:hr,0:wr]=cv2.cvtColor(roi,cv2.COLOR_YCrCb2BGR)
		cv2.rectangle(ret, (maxl[0], maxl[1]), (int(maxl[0]+w),int(maxl[1]+h)), 150, 1)
		ret[0:res.shape[0],wr:]=cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
		return maxv, max_loc, ret, minv,minl
		
	def ProjectSet(self,tempSet):
		mxn=-1000000000
		ml=deepcopy(self.pos)
		mref=tempSet[0][4][0:2]
		self.res=[[0],[0]]
		mxpim=deepcopy(self.templ)
		mxcpim=deepcopy(self.colTempl)
		mxpt=self.mxpt if hasattr(self,"mxpt") and self.mxpt is not None else self.pos+tempSet[0][4][0:2]/(2**self.scale)
		mxprj=self.state
		mxref=deepcopy(tempSet[0][4])
		mxloc=self.pos
		mfinger=deepcopy(tempSet[0][3])
		ref=deepcopy(tempSet[0][4])

		for t in tempSet:
			pim=t[1]
			cpim=t[2]
			tf=t[0][1]
			pt=self.pos+t[4][0:2]/(2**self.scale)
			max_val, max_loc, res,min_val,min_loc = self.Project(self.cframes, cpim, pt, self.scale, 1.1, cv2.TM_CCOEFF_NORMED)
			max_val2, max_loc2, res2,min_val2,min_loc2 = self.Project(self.pyrs, pim, pt, self.scale,1.1)
			max_val2 = max_val2*45/((-tf[0][3])**(0.5))

			if not max_val:
				continue
			max_val = max_val*45/((-tf[0][3])**(0.5))
			if max_val>mxn and max_val2 >=(self.mn*0.9):
				mxn=max_val
				mxloc=deepcopy(max_loc-t[4][0:2]/(2**self.scale))
				self.res=res.copy()
				mxprj=deepcopy(tf)
				mxpim=deepcopy(pim)
				mxcpim=deepcopy(cpim)
				mxref=deepcopy(t[4])
				mxpt=deepcopy(pt)
				mfinger=deepcopy(t[3])
		#cv2.rectangle(frame,(int(rectm[0]),int(rectm[1])), (int(rectm[0]+rectm[2]),int(rectm[1]+rectm[3])), 150, 1)
		max_val, max_loc, res,min_val,min_loc = self.Project(self.pyrs, mxpim, mxpt, self.scale, 1.1, cv2.TM_CCOEFF_NORMED)
		max_val = max_val*45/((-self.state[0][3])**(0.5))
		#if max_val>mx:
		self.state=mxprj
		self.pos=mxloc
		self.mn=max_val
		self.templ=mxpim
		self.colTempl=deepcopy(mxcpim)
		self.ref=mxref
		self.mxpt = mxpt

	def ProjectSpaceSet(self,ax, delta, limit=60):
		mp=deepcopy(self.state)
		prjSet=[]
		roty=[mp[0][1]-2,mp[0][1]+2]
		rotx=[mp[0][0]-2,mp[0][0]+2]
		rotz=[mp[0][2]-2,mp[0][2]+2]
		sclz=[mp[0][3]*0.95,mp[0][3],mp[0][3]/0.95]
		for rx,ry,rz,sz in itertools.product(rotx,roty,rotz,sclz):
			if sz > -180 or sz <= -440:
				continue
			tf=deepcopy(self.state)
			tf[0]=deepcopy((rx,ry,rz,sz))
			prjSet.append((ax,tf))
		if len(prjSet)==0: 
			self.res=[[0],[0]]
			self.ref=(0,0)
		tempSet=GetTemplateSetSpace(prjSet)
		return self.ProjectSet(tempSet)

	def ProjectSpaceAxis(self, ax, delta, limitLow=None, limitHigh=None):
		spaceState=deepcopy(self.state[0])
		prjSet=[]
		th = spaceState[ax]+delta
		if (limitHigh is not None) and th > limitHigh: th = limitHigh
		tl = spaceState[ax]-delta
		if (limitLow is not None) and tl < limitLow: tl = limitLow
		trf=[tl, spaceState[ax], th]
		for t in trf:
			tf=deepcopy(self.state)
			d=np.array(tf[0])
			d[ax]=t
			tf[0]=deepcopy(d)
			prjSet.append((ax,tf))

		tempSet=self.GetTemplateSetSpace(prjSet)
		return self.ProjectSet(tempSet)

	handLimits=[( (-360,360),(-360,360),(-360,360),(-500,-180) ),
		([	(0,45),(0,45),(0,45),(0,45),(0,45),(0,45)], 
		[	[(0,60),(0,60),(0,60),(0,60)],
			[(0,60),(0,60),(0,60),(0,60)],
			[(0,60),(0,60),(0,60),(0,60)],
			[(0,60),(0,60),(0,60),(0,60)],
			[(0,60),(0,60),(0,60),(0,60)],
			[(0,60),(0,60),(0,60),(0,60)]])]
		
	def ProjectAxis(self,ax, delta, limit=60):
		tf0=deepcopy(self.state) # current state
		tf1=deepcopy(self.state)

		if tf1[ax[0]][ax[1]][ax[2]]>0: tf1[ax[0]][ax[1]][ax[2]]-=delta#if (tf1[ax[0]][ax[1]][ax[2]]+tf1[ax[0]][ax[1]][f])>0: tf1[ax[0]][ax[1]][ax[2]]-=delta
		tf2=deepcopy(self.state)
		if tf2[ax[0]][ax[1]][ax[2]]<limit: tf2[ax[0]][ax[1]][ax[2]]+=delta#if (tf1[ax[0]][ax[1]][ax[2]]+tf1[ax[0]][ax[1]][f])<45: tf2[ax[0]][ax[1]][ax[2]]+=delta

		prjSet=[(ax,tf0),(ax,tf1),(ax,tf2)]
		SpaceTransform(self.state[0], self.width, self.height)
		tempSet=self.GetTemplateSet(prjSet)
		return self.ProjectSet(tempSet)

	def DetectColor(self,frame,mprj, pos):
		SpaceTransform(mprj[0], self.width, self.height)
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
		maskColors=np.array([[0,255,255],[255,0,0],[0,255,0],[0,0,255],[255,0,255],[255,255,0]])
		self.hand.Draw(mprj[1],thickRatio=self.hand.thickRatio/6, colors=maskColors)
		buffer = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
		pix = np.asarray(Image.frombuffer(mode="RGB", size=(self.width, self.height), data=buffer))
		maskAll=None
		srcAll=None

		for i in range(0,len(maskColors)):
			reg=pix.copy()
			np.set_printoptions(threshold=sys.maxsize)
			colBgr=np.flip(maskColors[i])
			mask=(pix==colBgr).all(axis=2)
			reg[~mask]=[0,0,0]
			#reg[np.where((reg==maskColors[i]).all(axis=2))]=[255,255,255]
			reg=cv2.pyrUp(reg)
			h,w=reg.shape[0:2]
			pmask=np.array(~(reg==[0,0,0]).all(axis=2),np.uint8)#np.where((reg!=[0,0,0]).all(axis=2))
			hf,wf=frame.shape[0:2]
			x1=int(round(pos[0]))
			xm1=0
			ym1=0
			if x1<0: xm1=-x1;x1=0
			y1=int(round(pos[1]))
			if y1<0: ym1=-y1;y1=0
			x2=min(int(round(pos[0]))+w,wf)
			y2=min(int(round(pos[1]))+h,hf)
			srcIm=deepcopy(frame[y1:y2,x1:x2])
			pmask=deepcopy(pmask[ym1:ym1+y2-y1,xm1:xm1+x2-x1])
			if maskAll is None: maskAll=deepcopy(pmask)
			else: maskAll = maskAll | pmask
			srcAll=srcIm.copy()
			srcYCrCb = cv2.cvtColor(srcIm,cv2.COLOR_BGR2YCR_CB)
			hcol=GetColor(srcYCrCb, pmask)
			tcol=cv2.mean(srcYCrCb, mask=pmask)[0:3]#
			col=np.array([int(tcol[0]),int(tcol[1]),int(tcol[2])],np.uint8)
			bgrCol=ConvertYCrCb2BGR(col)
			h,w=srcIm.shape[0:2]
			pt=(int(5*w/11),int(3*h/11))
			ptCol=srcYCrCb[pt[1],pt[0]]
			#srcIm[~pmask]=[0,0,0]
			#H, edges=np.histogramdd(srcYCrCb, bins=(128,128,128), density=False, weights=pmask)
			#part = cv2.bitwise_and(sourceImage, sourceImage, mask = reg)
			if i==3: 
				#srcHsv = cv2.cvtColor(srcIm,cv2.COLOR_BGR2HSV)
				#avg_color=cv2.mean(srcYCrCb, mask=pmask)[0:3]#
				#min_YCrCb = ptCol-np.array([30,100,250],np.int16)
				#max_YCrCb = ptCol+np.array([30,100,250],np.int16)
				#partRegion = cv2.inRange(srcYCrCb, min_YCrCb,max_YCrCb)
				#pskin = cv2.bitwise_and(srcIm, srcIm, mask = partRegion)
				pskin=RemoveBackground2(srcIm,srcYCrCb,ptCol)
				cv2.circle(srcYCrCb,pt,5, (128,0,255), 2)
				#print('detect colors', hcol, col,ptCol); 
			self.hand.colors[i]=np.flip(bgrCol)
		srcMasked = cv2.bitwise_and(srcAll, srcAll, mask = maskAll)
		nmask=np.array((srcMasked==[0,0,0]).all(axis=2),np.uint8)
		srcFrame = cv2.bitwise_and(srcAll, srcAll, mask = nmask)
		bcol=cv2.mean(srcAll, mask=nmask)[0:3]#
		bcol=np.array([int(bcol[0]),int(bcol[1]),int(bcol[2])],np.uint8)
		#a = np.ma.array(srcAll, mask=~maskAll,axis=2)
		#a = np.ma.masked_where(maskAll == 0, srcAll)
		#bckColor=np.flip(bcol)#np.flip(ConvertYCrCb2BGR(bcol))
		#bck_color_per_row = np.average(srcAll[nmask])
		#bck_color = np.average(bck_color_per_row)
			
	def Detect(self):
		self.mn=-11000000000
		for i in range(0,len(self.ims)):
			h,w=self.ims[i][self.scale].shape
			fh,fw=self.pyrs[self.scale].shape
			if fh<h or fw<w:
				continue
			res3 = cv2.matchTemplate(self.pyrs[self.scale],self.ims[i][self.scale],cv2.TM_CCOEFF_NORMED)
			min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res3)
			max_val = max_val*45/((-self.prj[i][0][3])**(0.5))#max_val * (1400/-prj[i][3])**(1/2)
			if max_val>self.mn:
				self.mn=max_val
				self.ml=deepcopy(max_loc)#+(i1,j1)#(pt[1],pt[0])#
				self.maxi=i
				rescc=res3
				self.state=deepcopy(self.prj[i])#p#
				self.templ=deepcopy(self.ims[i])#pim[0]#
				self.ref=deepcopy(self.refPoints[i])
		self.pos=deepcopy(np.array(self.ml)-self.ref[0:2]/(2**self.scale))
		templ,self.colTempl,gray,rect=self.GetProjection(None,self.state[0],self.state[1])

	def Track(self):
		self.ProjectSpaceAxis(0,4) # z axis
		self.ProjectSpaceAxis(1,4) # y axis
		#self.ProjectSpaceAxis(2,4) # x axis
		self.ProjectSpaceAxis(3,self.state[0][3]*0.05,-450,-180) # scale

		if self.pos is not None:
			start = time.time()

			self.ProjectAxis((1,0,0),4) # global finger stretch
			self.ProjectAxis((1,1,(0,0)),5, 15) # global bending

			for f in range(2,6):
				self.ProjectAxis((1,0,f),8,15) # angle of each finger

			for f in range(2,6):
				self.ProjectAxis((1,1,(f,0)),8,40) # bending of each finger
			self.ProjectAxis((1,1,(1,0)),8,32) # thumb bending

	def ProjectFrame(self, frame):
		fr=frame.copy()
		h,w,c = frame.shape
		grayfr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gradfr = GetGradient(grayfr)
		gradfr = cv2.GaussianBlur(gradfr, (9, 9), 0)
		cframe= cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
		self.cframes=[cframe]
		self.pyrs=[gradfr]#[frame]
		gframe=gradfr
		for i in range(self.scale):
			gframe = cv2.pyrDown(gframe)
			self.pyrs.append(gframe)
			cframe=cv2.pyrDown(cframe)
			self.cframes.append(cframe)

		if self.mn is not None and self.mn > 1.4:
		
			start = time.time()
			self.Track()
			end = time.time()
			print('Track time:',end-start)

			if self.mn>1.5: 
				self.ml=self.pos+self.ref[0:2]/(2**self.scale)
				self.DetectColor(frame, self.state, self.pos*2**self.scale)
		else:
			start = time.time()
			self.Detect()
			end = time.time()
			print('Detect time:',end-start)
			
		y=int(self.ml[1]*2**self.scale)
		x=int(self.ml[0]*2**self.scale)
		self.x=x
		self.y=y
		hg,wg=self.templ[0].shape
		hc,self.bckColor = GetAverageColor(frame,(x,y,wg,hg))
		#frameh=RemoveBackground(frame,(x1,y1,wg,hg))
		
		if self.templ is not None and self.templ[0] is not None:
			ov=cv2.cvtColor(self.templ[0], cv2.COLOR_GRAY2BGR)
			hh,ww=self.templ[0].shape[0:2]
			if self.ml is not None and ov is not None:
				resim = OverlayImage(frame, ov, x, y, 0.6)
				top_left = (x, y)
				bottom_right = (x+ww, y+hh)
				cv2.rectangle(frame, top_left, bottom_right, 255, 2)
			
		if self.colTempl is not None and self.colTempl[1] is not None:
			cvt = cv2.cvtColor(self.colTempl[0],cv2.COLOR_YCR_CB2BGR)
			#cv2.imshow('State:', cv2.cvtColor(self.colTempl[0],cv2.COLOR_YCR_CB2BGR))
			hh,ww=cvt.shape[0:2]
			frame[h-hh:,w-ww:,:] = cvt

		gesture = self.hand.RecogniseGesture(self.state[1])
		cv2.putText(frame,str(round(self.mn,3))+' Gesture: '+gesture,(10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(200,100,50),2,cv2.LINE_AA)

		if self.state:
			print("State: ",self.state[0])
			
		print('Position',self.pos)
		cv2.imshow('Hand Projection',frame)
		return gesture

def main():

	pygame.init()
	(width, height) = (150, 150)
	screen = pygame.display.set_mode((width, height), OPENGL | DOUBLEBUF)  
	done = False
	clock = pygame.time.Clock()

	cap = cv2.VideoCapture(0)
	notEnd = True

	# Set geometry proportions of Hand Model, can variate from person to person
	lRatio, wRatio, tRatio = (1.03, 0.8, 0.93)
	if len(sys.argv)>1:
		print (sys.argv[1:])
		lRatio, wRatio, tRatio = sys.argv[1:]

	hand = Hand(size = 80.0, lenRatio = float(lRatio), widthRatio = float(wRatio), thickRatio = float(tRatio))
	print('Building projection set, wait..')
	projector = Projector(hand, width, height, scale=2)

	while notEnd:
		#notEnd = False
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				sys.exit()

		ret, frame = cap.read()
		#frame = cv2.imread('frame.jpg')

		gesture=projector.ProjectFrame(frame)
		
		clock.tick(60)
		kbd=cv2.waitKey(1)
		if kbd & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
		if kbd & 0xFF == ord('s'):
			cv2.imwrite('Projection_'+gesture+time.strftime("_%Y%m%d-%H%M%S")+'.jpg',frame)

if __name__ == "__main__":
    main()