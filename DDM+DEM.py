import os
import cv2
import numpy as np
from random import randint
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


imgPath='nyu2_test'
LabelPath='nyu_labels'
DepthPath='nyu2_test'
smcLabel='nyu2_test_smclabel'
smcdepth='nyuv2_test_smcdepth'
avgscale=0
avgnum=0

for filedir in os.listdir(smcLabel):
    imgName = imgPath + '\\' + filedir[:-4]+"_colors.png"
    DepthName = DepthPath + '\\'+filedir[:-4]+'_depth.png'
    img = cv2.imread(imgName)
    depth =cv2.imread(DepthName,0)
    im_targetzong= cv2.imread(smcLabel+'\\'+filedir,0)
    im_color = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    oriImg = img.copy()
    labelImg = img.copy() 
    labelImg2 = img.copy() 
    predepthsum=depth.copy()
    errormap=np.ones(predepthsum.shape,np.float32)   
    scale=12
    pointEnd=150
    pointNum=0
    temp=0
    
    for row in range(0,img.shape[0],int(img.shape[0]/scale)):
        for col in range(0,img.shape[1],int(img.shape[1]/scale)):
            imageRegin = oriImg[row:row+int(img.shape[0]/scale),col:col+int(img.shape[1]/scale)]
            depthRegin = depth[row:row+int(img.shape[0]/scale),col:col+int(img.shape[1]/scale)]
            temp=temp+1
            im_target=im_targetzong[row:row+int(img.shape[0]/scale),col:col+int(img.shape[1]/scale)]
            preerror = np.ones(im_target.shape,np.float32)
            labelClass= np.unique(im_target)
            x=[]
            y=[]
            pos=[]
            for i in range(len(labelClass)):
                pointNum=pointNum+1
                if pointNum>pointEnd:
                    break
                c=labelClass[i]   
                classPosition = np.where(im_target==labelClass[i]) 
                index = randint(0,len(classPosition[0])-1)
                x0=classPosition[0][index]
                y0=classPosition[1][index]
                x.append(classPosition[0][index])
                y.append(classPosition[1][index])
                for j in range(len(classPosition[0])):
                    xnow=classPosition[0][j]
                    ynow=classPosition[1][j]
                    errornow=np.sqrt((xnow-x0)*(xnow-x0)+(ynow-y0)*(ynow-y0))
                    preerror[xnow][ynow]=errornow
                pos.append(classPosition)
            errormap[row:row+int(img.shape[0]/scale),col:col+int(img.shape[1]/scale)]=preerror
            pointColor = (0, 255, 255)
            for i in range(len(x)):
                cv2.circle(imageRegin,(y[i],x[i]),1,pointColor,4)
            img[row:row+int(img.shape[0]/scale),col:col+int(img.shape[1]/scale)]=imageRegin

            preDepth = np.ones(depthRegin.shape,np.float32)
            for i in range(len(x)):
                depthValue = depthRegin[x[i]][y[i]] 
                preDepth[pos[i]]=depthValue
            predepthsum[row:row+int(img.shape[0]/scale),col:col+int(img.shape[1]/scale)]=preDepth


    point_color = (0, 255, 0) 
    thickness = 1 
    lineType = 4
    for row in range(int(img.shape[0]/scale),img.shape[0],int(img.shape[0]/scale)):
        cv2.line(img, (0, row), (img.shape[1], row), point_color, thickness, lineType)
    for col in range(int(img.shape[1]/scale),img.shape[1],int(img.shape[1]/scale)):
        cv2.line(img, (col, 0), (col, img.shape[0]), point_color, thickness, lineType)
    im_color2 = cv2.applyColorMap(predepthsum, cv2.COLORMAP_JET)
    cv2.imwrite(smcdepth+ '\\'+filedir[:-4]+'.png',predepthsum)
    
    minA=np.min(errormap)
    maxA=np.max(errormap)
    errorNorm=((errormap-minA)/(maxA-minA))*255
    errorNorm=errorNorm.astype(np.uint8)
    cv2.imwrite(smcdepth+ 'error\\'+filedir[:-4]+'.png',errorNorm)

