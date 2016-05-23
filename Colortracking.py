# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import cv2
import cv2 as cv


kernel = np.ones((5,5),np.uint8)
def nothing(x):
    pass
# Creating a windows for later use
cv2.namedWindow('HueComp')
cv2.namedWindow('SatComp')
cv2.namedWindow('ValComp')
cv2.namedWindow('closing')
cv2.namedWindow('tracking')


# Creating track bar for min and max for hue, saturation and value
# You can adjust the defaults as you like
cv2.createTrackbar('hmin', 'HueComp',12,179,nothing)
cv2.createTrackbar('hmax', 'HueComp',37,179,nothing)

cv2.createTrackbar('smin', 'SatComp',96,255,nothing)
cv2.createTrackbar('smax', 'SatComp',255,255,nothing)

cv2.createTrackbar('vmin', 'ValComp',186,255,nothing)
cv2.createTrackbar('vmax', 'ValComp',255,255,nothing)

# My experimental values
# hmn = 12
# hmx = 37
# smn = 145
# smx = 255
# vmn = 186
# vmx = 255



# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (320, 240)
#camera.framerate = 15
rawCapture = PiRGBArray(camera, size=(320, 240))
 
# allow the camera to warmup
time.sleep(0.1)
 
# capture frames from the camera
for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
	frame = image.array

	#converting to HSV
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        hue,sat,val = cv2.split(hsv)

        # get info from track bar and appy to result
        hmn = cv2.getTrackbarPos('hmin','HueComp')
        hmx = cv2.getTrackbarPos('hmax','HueComp')
        

        smn = cv2.getTrackbarPos('smin','SatComp')
        smx = cv2.getTrackbarPos('smax','SatComp')


        vmn = cv2.getTrackbarPos('vmin','ValComp')
        vmx = cv2.getTrackbarPos('vmax','ValComp')

        # Apply thresholding
        hthresh = cv2.inRange(np.array(hue),np.array(hmn),np.array(hmx))
        sthresh = cv2.inRange(np.array(sat),np.array(smn),np.array(smx))
        vthresh = cv2.inRange(np.array(val),np.array(vmn),np.array(vmx))

        # AND h s and v
        tracking = cv2.bitwise_and(hthresh,cv2.bitwise_and(sthresh,vthresh))

        # Some morpholigical filtering
        dilation = cv2.dilate(tracking,kernel,iterations = 1)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
        closing = cv2.GaussianBlur(closing,(5,5),0)

        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(closing,cv.HOUGH_GRADIENT,2,120,param1=120,param2=50,minRadius=10,maxRadius=0)
        # circles = np.uint16(np.around(circles))

        #Draw Circles
        if circles is not None:
                for i in circles[0,:]:
                    # If the ball is far, draw it in green
                    if int(round(i[2])) < 30:
                        cv2.circle(frame,(int(round(i[0])),int(round(i[1]))),int(round(i[2])),(0,255,0),5)
                        cv2.circle(frame,(int(round(i[0])),int(round(i[1]))),2,(0,255,0),10)
                                    # else draw it in red
                    elif int(round(i[2])) > 35:
                        cv2.circle(frame,(int(round(i[0])),int(round(i[1]))),int(round(i[2])),(0,0,255),5)
                        cv2.circle(frame,(int(round(i[0])),int(round(i[1]))),2,(0,0,255),10)
                        buzz = 1

            #you can use the 'buzz' variable as a trigger to switch some GPIO lines on Rpi :)
        # print buzz                    
        # if buzz:
            # put your GPIO line here

        
        #Show the result in frames
        cv2.imshow('HueComp',hthresh)
        cv2.imshow('SatComp',sthresh)
        cv2.imshow('ValComp',vthresh)
        cv2.imshow('closing',closing)
        cv2.imshow('tracking',frame)









	
##	# Convert BGR to HSV
##        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
##
##        # define range of blue color in HSV
##        lower_blue = np.array([110,50,50])
##        upper_blue = np.array([130,255,255])
##
##        # Threshold the HSV image to get only blue colors
##        mask = cv2.inRange(hsv, lower_blue, upper_blue)
##
##        # Bitwise-AND mask and original image
##        res = cv2.bitwise_and(image,image, mask= mask)
##
##        cv2.imshow('image',image)
##        cv2.imshow('mask',mask)
##        cv2.imshow('res',res)
##
##	# show the frame

	key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

cv2.destroyAllWindows()
