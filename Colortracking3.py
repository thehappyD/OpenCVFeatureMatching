# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import cv2
import cv2 as cv
import smbus
import time

bus = smbus.SMBus(1)

#The arduino address
address = 0x04;



def writeNumber(value):
    bus.write_byte(address, value)
    return -1

def readNumber():
    number = bus.read_byte(address)
    return number

def servo_Move(tiltServoPosition, panServoPosition): # Servo control method
    address = 0x04;
    tiltChannel = 0;
    panChannel = 1;
    print "Setting the tiltChannel";
    bus.write_byte(address, tiltChannel);

    print "Setting the tiltServoPosition to ",(tiltServoPosition);
    bus.write_byte(address, tiltServoPosition);

    print "Setting the panChannel";
    bus.write_byte(address, panChannel);

    print "Setting the panServoPosition to ",(panServoPosition);
    bus.write_byte(address, panServoPosition);
    return 1
    

buzz = 0;
counter = 0;

# Setting Screen Parameters
width = 320;
height = 240;

#Setting middle of the screen
midScreenY = height/2;
midScreenX = width/2;
midScreenError = 10; #The acceptable error

#Pan-tilt servo ids for the Arduino;
tiltChannel = 0;
panChannel = 1;

#Variables to track servo position
tiltServoPosition = 90;
panServoPosition = 90;

#Degree of change
stepSize = 1;

servo_Move(tiltServoPosition, panServoPosition); # Setting the servos to initial position


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
cv2.createTrackbar('hmin', 'HueComp',0,179,nothing)
cv2.createTrackbar('hmax', 'HueComp',179,179,nothing)

cv2.createTrackbar('smin', 'SatComp',0,255,nothing)
cv2.createTrackbar('smax', 'SatComp',255,255,nothing)

cv2.createTrackbar('vmin', 'ValComp',0,255,nothing)
cv2.createTrackbar('vmax', 'ValComp',30,255,nothing)

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

        counter = 0;

        #Draw Circles
        if circles is not None:
                counter = counter + 1;
                for i in circles[0,:]:
                    # If the ball is far, draw it in green
                    if int(round(i[2])) < 30:
					
                        midCircleX = int(round(i[0]));
                        midCircleY = int(round(i[1]));

                        #Find if the Y components are below the screen
                        if(midCircleY < (midScreenY-midScreenError)) :
                            if(tiltServoPosition >= 5) :
                               tiltServoPosition += stepSize;

                           
                        #Find if the Y components are above the screen
                        elif(midCircleY > (midScreenY+midScreenError)) :
                            if(tiltServoPosition <= 175) :
                               tiltServoPosition -= stepSize;


                        #Find if the X components are below the screen
                        if(midCircleX < (midScreenX-midScreenError)) :
                            if(panServoPosition >= 5) :
                               panServoPosition -= stepSize;

                        #Find if the X components are above the screen   
                        elif(midCircleX > (midScreenX+midScreenError)) :
                            if(panServoPosition <= 175) :
                               panServoPosition += stepSize;

                           
                               
                        
                        cv2.circle(frame,(int(round(i[0])),int(round(i[1]))),int(round(i[2])),(0,255,0),5)
                        #print int(round(i[1]));
##                        print "Setting the tiltChannel";
##                        bus.write_byte(address, tiltChannel);
##                        print "Setting the tiltServoPosition to ",(tiltServoPosition);
##                        bus.write_byte(address, tiltServoPosition);
##                        print "Setting the panChannel";
##                        bus.write_byte(address, panChannel);
##                        print "Setting the panServoPosition to ",(panServoPosition);
##                        bus.write_byte(address, panServoPosition);
                           
                        cv2.circle(frame,(int(round(i[0])),int(round(i[1]))),2,(0,255,0),10)
                        buzz = 1;

                    else :
                        buzz = 0;
                                    # else draw it in red
                    if int(round(i[2])) > 35:
							
                        midCircleX = int(round(i[0]));
                        midCircleY = int(round(i[1]));

                        #Find if the Y components are below the screen
			if(midCircleY < (midScreenY-midScreenError)) :
                            if(tiltServoPosition >= 5) :
                                tiltServoPosition += stepSize;
                                print midCircleY;

                           
                        #Find if the Y components are above the screen
                        elif(midCircleY > (midScreenY+midScreenError)) :
                            if(tiltServoPosition <= 175) :
				tiltServoPosition -= stepSize;
				print midCircleY;


                        #Find if the X components are below the screen
                        if(midCircleX < (midScreenX-midScreenError)) :
                            if(panServoPosition >= 5) :
				panServoPosition -= stepSize;

                        #Find if the X components are above the screen   
                        elif(midCircleX > (midScreenX+midScreenError)) :
                            if(panServoPosition <= 175) :
				panServoPosition += stepSize;
                               
                           

                           
                        cv2.circle(frame,(int(round(i[0])),int(round(i[1]))),int(round(i[2])),(0,0,255),5)
                                    #print int(round(i[1]));
                        cv2.circle(frame,(int(round(i[0])),int(round(i[1]))),2,(0,0,255),10)
                        
                        buzz = 1
                    else :
                        buzz = 0;


        #Updating the servo positions;
        

            #you can use the 'buzz' variable as a trigger to switch some GPIO lines on Rpi :)
        # print buzz

##        else:
##            time.sleep(5);
##            servo_Move(90,180);
##            continue
        
        if buzz == 1:
            print "Setting the tiltChannel";
            bus.write_byte(address, tiltChannel);
            print "Setting the tiltServoPosition to ",(tiltServoPosition);
            bus.write_byte(address, tiltServoPosition);
            print "Setting the panChannel";
            bus.write_byte(address, panChannel);
            print "Setting the panServoPosition to ",(panServoPosition);
            bus.write_byte(address, panServoPosition);
            time.sleep(0.001);
            buzz = 0;



            
            

            


        
        #Show the result in frames
        cv2.imshow('HueComp',hthresh)
        cv2.imshow('SatComp',sthresh)
        cv2.imshow('ValComp',vthresh)
        cv2.imshow('closing',closing)
        cv2.imshow('tracking',frame)


##        if counter == 0:
##            # it hasnt found anything;
##            print "Nothing found";
##            time.sleep(5);
##            servo_Move(90,180);





	key = cv2.waitKey(1) & 0xFF
 
	# clear the stream in preparation for the next frame
	rawCapture.truncate(0)
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

cv2.destroyAllWindows()
