import cv2 #Import OpenCV
import numpy as np
import math
import time
from picamera.array import PiRGBArray
from picamera import PiCamera

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.neighbors import kneighbors_graph
global keyPointsToVotes

nndistanceParam = 0.70
#minVoteShare = 0.25
targetFoundCount = 0
searchRounds = 0
#REFERENCE
#This program is based on similar examples found at https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/
#centr = np.matrix('135;214') #Centre of CMOS Boox
#centr = np.matrix('248;343') #Centre of Pi Camera Box
centr = np.matrix('265;321') #Centre of Pen
def drawToScreen(img1, keypoints1, img2, keypoints2, matches, estCent, votedList):

# Create a new output image that concatenates the two images together
# (a.k.a) a montage
	#We want to have the target image and the search image 
	#appear on the screen at the same time
	rows1 = img1.shape[0] #This sets up the proper screen dimensions based on size of input images
	cols1 = img1.shape[1]
	#rows1 = 0
	#cols1 = 0
	rows2 = img2.shape[0]
	cols2 = img2.shape[1]
	print "Matches: " + str(len(matches))
	print "votedList: " + str(len(votedList))
	out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

	# Place the first image to the left
	out[:rows1,:cols1] = np.dstack([img1, img1, img1])

	# Place the next image to the right of it
	out[:rows2,cols1:] = np.dstack([img2, img2, img2])
	# For each pair of points we have between both images
	# draw circles, then connect a line between them
	if votedList[0,0] <> 0 or votedList[0,1] <> 0:
		for mat in matches:

			# Get the matching keypoints for each of the images
			img1_idx =  mat.trainIdx
			img2_idx = mat.queryIdx

			# x - columns
			# y - rows
			(x1,y1) = keypoints1[img1_idx].pt
			(x2,y2) = keypoints2[img2_idx].pt

		
			#Draw circles at keypoint matches
			cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
			cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (0, 255, 0), 1)
			#cv2.putText(out,str(keypoints2[img2_idx ].angle - keypoints1[img1_idx].angle),(int(x2 + cols1),int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
			#Draw Line 
			cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 255, 0), 1)
			cv2.line(out, (int(x1), int(y1)), (int(centr[0]), int (centr[1])), (0,255,0), 1)
		cv2.circle(out, (int(centr[0]),int(centr[1])), 8, (200, 50, 100), 3)  		
		for v in votedList:
			cv2.circle(out, (int(v[0] + cols1),int(v[1])), 8, (0, 144, 255), 3)
			
		# Show the image
		for key, val in keyPointsToVotes.items():
				print val.queryIdx
				img2_idx = val.queryIdx
				(x2,y2) = keypoints2[img2_idx].pt
				cv2.line(out, (int(x2 + cols1),int(y2)), (int(votedList[key][0])+cols1,int(votedList[key][1])), (200, 150, 30), 1)
		for c in estCent:
			
			cv2.circle(out, (int(c[0][0] + cols1),int(c[0][1])), 8, (0, 0, 255), 3)  
	cv2.imshow('Matched Features', out)

	# Also return the image if you'd like a copy
	return out
	
def siftKeypointMatching(img1, img2, siftInst):
	global keyPointsToVotes
	startTime = int(round(time.time() *1000))
	keypoints1, des1 = siftInst.detectAndCompute(img1,None) #Get Keypoints and descriptors for image 1
	keypoints2, des2 = siftInst.detectAndCompute(img2,None) #Get Keypoints and descriptors for image 2
	bfMatcher = cv2.BFMatcher() #Create instance of the Matcher (Brute Force Matcher)
	#flann_params = dict(algorithm = 1, trees = 5)
	#flann = cv2.FlannBasedMatcher(flann_params, {})
	matches = bfMatcher.knnMatch(des2,des1, k=2) #OpenCV Nearest Neighbour matches matches the keypoints
	endTime = int(round(time.time() *1000))
	print 'SIFT Takes ' + str(endTime - startTime) + ' mS'
#drawMatches(img1, kp1, img2, kp2, matches[:20])
# Apply ratio test which is outlined in Lowe's 2004 Paper (See Bibliography)
#This test rejects keypoints with descriptors that match multiple keypoints too closely
	accepted = []
	votingSpaceMat = np.array([[0,0]])
	keyPointsToVotes = {}
	index = 0
	for m,n in matches: #Go through all the matches 
		if m.distance < nndistanceParam * n.distance: #If there are two match pairs with descriptors more than 70% in agreement this is probably background clutter (the match is not distinct enough)
			if index > 0:
				if  ( votingSpaceMat[index -1 ,0] <> 0 or votingSpaceMat[index -1,1] <> 0):
						votingSpaceMat = np.concatenate((votingSpaceMat, np.array([[0,0]])))
			accepted.append(m)
			trainIndex = m.trainIdx
			qryIndex = m.queryIdx
			keyPtVec = np.matrix('0;0')
			qryKeyPtVec = np.matrix('0;0')
			beta = keypoints2[qryIndex].angle - keypoints1[trainIndex].angle
			print "Beta " + str(beta)
			beta = math.radians(beta)
			#scaleRatio =  (math.pow(2,max(0,keypoints1[trainIndex].octave & 255))/math.pow(2, max(0,keypoints2[ qryIndex].octave & 255)))
			octaveTrain = keypoints1[trainIndex].octave & 255
			octaveQry = keypoints2[ qryIndex].octave & 255
			
			if  octaveTrain > 128:
				octaveTrain = -128| octaveTrain 
			if octaveQry > 128:
				octaveQry = -128 | octaveQry
			scaleNumer = (1.0 *(1 << (-1*octaveTrain))) if ( octaveTrain) <= 0 else (1.0 /(1 << (octaveTrain)))
			scaleDen = 1.0*(1<<  (-1*octaveQry )) if (octaveQry) <= 0 else 1.0/(1<<  (octaveQry ))
			scaleRatio = scaleNumer / scaleDen			
			#scaleRatio = 1.0
			print "Scale: " + str(scaleRatio)
			rotMat = np.identity(2)			
			rotMat[0,0] = math.cos(beta)
			rotMat[0,1] = -1*math.sin(beta)
			rotMat[1,0] = 1*math.sin(beta)
			rotMat[1,1] = math.cos(beta)	
			keyPtVec[0,0] = keypoints1[trainIndex].pt[0]
			keyPtVec[1,0] = keypoints1[trainIndex].pt[1]
			qryKeyPtVec[0,0] = keypoints2[qryIndex].pt[0]
			qryKeyPtVec[1,0] = keypoints2[qryIndex].pt[1]
			np.concatenate(keyPtVec)
			locRel = np.matrix('0;0')
			locRel[0,0] =  centr[0,0] - keyPtVec[0,0] 
			locRel[1,0] =  centr[1,0] - keyPtVec[1,0]
			rotResult = np.dot(rotMat, locRel)
			pl = qryKeyPtVec + (scaleRatio*rotResult)
			#Add a row to the votingSpaceMatrix to be used by the Agglomerative clustering function
			votingSpaceMat[index,0] = pl[0,0]
			votingSpaceMat[index,1] = pl[1,0]	
			keyPointsToVotes[index] = m
			index = index + 1
	#print votingSpaceMat			
	return [keypoints1, keypoints2, accepted, votingSpaceMat ]
			
#Initialize Pi Camera for Video Capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera)
time.sleep(0.1)

img1 = cv2.imread('PiCameraBox.jpg') #Target Image is a textbook
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create() #Create an instance of the SIFT detector
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	img2 = frame.array #Read From Video Capture Object	
	img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)	
	[keypoints1, keypoints2, accepted, votingSpace] = siftKeypointMatching(img1, img2, sift)
	X = votingSpace
	#X = np.reshape(votingSpace, (-1,1))
	#numVotingClusters = max(3,min(15,int(len(accepted)/3)))
	numVotingClusters = 1
	print numVotingClusters
	estCent = np.array([[0,0]])
	voted = []
	centres = []
	rawCapture.truncate(0)
	if len(X) > numVotingClusters:
		knn_graph = kneighbors_graph(X, min(3,len(X)-1), include_self=False)
		#clustering = AgglomerativeClustering(linkage='complete', connectivity=knn_graph)
		clustering = DBSCAN(eps=15, min_samples=3)	
		clustering.fit(X)		
		clusters = clustering.labels_
		clusterList = list(clusters)
		#minVoteNum = math.ceil(minVoteShare * len(X))
		minVoteNum = 3
		print "Min Voting:" + str(minVoteNum)		
		print clustering.labels_
		clusterWMaxVote = -1
		curMaxVote = 0
		for n in range(0,numVotingClusters):			
			voteCount = clusterList.count(n)
			if voteCount >  curMaxVote and voteCount >= minVoteNum:
				curMaxVote = voteCount
				clusterWMaxVote = n				
		if clusterWMaxVote <> -1:
			print "Cluster " + str(clusterWMaxVote)
			estCent = np.array([[0,0]])							
			for m in range(0,len(clusterList)):
				if clusterList[m] == clusterWMaxVote:					
					voted.append(list(votingSpace[m]))
					estCent[0][0] = estCent[0][0] + votingSpace[m][0]
					estCent[0][1] = estCent[0][1] + votingSpace[m][1]
			estCent[0][0] = (estCent[0][0]/curMaxVote)
			estCent[0][1] = (estCent[0][1]/curMaxVote)
			print estCent
			centres.append(estCent)
	if cv2.waitKey(1) & 0xFF == ord('q'):	
		break
	if estCent[0][0] <> 0 or estCent[0][1] <> 0:
		targetFoundCount = targetFoundCount + 1
	searchRounds = searchRounds + 1
	#print votingSpace
	drawToScreen(img1, keypoints1, img2, keypoints2, accepted, centres, votingSpace)

print "Target Found: " + str(targetFoundCount) + " times "
print "Attempts: " + str(searchRounds) 
print "Detection Rate: " + str(100.0 * ((1.0 * targetFoundCount) / searchRounds)) 
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()


	
