import socket
import cv2
import numpy as np
import time
import math
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.neighbors import kneighbors_graph
global keyPointsToVotes

nndistanceParam = 0.70
#minVoteShare = 0.25
targetFoundCount = 0
searchRounds = 0
centr = np.matrix('248;343') #Centre of Pi Camera Box

#IP_ADDR = '192.168.0.100' #Home WiFi
IP_ADDR = '169.254.154.9' #Ethernet
PORT = 5000
BUFFER_SIZE = 100000

def siftKeypointMatching(img1, img2, siftInst):
	global keyPointsToVotes
	startTime = int(round(time.time() *1000))
	keypoints1, des1 = siftInst.detectAndCompute(img1,None) #Get Keypoints and descriptors for image 1
	keypoints2, des2 = siftInst.detectAndCompute(img2,None) #Get Keypoints and descriptors for image 2
	bfMatcher = cv2.BFMatcher() #Create instance of the Matcher (Brute Force Matcher)
	matches = bfMatcher.knnMatch(des2,des1, k=2) #OpenCV Nearest Neighbour matches matches the keypoints
	endTime = int(round(time.time() *1000))
	#print 'SIFT Takes ' + str(endTime - startTime) + ' mS'
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
			#print "Beta " + str(beta)
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
			#print "Scale: " + str(scaleRatio)
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
	
	
def recComplete(s, length):
    buffr = b''
    while length:
            addBuffer = s.recv(length)
            if not addBuffer: return None
            buffr += addBuffer
            length -= len(addBuffer)
    return buffr
	
	
	
#MAIN	
img1 = cv2.imread('PiCameraBox.jpg') #Target Image is a textbook
height, width = img1.shape[:2]
print height
print width
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((IP_ADDR, PORT))
s.listen(1)

conn, addr = s.accept()
print 'Connection Established with ' + str(addr)
cnt = 0
sift = cv2.SIFT() #Create an instance of the SIFT detector
time.sleep(1)
while True:		
		length = recComplete(conn, 16)        
		data = recComplete(conn, int(length))
		#print 'Receiving Image, length: ' + str(int(length))
		imgData = np.fromstring(data, dtype='uint8')
		decImgData = cv2.imdecode(imgData,1)
		decImgData = cv2.cvtColor(decImgData, cv2.COLOR_BGR2GRAY)
		[keypoints1, keypoints2, accepted, votingSpace] = siftKeypointMatching(img1, decImgData, sift)
		X = votingSpace
		#X = np.reshape(votingSpace, (-1,1))
		#numVotingClusters = max(3,min(15,int(len(accepted)/3)))
		numVotingClusters = 1
		estCent = np.array([[0,0]])
		voted = []
		centres = []	
		if len(X) > numVotingClusters:
			knn_graph = kneighbors_graph(X, min(3,len(X)-1), include_self=False)
			#clustering = AgglomerativeClustering(linkage='complete', connectivity=knn_graph)
			clustering = DBSCAN(eps=15, min_samples=3)	
			clustering.fit(X)		
			clusters = clustering.labels_
			clusterList = list(clusters)
			#minVoteNum = math.ceil(minVoteShare * len(X))
			minVoteNum = 3
			clusterWMaxVote = -1
			curMaxVote = 0
			for n in range(0,numVotingClusters):			
				voteCount = clusterList.count(n)
				if voteCount >  curMaxVote and voteCount >= minVoteNum:
					curMaxVote = voteCount
					clusterWMaxVote = n				
			if clusterWMaxVote <> -1:
				estCent = np.array([[0,0]])							
				for m in range(0,len(clusterList)):
					if clusterList[m] == clusterWMaxVote:					
						voted.append(list(votingSpace[m]))
						estCent[0][0] = estCent[0][0] + votingSpace[m][0]
						estCent[0][1] = estCent[0][1] + votingSpace[m][1]
				estCent[0][0] = (estCent[0][0]/curMaxVote)
				estCent[0][1] = (estCent[0][1]/curMaxVote)
				print 'CENTER FOUND AT: ' + str(estCent)

				centres.append(estCent)
			
		if cv2.waitKey(1) & 0xFF == ord('q'):	
			break
		if estCent[0][0] <> 0 or estCent[0][1] <> 0:
			targetFoundCount = targetFoundCount + 1
		#Send Centre to Raspberry Pi
		conn.send(';' + str(estCent[0][0]) + ',' + str(estCent[0][1]))
			
conn.close()
