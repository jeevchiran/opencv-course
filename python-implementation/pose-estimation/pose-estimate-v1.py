from hashlib import sha1
import cv2
import numpy as np

def homography(corners,dim):
    #Define the eight points to compute the homography matrix
    x = []
    y = []

    #ccw corners
    x_actual=np.array([0,dim,dim,0])
    y_actual=np.array([0,0,dim,dim])
    A = np.zeros([8,9],dtype=None)
    for i in range(4):
        x_prime = x_actual[i]
        y_prime = y_actual[i]
        x = corners[i][0]
        y = corners[i][1]
        A[2 * i, :] = [-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime]
        A[2 * i+1, :] = [0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime]
    [U,S,V] = np.linalg.svd(A)
    # x is equivalent to the eigenvector column of V that corresponds to the 
    # smallest singular value. A*x ~ 0
    x = V[-1]

    # reshape x into H
    H = np.reshape(x,[3,3])
    return H

def projection_mat(K,H):
	h1=H[:,0]
	h2=H[:,1]

	K_inv=np.linalg.inv(K)

    # depth
	a=K_inv@h1
	c=K_inv@h2
	lamda=1/((np.linalg.norm(a)+np.linalg.norm(c))/2)

    # H_inverse
    # R=K^-1*H^-1
	rotationMatrix=K_inv@H

	if np.linalg.det(rotationMatrix)>0:
		R=1*rotationMatrix
	else:
		R=-1*rotationMatrix

	b1=R[:,0]
	b2=R[:,1]
	b3=R[:,2]
	r1=lamda*b1
	r2=lamda*b2
	r3=np.cross(r1,r2)
	t=lamda*b3
    # P = K*[r1,r1,r3,t]
	P=K@(np.stack((r1,r2,r3,t), axis=1))

	return P
def cubePoints(corners, H, P, dim):
	# corners =[p1,p2,p3,p4]
	# pi=[xi,yi]
    new_points = []
    new_corners=[]
    x = []
    y = []
    for point in corners:
        x.append(point[0])
        y.append(point[1])
    # add arrays about new axis 
    # homogeneous coordinates
    H_c = np.stack((np.array(x),np.array(y),np.ones(len(x))))
    print("homogeneous coordinates \n",H_c)
    # world coordinates            
    sH_w=H@H_c
    print("world coordinates\n",sH_w)
    H_w=sH_w/sH_w[2]
    print("actual world coordinates\n",H_w)

    # projection points
    P_w=np.stack((H_w[0],H_w[1],np.full(4,-dim),np.ones(4)),axis=0)

    sP_c=P@P_w
    P_c=sP_c/(sP_c[2])

    for i in range(4):
        new_corners.append([int(P_c[0][i]),int(P_c[1][i])])

    return new_corners

def drawCube(tagcorners, new_corners,frame,edge_color):
	thickness=5
    
	for i, point in enumerate(tagcorners):
		cv2.line(frame, tuple(point), tuple(new_corners[i]), edge_color, thickness) 

	for i in range (4):
		if i==3:
			cv2.line(frame,tuple(tagcorners[i]),tuple(tagcorners[0]),edge_color,thickness)
			cv2.line(frame,tuple(new_corners[i]),tuple(new_corners[0]),edge_color,thickness)
		else:
			cv2.line(frame,tuple(tagcorners[i]),tuple(tagcorners[i+1]),edge_color,thickness)
			cv2.line(frame,tuple(new_corners[i]),tuple(new_corners[i+1]),edge_color,thickness)
        
	return frame


def makeContours(corners1,corners2):
	contours = []
	for i in range(len(corners1)):
		if i==3:
			p1 = corners1[i]
			p2 = corners1[0]
			p3 = corners2[0]
			p4 = corners2[i]
		else:
			p1 = corners1[i]
			p2 = corners1[i+1]
			p3 = corners2[i+1]
			p4 = corners2[i]
		contours.append(np.array([p1,p2,p3,p4], dtype=np.int32))
	contours.append(np.array([corners1[0],corners1[1],corners1[2],corners1[3]], dtype=np.int32))
	contours.append(np.array([corners2[0],corners2[1],corners2[2],corners2[3]], dtype=np.int32))

	return contours

def findcontours(frame,threshold):
    imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgray= cv2.medianBlur(imgray,5)
    ret, thresh = cv2.threshold(imgray, threshold, 255, cv2.THRESH_BINARY)

    all_cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # remove any contours that do not have a parent or child
    wrong_cnts = []
    for i,h in enumerate(hierarchy[0]):
        if h[2] == -1 or h[3] == -1:
            wrong_cnts.append(i)
    cnts = [c for i, c in enumerate(all_cnts) if i not in wrong_cnts]

    # sort the contours to include only the three largest
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:3]

    return [all_cnts,cnts]

def approx_quad(cnts):
    tag_cnts = []
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, peri*.015, True)
        # if the countour can be approximated by a polygon with four sides include it
        if len(approx) == 4:
            tag_cnts.append(approx)

    corners = []
    for shape in tag_cnts:
        coords = []
        for p in shape:
            coords.append([p[0][0],p[0][1]])
        corners.append(coords)

    return tag_cnts,corners    

frame = cv2.imread('image001.jpg')
gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

[all_cnts,cnts] = findcontours(frame,100)

[tag_cnts,corners] = approx_quad(cnts)

for item in corners[0]:
    x, y = item
    x = int(x)
    y = int(y)
    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
cv2.imshow("QR",frame)

# cv2.imshow("pose",frame)

for i,tag in enumerate(corners):
    scalar = 1.0e+02
    K = scalar * np.array([[7.661088867187500,0,3.139585628047498],[0,7.699354248046875,2.503607131410900],[0,0,0.010000000000000]])

    H=homography(tag,40)
    H_inv = np.linalg.inv(H)
    P=projection_mat(K,H_inv)
    print("Projection matrix\n",P)
    print("Homography\n",H)
    print("corners\n",tag)
    new_corners=cubePoints(tag, H, P, 40)

    # draw the cube onto the frame
    edge_color=(0,255,0)
    frame=drawCube(tag, new_corners,frame,edge_color)
cv2.imshow("pose",frame)
cv2.waitKey(0)
