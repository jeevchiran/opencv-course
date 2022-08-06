import cv2
import numpy as np

img = cv2.imread('BarcaReal001.jpg')

dst_points = np.array([[141,175],[293,142],
                    [295,219],[144,261]])
# cv2.fillPoly(img,pts = [points],color = (0,255,0))

# cv2.circle(img,(141,175),5,(0,0,255),-1)
# cv2.circle(img,(293,142),5,(0,0,255),-1)
# cv2.circle(img,(295,219),5,(0,0,255),-1)
# cv2.circle(img,(144,261),5,(0,0,255),-1)
img_src = cv2.imread('penn_logo.jpeg')
# cv2.imshow("BarcaReal",img)
img_src = cv2.rotate(img_src,cv2.ROTATE_90_COUNTERCLOCKWISE)

src_points = np.array([[0,0],[0,514],[514,514],[514,0]])

h,status = cv2.findHomography(src_points,dst_points)
print("homography matrix using opencv \n",h)
im_out = cv2.warpPerspective(img_src,h,(img.shape[1],img.shape[0]))

# cv2.imshow("homograph",im_out)

bitwise_or = cv2.bitwise_or(img, im_out)
# cv2.imshow("homograph",bitwise_or)

# cv2.waitKey(0)
A = np.zeros([8,9],dtype=None)
for i in range(4):
    x_prime = dst_points[i,0]
    y_prime = dst_points[i,1]

    x = src_points[i,0]
    y = src_points[i,1]

    A[2 * i, :] = [-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime]
    A[2 * i+1, :] = [0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime]

# print("matrix A: ", A)

[U,S,V]=np.linalg.svd(A)
m = V[-1,:]
H = np.reshape(m,(3,3))

print("homography matrix \n",H)

im_out = cv2.warpPerspective(img_src,H,(img.shape[1],img.shape[0]))

cv2.imshow("homograph",im_out)

bitwise_or = cv2.bitwise_or(img, im_out)
cv2.imshow("homograph",bitwise_or)
cv2.waitKey(0)