import numpy as np
import cv2

def sobel_x(img):
    kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    filter_image_cv2 = cv2.filter2D(img,ddepth=-1,kernel=kernel_x)
    return filter_image_cv2

def sobel_y(img):
    kernel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    filter_image_cv2 = cv2.filter2D(img,ddepth=-1,kernel=kernel_y)
    return filter_image_cv2

def Canny_detector(img, weak_th = None, strong_th = None):   
    # Noise reduction step
    img = cv2.GaussianBlur(img, (5, 5), 1.4)
       
    # Calculating the gradients
    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)

    # get magnitude of edge and the direction of the egde
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees = True)
    print("mag",mag) 
    print(ang.shape)
    print("angle" , ang)  

    # setting the minimum and maximum thresholds 
    # for double thresholding
    mag_max = np.max(mag)
    if not weak_th:weak_th = mag_max * 0.1
    if not strong_th:strong_th = mag_max * 0.5
      
    # getting the dimensions of the input image  
    height, width = img.shape
       
    # Looping through every pixel of the grayscale 
    # image
    for i_x in range(width):
        for i_y in range(height):
               
            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang-180) if abs(grad_ang)>180 else abs(grad_ang)
               
            # selecting the neighbours of the target pixel
            # according to the gradient direction
            # In the x axis direction
            if grad_ang<= 22.5:
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y
              
            # top right (diagonal-1) direction  
            elif grad_ang>22.5 and grad_ang<=(22.5 + 45):
                neighb_1_x, neighb_1_y = i_x-1, i_y-1
                neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
              
            # In y-axis direction
            elif grad_ang>(22.5 + 45) and grad_ang<=(22.5 + 90):
                neighb_1_x, neighb_1_y = i_x, i_y-1
                neighb_2_x, neighb_2_y = i_x, i_y + 1
              
            # top left (diagonal-2) direction
            elif grad_ang>(22.5 + 90) and grad_ang<=(22.5 + 135):
                neighb_1_x, neighb_1_y = i_x-1, i_y + 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y-1
              
            # Now it restarts the cycle
            elif grad_ang>(22.5 + 135) and grad_ang<=(22.5 + 180):
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y
               
            # Non-maximum suppression step
            if width>neighb_1_x>= 0 and height>neighb_1_y>= 0:
                if mag[i_y, i_x]<mag[neighb_1_y, neighb_1_x]:
                    mag[i_y, i_x]= 0
                    continue
   
            if width>neighb_2_x>= 0 and height>neighb_2_y>= 0:
                if mag[i_y, i_x]<mag[neighb_2_y, neighb_2_x]:
                    mag[i_y, i_x]= 0
   
    weak_ids = np.zeros_like(img)
    strong_ids = np.zeros_like(img)              
    ids = np.zeros_like(img)
       
    # double thresholding step
    for i_x in range(width):
        for i_y in range(height):
              
            grad_mag = mag[i_y, i_x]
              
            if grad_mag<weak_th:
                mag[i_y, i_x]= 0
            elif strong_th>grad_mag>= weak_th:
                ids[i_y, i_x]= 1
            else:
                ids[i_y, i_x]= 2
       
       
    # finally returning the magnitude of
    # gradients of edges
    return mag

image = cv2.imread('../Resources/Photos/park.jpg')
print(image.shape)
cv2.imshow("park",image)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray,(7,7),sigmaX=2,sigmaY=3)

sobelx = sobel_x(gray)
sobely = sobel_y(gray)

combined_sobel = cv2.bitwise_or(sobelx,sobely)
# cv2.imshow("sobel ",combined_sobel)

canny_edge = Canny_detector(gray,50,100)
cv2.imshow("canny_edge",canny_edge)
canny = cv2.Canny(gray,50,100)
cv2.imshow("canny",canny)
cv2.waitKey(0)