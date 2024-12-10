import cv2 
import numpy as np


cam=cv2.VideoCapture(0)

while (1):
    ret, imageFrame = cam.read() 
    print(ret)
    
    hsvframe=cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
    grayframe=cv2.cvtColor(imageFrame, cv2.COLOR_BGR2GRAY)
    
    #red color
    red_lower=np.array([136,87,111],np.uint8)
    red_upper=np.array([180,255,255], np.uint8)
    red_mask=cv2.inRange(hsvframe, red_lower, red_upper)
    
    #green color
    green_lower=np.array([25, 52, 72], np.uint8)
    green_upper=np.array([102, 255, 255], np.uint8)
    green_mask=cv2.inRange(hsvframe, green_lower, green_upper)
    
    #blue color
    blue_lower=np.array([94, 80, 2], np.uint8)
    blue_upper=np.array([120, 255, 255], np.uint8)
    blue_mask=cv2.inRange(hsvframe, blue_lower, blue_upper)
    
    kernel=np.ones((5,5), "uint8")
    
    #for red color
    red_mask=cv2.dilate(red_mask, kernel)
    res_red=cv2.bitwise_and(imageFrame, imageFrame, mask=red_mask)
    
    #for green color
    green_mask=cv2.dilate(green_mask, kernel)
    res_green=cv2.bitwise_and(imageFrame, imageFrame, mask=green_mask)
    
    #for blue color
    blue_mask=cv2.dilate(blue_mask, kernel)
    res_blue=cv2.bitwise_and(imageFrame, imageFrame, mask=blue_mask)
    
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 500): 
            x, y, w, h = cv2.boundingRect(contour) 
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 2) 
            
            cv2.putText(imageFrame, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
    
    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 500): 
            x, y, w, h = cv2.boundingRect(contour) 
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
            
            cv2.putText(imageFrame, "Green Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
            
    contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 500): 
            x, y, w, h = cv2.boundingRect(contour) 
            imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2) 
            
            cv2.putText(imageFrame, "Blue Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))  
    
    
    cv2.imshow("Multiple Color Detection in Real-Time", imageFrame) 
    if cv2.waitKey(10) & 0xFF == ord('q'): 
        cam.release() 
        cv2.destroyAllWindows() 
        break   