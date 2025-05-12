import cv2
import math
from collections import Counter
import numpy as np

def get_circle_size(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold_value1 = 185
    ret, thresholded_image1 = cv2.threshold(gray_image, threshold_value1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded_image1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    image_shape = image.shape[0]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w>image_shape*0.5 and h >image_shape*0.5:
            return (h+w)/2
        
        
def find_space(x1,y1,x2,y2,height,width,Is_Dark,gray):
    #global Dark_Distance,White_Distance,height,width,Is_Dark,gray
    temp1=y2-y1
    temp2=x2-x1
    Dark_Distance=list()
    White_Distance=list()
    if temp1==0: 
        y=y1
        Being_Dark=True
        start_point=(0,y)
        for x in range(0,width):
            if gray[y][x]<=Is_Dark: # is dark
                if not Being_Dark:
                    Being_Dark=True
                    end_point=(x,y)
                    distance=math.sqrt((end_point[0]-start_point[0])**2+(end_point[1]-start_point[1])**2)
                    White_Distance.append(distance)
                    start_point=end_point
            else: # is white
                if Being_Dark:
                    Being_Dark=False
                    end_point=(x,y)
                    distance=math.sqrt((end_point[0]-start_point[0])**2+(end_point[1]-start_point[1])**2)
                    Dark_Distance.append(distance)
                    start_point=end_point
    elif temp2==0:
        x=x1
        Being_Dark=True
        start_point=(x,0)
        for y in range(0,height):
            if gray[y][x]<=Is_Dark: # is dark
                if not Being_Dark:
                    Being_Dark=True
                    end_point=(x,y)
                    distance=math.sqrt((end_point[0]-start_point[0])**2+(end_point[1]-start_point[1])**2)
                    White_Distance.append(distance)
                    start_point=end_point
            else: # is white
                if Being_Dark:
                    Being_Dark=False
                    end_point=(x,y)
                    distance=math.sqrt((end_point[0]-start_point[0])**2+(end_point[1]-start_point[1])**2)
                    Dark_Distance.append(distance)
                    start_point=end_point
    else:
        # ax+b=y
        a=temp1/temp2
        b=y1-a*x1
        Being_Dark=True
        temp=round(b)

        if temp>=0:
            if temp>=height:
                start_x=math.floor((int(height-1)-b)/a)
                start_point=(start_x,height-1)
            else:
                start_x=0
                start_point=(start_x,temp)
        else:
            start_x=round(-b/a)
            start_point=(start_x,0)

        for x in range(start_x,width):
            y=round(a*x+b)
            if not 0<=y<height:break
            if gray[y][x]<=Is_Dark: # is dark
                if not Being_Dark:
                    Being_Dark=True
                    end_point=(x,y)
                    distance=math.sqrt((end_point[0]-start_point[0])**2+(end_point[1]-start_point[1])**2)
                    White_Distance.append(distance)
                    start_point=end_point
            else: # is white
                if Being_Dark:
                    Being_Dark=False
                    end_point=(x,y)
                    distance=math.sqrt((end_point[0]-start_point[0])**2+(end_point[1]-start_point[1])**2)
                    Dark_Distance.append(distance)
                    start_point=end_point
    return White_Distance,Dark_Distance

def Finding_ans(List,min_ans,max_ans):
    Frequency_Board=dict(Counter(List))
    #Check_Board=dict()
    Count=0
    Sum=0
    for i in Frequency_Board:
        if min_ans<=i<=max_ans:
            Sum+=i*Frequency_Board[i]
            Count+=Frequency_Board[i]
            #Check_Board[i]=Frequency_Board[i]
    Ans=Sum/Count
    return Ans

def black_percentage(image):
    total_pixels = image.size
    black_pixels = np.count_nonzero(image == 0)
    percentage_black = (black_pixels / total_pixels) * 100
    return percentage_black