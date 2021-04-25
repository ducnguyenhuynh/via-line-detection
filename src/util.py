import torch.nn as nn
import cv2
import torch
from copy import deepcopy
import numpy as np
from torch.autograd import Variable
from torch.autograd import Function as F
from numpy.polynomial import Polynomial as P
from parameters import Parameters
# from src.parameters import Parameters
import math

p = Parameters()

###############################################################
##
## visualize
## 
###############################################################

def visualize_points(image, x, y):
    image = image
    image =  np.rollaxis(image, axis=2, start=0)
    image =  np.rollaxis(image, axis=2, start=0)#*255.0
    image = image.astype(np.uint8).copy()

    for k in range(len(y)):
        for i, j in zip(x[k], y[k]):
            if i > 0:
                image = cv2.circle(image, (int(i), int(j)), 2, p.color[1], -1)

    cv2.imshow("test2", image)
    cv2.waitKey(0)  

def visualize_points_origin_size(x, y, test_image, ratio_w, ratio_h):
    color = 0
    image = deepcopy(test_image)
    image =  np.rollaxis(image, axis=2, start=0)
    image =  np.rollaxis(image, axis=2, start=0)#*255.0
    image = image.astype(np.uint8).copy()

    image = cv2.resize(image, (int(p.x_size/ratio_w), int(p.y_size/ratio_h)))

    for i, j in zip(x, y):
        color += 1
        for index in range(len(i)):
            cv2.circle(image, (int(i[index]), int(j[index])), 10, p.color[color], -1)
    cv2.imshow("test2", image)
    cv2.waitKey(0)  

    return test_image

def visualize_gt(gt_point, gt_instance, ground_angle, image):
    image =  np.rollaxis(image, axis=2, start=0)
    image =  np.rollaxis(image, axis=2, start=0)#*255.0
    image = image.astype(np.uint8).copy()

    for y in range(p.grid_y):
        for x in range(p.grid_x):
            if gt_point[0][y][x] > 0:
                xx = int(gt_point[1][y][x]*p.resize_ratio+p.resize_ratio*x)
                yy = int(gt_point[2][y][x]*p.resize_ratio+p.resize_ratio*y)
                image = cv2.circle(image, (xx, yy), 10, p.color[1], -1)

    cv2.imshow("image", image)
    cv2.waitKey(0)

def visualize_regression(image, gt):
    image =  np.rollaxis(image, axis=2, start=0)
    image =  np.rollaxis(image, axis=2, start=0)*255.0
    image = image.astype(np.uint8).copy()

    for i in gt:
        for j in range(p.regression_size):#gt
            y_value = p.y_size - (p.regression_size-j)*(220/p.regression_size)
            if i[j] >0:
                x_value = int(i[j]*p.x_size)
                image = cv2.circle(image, (x_value, y_value), 5, p.color[1], -1)
    cv2.imshow("image", image)
    cv2.waitKey(0)   

def draw_points(x, y, image):
    color_index = 0
    for i, j in zip(x, y):
        color_index += 1
        if color_index > 12:
            color_index = 12
        for index in range(len(i)):
            # print( (int(i[index]), int(j[index])))
            image = cv2.circle(image, (int(i[index]), int(j[index])), 5, p.color[color_index], -1)

    return image

def draw_poly(poly, image, color):
    if poly == []:
        return image
    y =  np.linspace(256*12/20, 256, 10)
    p = np.poly1d(poly)
    x = [(p - _y).roots[0] for _y in y ]
    draw_points = (np.asarray([x, y]).T).astype(np.int32)

    cv2.polylines(image, [draw_points], False, color,3)
    return image

###############################################################
##
## calculate
## 
###############################################################
def adjust_fits(fits):
    min_y = 20
    len_fit = fits.shape[0]
    
    values_x = np.array([np.poly1d(fit)(min_y) for fit in fits ])
    order = np.argsort(values_x)
    fits_sorted = fits[order]
    if len(fits_sorted) > 3:
        fits_sorted = fits_sorted[:3]
    
    return fits_sorted

def get_steer_angle(fits):
    
    min_y = 20
    len_fit = fits.shape[0]

    if len_fit > 3:
        pass
    if len_fit >= 2:
        y = 20
        x = (np.poly1d(fits[-1])(y) + np.poly1d(fits[-2])(y)) // 2
        return_value = errorAngle((x,y))
        
        #update point in lane
        temp_y = 200
        temp_x = (np.poly1d(fits[-1])(temp_y) + np.poly1d(fits[-2])(temp_y)) // 2
        p.point_in_lane = (temp_x,temp_y)
        
        return return_value

    if len_fit == 1:# missing 1 line
        y = 20
        avaiable_fit =  np.poly1d(fits[0])
        x_avaiable = avaiable_fit(y)

        # check where do line?
        point_x = p.point_in_lane[0]
        point_y = p.point_in_lane[1]

        val = point_x - avaiable_fit(point_y)
        # print(val)
        if val > 0: # is right
            x = x_avaiable + 150
        else: # is left
            x = x_avaiable - 150
        return_value = errorAngle((x,y))
        return return_value
    return  0

def convert_to_original_size(x, y, ratio_w, ratio_h):
    # convert results to original size
    out_x = []
    out_y = []

    for i, j in zip(x,y):
        out_x.append((np.array(i)/ratio_w).tolist())
        out_y.append((np.array(j)/ratio_h).tolist())

    return out_x, out_y



def get_closest_upper_point(x, y, point, n):
    x = np.array(x)
    y = np.array(y)

    x = x[y<point[1]]
    y = y[y<point[1]]

    dis = (x - point[0])**2 + (y - point[1])**2

    ind = np.argsort(dis, axis=0)
    x = np.take_along_axis(x, ind, axis=0).tolist()
    y = np.take_along_axis(y, ind, axis=0).tolist()

    points = []
    for i, j in zip(x[:n], y[:n]):
        points.append((i,j))

    return points

def sort_along_y(x, y):
    out_x = []
    out_y = []

    for i, j in zip(x, y):
        i = np.array(i)
        j = np.array(j)

        ind = np.argsort(j, axis=0)
        out_x.append(np.take_along_axis(i, ind[::-1], axis=0).tolist())
        out_y.append(np.take_along_axis(j, ind[::-1], axis=0).tolist())
    
    return out_x, out_y

def sort_along_x(x, y):
    
    temp = np.min(y)
    try:
        min_y = temp[0]
    except:
        min_y = temp
    
        
    # print(min_y)
    fits = np.array([np.polyfit(_y,_x, 2) for _x, _y in zip(x,y)])
    # print(fits)
    values_x = np.array([np.poly1d(fit)(min_y) for fit in fits ])
    # print(values_x)
    order = np.argsort(values_x)
    print(order)
    return np.array(x)[order], np.array(y)[order]

def sort_batch_along_y(target_lanes, target_h):
    out_x = []
    out_y = []

    for x_batch, y_batch in zip(target_lanes, target_h):
        temp_x = []
        temp_y = []
        for x, y, in zip(x_batch, y_batch):
            ind = np.argsort(y, axis=0)
            sorted_x = np.take_along_axis(x, ind[::-1], axis=0)
            sorted_y = np.take_along_axis(y, ind[::-1], axis=0)
            temp_x.append(sorted_x)
            temp_y.append(sorted_y)
        out_x.append(temp_x)
        out_y.append(temp_y)
    
    return out_x, out_y


def errorAngle(point):
    carPosx , carPosy = 512//2, 254
    dstx, dsty = point

    if dstx == carPosx:
        return 0
    if dsty == carPosy:
        if dstx < carPosx:
            return -25
        else:
            return 25
    pi = math.acos(-1.0)
    dx = dstx - carPosx
    dy = carPosy - dsty
    if dx < 0: 
        angle = (math.atan(-dx / dy) * -180 / pi)/2.5
        if angle >= 16 or angle <= -16: # maybe must turn 90
            if angle > 0:
                return 25
            return -25
        return angle
    #################################################
    angle = (math.atan(dx / dy) * 180 / pi)/2.5
    if angle >= 16 or angle <= -16: # maybe must turn 90
        if angle > 0:
            return 25
        return -25
    return angle

def calcul_speed(steer_angle):
    max_speed = 70
    max_angle = 25
    if steer_angle == -10 or steer_angle == 10:
        return 0
    if steer_angle >= 1 or steer_angle <= -1:
        if steer_angle > 0:
            return max_speed - (max_speed/max_angle)*steer_angle
        else:
            return max_speed + (max_speed/max_angle)*steer_angle 

    elif steer_angle >= 4 or steer_angle <= -4:
        if steer_angle > 0:
            return 40 - (40/max_angle)*steer_angle
        else:
            return 40 + (30/max_angle)*steer_angle
    # elif steer_angle >= 10 or steer_angle <= -10:
    #     if steer_angle > 0:
    #         return max_speed - (max_speed/max_angle)*steer_angle
    #     else:
    #         return max_speed + (max_speed/max_angle)*steer_angle 
    # if steer_angle >=0:
    #     return max_speed - (max_speed/max_angle)*steer_angle
    return max_speed 

def clear_StatusObjs(StatusObjs):
    list_result = []
    for obj in StatusObjs:
        if 'i5' in obj:
            obj.remove('i5')
        if 'pne' in obj:
            obj.remove('pne')
        if 'car' in obj:
            obj.remove('car')
        if 'w65' in obj:
            obj.remove('w65')
        list_result.append(obj)
    return list_result



