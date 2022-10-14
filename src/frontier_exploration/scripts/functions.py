#!/usr/bin/env python3

from turtle import speed
import numpy as np
from numpy import array
from numpy import floor
from numpy.linalg import norm
from numpy import inf
import sys
import os
from collections import deque as queue
from skimage.segmentation import flood

sys.path.append(os.path.dirname(__file__))


def index_of_point(mapData, Xp):
    resolution = mapData.info.resolution
    Xstartx = mapData.info.origin.position.x
    Xstarty = mapData.info.origin.position.y
    width = mapData.info.width
    Data = mapData.data
    index = int((floor((Xp[1]-Xstarty)/resolution) *
                 width)+(floor((Xp[0]-Xstartx)/resolution)))
    return index

def index_2d_of_point(mapData, Xp):
    resolution = mapData.info.resolution
    Xstartx = mapData.info.origin.position.x
    Xstarty = mapData.info.origin.position.y
    width = mapData.info.width
    Data = mapData.data
    return [int(floor((Xp[1]-Xstarty)/resolution)), int(floor((Xp[0]-Xstartx)/resolution))]

def point_of_index(mapData, i):
    y = mapData.info.origin.position.y + \
        (i/mapData.info.width)*mapData.info.resolution
    x = mapData.info.origin.position.x + \
        (i-(i/mapData.info.width)*(mapData.info.width))*mapData.info.resolution
    return array([x, y])


def informationGain(mapData, point, r):
    r_region = int(floor(r/mapData.info.resolution))
    map2 = np.reshape(mapData.data, (mapData.info.width, mapData.info.height))
    x, y = index_2d_of_point(mapData, point)
    x_min = max(x-r_region, 0)
    x_max = min(x+r_region, mapData.info.width)
    y_min = max(y-r_region, 0)
    y_max = min(y+r_region, mapData.info.height)

    area = map2[x_min:x_max, y_min:y_max]
    seed = (min(x,x_max-1)-x_min, min(y,y_max-1)-y_min)
    mask = flood(area, seed, tolerance=1)

    contains_free = np.array(mask==1) & np.array(area==0)
    if np.any(contains_free):
        info_mask = np.array(mask==1) & np.array(area==1)
        infoGain = np.sum(info_mask)
    else:
        return 0

    # infoGain = 0
    # for i in range(x_min, x_max):
    #     for j in range(y_min, y_max):
    #         if map2[i][j] == -1:
    #             infoGain += 1   
    # if infoGain == 0:
    #     return infoGain


    # infoGain = 0
    # index = index_of_point(mapData, point)  # flattened index of point
    # r_region = int(r/mapData.info.resolution)  # region radius in cells
    # init_index = index-r_region*(mapData.info.width+1)
    # corners = [point[0]-r, point[0]+r, point[1]-r, point[1]+r] # in map coords

    # q = queue()
    # q.append(index)
    # visited = set()

    # while (len(q) > 0):
    #     cell = q.popleft()

    #     neighbors = [cell-1, cell+1, cell-mapData.info.width, cell-mapData.info.width-1,
    #                 cell-mapData.info.width+1, cell+mapData.info.width, cell+mapData.info.width-1,
    #                 cell+mapData.info.width+1]
    #     for n in neighbors:
    #         map_n = point_of_index(mapData, n)
    #         if n not in visited and n >=0 and n < len(mapData.data) and map_n[0] >= corners[0] and map_n[0] <= corners[1] and map_n[1] >= corners[2] and map_n[1] <=corners[3]:
    #             visited.add(n)
    #             if mapData.data[n] != 1:
    #                 q.append(n)
    #             if mapData.data[n] == -1:
    #                 infoGain += 1

    # infoGain = 0
    # index = index_of_point(mapData, point)
    # r_region = int(r/mapData.info.resolution)
    # init_index = index-r_region*(mapData.info.width+1)
    # for n in range(0, 2*r_region+1):
    #     start = n*mapData.info.width+init_index
    #     end = start+2*r_region
    #     limit = ((start/mapData.info.width)+2)*mapData.info.width
    #     for i in range(start, end+1):
    #         if (i >= 0 and i < limit and i < len(mapData.data)):
    #             if(mapData.data[i] == -1 and norm(array(point)-point_of_index(mapData, i)) <= r):
    #                 infoGain += 1
    return infoGain/((x_max-x_min)*(y_max-y_min))


def discount(mapData, assigned_pt, centroids, infoGain, r):
    index = index_of_point(mapData, assigned_pt)
    r_region = int(r/mapData.info.resolution)
    init_index = index-r_region*(mapData.info.width+1)
    for n in range(0, 2*r_region+1):
        start = n*mapData.info.width+init_index
        end = start+2*r_region
        limit = ((start/mapData.info.width)+2)*mapData.info.width
        for i in range(start, end+1):
            if (i >= 0 and i < limit and i < len(mapData.data)):
                for j in range(0, len(centroids)):
                    current_pt = centroids[j]
                    if (mapData.data[i] == -1 and norm(point_of_index(mapData, i)-current_pt) <= r and norm(point_of_index(mapData, i)-assigned_pt) <= r):
                        # this should be modified, subtract the area of a cell, not 1
                        infoGain[j] -= 1
    return infoGain


def pathCost(path):
    if (len(path) > 0):
        i = len(path)/2
        p1 = array([path[i-1].pose.position.x, path[i-1].pose.position.y])(2*r_region)**2
        p2 = array([path[i].pose.position.x, path[i].pose.position.y])
        return norm(p1-p2)*(len(path)-1)
    else:
        return inf


def unvalid(mapData, pt):
    index = index_of_point(mapData, pt)
    r_region = 5
    init_index = index-r_region*(mapData.info.width+1)
    for n in range(0, 2*r_region+1):
        start = n*mapData.info.width+init_index
        end = start+2*r_region
        limit = ((start/mapData.info.width)+2)*mapData.info.width
        for i in range(start, end+1):
            if (i >= 0 and i < limit and i < len(mapData.data)):
                if (mapData.data[i] == 1):
                    return True
    return False


def Nearest(V, x):
    n = inf
    i = 0
    for i in range(0, V.shape[0]):
        n1 = norm(V[i, :]-x)
        if (n1 < n):
            n = n1
            result = i
    return result


def Nearest2(V, x):
    n = inf
    result = 0
    for i in range(0, len(V)):
        n1 = norm(V[i]-x)

        if (n1 < n):
            n = n1
    return i


def gridValue(mapData, Xp):
    resolution = mapData.info.resolution
    Xstartx = mapData.info.origin.position.x
    Xstarty = mapData.info.origin.position.y

    width = mapData.info.width
    Data = mapData.data
    # returns grid value at "Xp" location
    # map data:  100 occupied      -1 unknown       0 free
    index = (floor((Xp[1]-Xstarty)/resolution)*width) + \
        (floor((Xp[0]-Xstartx)/resolution))

    if int(index) < len(Data):
        return Data[int(index)]
    else:
        return 100
