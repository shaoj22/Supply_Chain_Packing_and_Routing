'''
File: 3DBinPackingConstrains.py
Project: Supply Chain Consolidation and Vehicle Routing Optimization
Description:
-----------

-----------
Author: 626
Created Date: 2023-1009
'''


import json
import os
import re
import numpy as np


class Data():
    def __init__(self):
        self.customerNum = 0 # Number of pick-up nodes
        self.nodeNum = self.customerNum + 2 # Total number of nodes
        self.platform = [] # Pick-up point number and its information
        self.truckType = [[]] # Vehicle type number and its information
        self.disMatrix = [[]] # Distance Matrix
        self.boxes = [[]]

# read from a single file
def read_file(data, path):
    with open(path, 'r') as f:
        j = json.load(f)
        ad = j['algorithmBaseParamDto']  # algorithm dict

        # read platform
        platform = [1000] + [int(re.findall(r"\d+", i['platformCode'])[0]) for i in ad['platformDtoList']] + [2000]
        data.platform = platform

        # read truckType
        tl = ad['truckTypeDtoList']  # truck list
        trucks = []
        for t in tl:
            trucks.append([t['truckTypeId'],  # note this is a String, others are Numbers
                           t['length'], t['width'], t['height'], t['maxLoad']])
        data.truckType = trucks

        # read boxes
        bl = j['boxes']  # box list
        boxes = []
        for b in bl:
            boxes.append([b['spuBoxId'],  # note this is a String, others are Numbers
                          int(re.findall(r"\d+", b['platformCode'])[0]),
                          b['length'], b['width'], b['height'], b['weight']])
        data.boxes = boxes

        # read distance matrix
        dd = ad['distanceMap']  # distance dict
        disMatrix = np.zeros((len(platform), len(platform)), dtype=int)
        # TODO: catch input error, e.g. wrong key format
        for k, v in dd.items():
            p = [int(i) for i in re.findall(r"\d+", k)]  # platform number list
            if len(p) == 0:
                p = [1000, 2000]
            elif len(p) == 1:
                if 'start_point' in k:
                    p = [1000] + p
                elif 'end_point' in k:
                    p = p + [2000]
            p_id = [platform.index(i) for i in p]  # platform id list
            disMatrix[p_id[0]][p_id[1]] = v
        # print(disMatrix)  # test
        disMatrix = disMatrix.tolist()  # transform ndarray to list
        data.disMatrix = disMatrix
        # print(disMatrix) # test
        for i in range(len(data.truckType)):
            data.truckType[i].append(i)
        for i in range(len(data.boxes)):
            data.boxes[i].append(i)
    return data

def read_result_file(input_path, output_path):
    # 存储输出结果的dict
    resultInfoDict = {}
    with open(output_path) as f:
        j = json.load(f)
    solutionArray = j["solutionArray"] # 获取解集合
    solutionCarsPathArray = [] # 所有解cars路径集合
    solutionCarsLoadArray = [] # 所有解cars装载集合
    # 获取车辆路径和装载率
    for solutionNum in range(len(solutionArray)): 
        oneSolutionCarsPathArray = [] # 一个解cars路径集合
        oneSolutionCarsLoadArray = [] # 一个解cars装载集合
        for carsNum in range(len(solutionArray[solutionNum])):
            oneCarPathArray = [int(platform.replace("platform", "")) for platform in solutionArray[solutionNum][carsNum]["platformArray"]]
            # 给路径加上起点和终点
            oneCarLoadArray = solutionArray[solutionNum][carsNum]["volume"]/\
            (solutionArray[solutionNum][carsNum]["innerLength"]*\
            solutionArray[solutionNum][carsNum]["innerWidth"]*\
            solutionArray[solutionNum][carsNum]["innerHeight"])
            oneSolutionCarsPathArray.append(oneCarPathArray)
            oneSolutionCarsLoadArray.append(oneCarLoadArray)
        solutionCarsPathArray.append(oneSolutionCarsPathArray)
        solutionCarsLoadArray.append(oneSolutionCarsLoadArray)
    # 获取车辆的行驶距离
    # d = Data()
    # data = read_file(d, input_path)
    # print(data.disMatrix[0][1])
    # print(data.platform)
    # data.platform[0] = -1
    # platformSort = sorted(data.platform)
    # print(platformSort)

    # 存储结果
    resultInfoDict["loadResult"] = solutionCarsLoadArray # 装载率
    resultInfoDict["pathResult"] = solutionCarsPathArray # 路径
    
    return resultInfoDict

    
if __name__ == "__main__":
    input_path = "D:\\Desktop\\python_code\\Supply_Chain_Packing_and_Routing\\inputs\\myInstance3"
    output_path = 'D:\\Desktop\\python_code\\Supply_Chain_Packing_and_Routing\\outputs\\myInstance3'
    resultInfoDict = read_result_file(input_path, output_path)
    print(resultInfoDict)
