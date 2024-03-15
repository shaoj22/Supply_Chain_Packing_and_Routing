


import json
import re
import os
import numpy as np


# read from a directory
def read_data(path):
    fs = os.walk(path)
    dl = []  # data_list
    for path, _, fl in fs:
        for f in fl:
            data = Data()
            dl.append(read_file(data, path + f))
    return dl


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

class Data():
    def __init__(self):
        self.customerNum = 0 # Number of pick-up nodes
        self.nodeNum = self.customerNum + 2 # Total number of nodes
        self.platform = [] # Pick-up point number and its information
        self.truckType = [[]] # Vehicle type number and its information
        self.disMatrix = [[]] # Distance Matrix
        self.boxes = [[]]

if __name__ == '__main__':
    d = Data()
    # test reading from a single file
    data = read_file(d, 'D:\\Desktop\\python_code\\track1_submission\\inputs\\E1595638696418')
    print(data.nodeNum)
    print(data.platform)
    print(len(data.boxes))

    # test reading from a directory
    # dl = read_data('./evaluation/inputs/')
