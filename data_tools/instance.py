'''
File: instance.py
Project: Supply_Chain_Packing_and_Routing
Description:
-----------
generate the random Instance(input file).
-----------
Author: 626
Created Date: 2023-1019
'''


import random
import json
import math

class Instance:
    def __init__(self, platform_num, box_num, estimateCode):
        """
        init the Instance

        Args:
            platform_num (int): number of the instance's platform
            box_num (int): number of the instance's box_num

        """
        self.platform_num = platform_num
        self.box_num = box_num
        self.platformDtoList = []
        self.truckTypeDtoList = []
        self.truckTypeMap = {}
        self.distanceMap = {}
        self.boxes = []
        self.estimateCode = estimateCode
        self.generate_json_file()

    def generate_platform(self):
        """ generate the platform """
        for plat in range(self.platform_num):
            platformDict = {}
            if plat < 9:
                platformCode = "platform0" + str(plat+1)
            else:
                platformCode = "platform" + str(plat+1)
            platformDict["platformCode"] = platformCode
            platformDict["mustFirst"] = False
            platformDict["x_coords"] = random.randint(0, 10000)
            platformDict["y_coords"] = random.randint(0, 10000)

            self.platformDtoList.append(platformDict)

    def generate_truck(self):
        """ generate the truckType """
        self.truckTypeDtoList = \
        [
            {
                "truckTypeId": "40001",
                "truckTypeCode": "R110",
                "truckTypeName": "20GP",
                "length": 5890.0,
                "width": 2318.0,
                "height": 2270.0,
                "maxLoad": 18000.0
            },
            {
                "truckTypeId": "41001",
                "truckTypeCode": "CT10",
                "truckTypeName": "40GP",
                "length": 11920.0,
                "width": 2318.0,
                "height": 2270.0,
                "maxLoad": 23000.0
            },
            {
                "truckTypeId": "42001",
                "truckTypeCode": "CT03",
                "truckTypeName": "40HQ",
                "length": 11920.0,
                "width": 2318.0,
                "height": 2600.0,
                "maxLoad": 23000.0
            }
        ]
        self.truckTypeMap = \
        {
            "40001": {
                "truckTypeId": "40001",
                "truckTypeCode": "R110",
                "truckTypeName": "20GP",
                "length": 5890.0,
                "width": 2318.0,
                "height": 2270.0,
                "maxLoad": 18000.0
            },
            "41001": {
                "truckTypeId": "41001",
                "truckTypeCode": "CT10",
                "truckTypeName": "40GP",
                "length": 11920.0,
                "width": 2318.0,
                "height": 2270.0,
                "maxLoad": 23000.0
            },
            "42001": {
                "truckTypeId": "42001",
                "truckTypeCode": "CT03",
                "truckTypeName": "40HQ",
                "length": 11920.0,
                "width": 2318.0,
                "height": 2600.0,
                "maxLoad": 23000.0
            }
        }
    
    def generate_distanceMap(self):
        """ generate the distanceMap """
        # generate the platform to platform, include end_point.
        for plat1 in range(self.platform_num):
            for plat2 in range(self.platform_num):
                if plat1 != plat2:
                    name = "{}+{}".format(
                        self.platformDtoList[plat1]["platformCode"], 
                        self.platformDtoList[plat2]["platformCode"]
                        )
                    distance = math.sqrt(
                                        (self.platformDtoList[plat1]["x_coords"]
                                          - self.platformDtoList[plat2]["x_coords"])**2
                                        + ( 
                                        self.platformDtoList[plat1]["y_coords"]
                                          - self.platformDtoList[plat2]["y_coords"]
                                        )**2
                                        )
                    self.distanceMap[name] = distance
            # generate the platform to end_point.
            name = "{}+{}".format(
                self.platformDtoList[plat1]["platformCode"],
                "end_point"
            )
            self.distanceMap[name] = random.randint(2000, 20000)
        # generate the start_point to platform.
        for plat in range(self.platform_num):
            name = "{}+{}".format(
                "start_point",
                self.platformDtoList[plat]["platformCode"]
            )
            self.distanceMap[name] = random.randint(2000, 20000)

    def generate_boxes(self):
        # random split the boxes into platform's number.
        def random_split_boxes(box_num, platform_num):
            partitions = sorted(random.sample(range(1, box_num), platform_num - 1))
            result = [partitions[0]]
            for i in range(1, platform_num - 1):
                result.append(partitions[i] - partitions[i - 1])
            result.append(box_num - partitions[-1])
            return result
        if self.platform_num == 1 or self.platform_num == 5:
            if self.platform_num == 1:
                boxInEachPlatform = [self.box_num]
            else:
                boxInEachPlatform = [self.box_num, 0, 0, 0, 0]
        else:
            boxInEachPlatform = random_split_boxes(self.box_num, self.platform_num)
        # generate random boxes for each platform.
        box_num = 1
        for plat in range(self.platform_num):
            for box in range(boxInEachPlatform[plat]):
                boxInfo = {}
                boxInfo["spuBoxId"] = str(box_num)
                box_num += 1
                boxInfo["platformCode"] = self.platformDtoList[plat]["platformCode"]
                boxInfo["length"] = random.randint(5,15)*100
                boxInfo["width"] = random.randint(4,12)*100
                boxInfo["height"] = random.randint(3,9)*100
                boxInfo["weight"] = random.randint(10,40)
                self.boxes.append(boxInfo)
            
    def generate_json_file(self):
        """ generate the instance's json file """
        json_file = {}
        estimateCode = self.estimateCode
        algorithmBaseParamDto = {}
        # generate the platform, truck, boxes.
        self.generate_platform()
        self.generate_truck()
        self.generate_distanceMap()
        self.generate_boxes()
        # generate the json file.
        algorithmBaseParamDto["platformDtoList"] = self.platformDtoList
        algorithmBaseParamDto["truckTypeDtoList"] = self.truckTypeDtoList
        algorithmBaseParamDto["truckTypeMap"] = self.truckTypeMap
        algorithmBaseParamDto["distanceMap"] = self.distanceMap
        json_file["estimateCode"] = estimateCode
        json_file["algorithmBaseParamDto"] = algorithmBaseParamDto
        json_file["boxes"] = self.boxes
        # save into json file.
        file_path = "D:\\Desktop\\python_code\\Supply_Chain_Packing_and_Routing\\inputs\\" + self.estimateCode
        with open(file_path, 'w') as f:
            json.dump(json_file, f, indent=4)

if __name__ == "__main__":
    for i in range(1, 6):
        instance_name = "myInstance" + str(i)
        instance = Instance(8*i, 200*i, instance_name)
    # print("platform:", instance.platformDtoList)
    # print("truckTypeDtoList:", instance.truckTypeDtoList)
    # print("truckTypeMap:", instance.truckTypeMap)
    # print("distanceMap:", instance.distanceMap)
    # print("boxes:", instance.boxes)
