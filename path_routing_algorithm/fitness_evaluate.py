'''
File: fitness_evaluate.py
Project: Supply Chain Packing and Routing
Description:
-----------
some functions for fitness evaluate.
-----------
Author: 626
Created Date: 2023-1009
'''


def get_sub_sol_according2_mustfirst_points(sol_list, allplatform_listdict):
    """
    倒序检查sol_list中的bondedPoints,按bondedPoints对sol_list进行切分;
    i.e.,从倒数第一个bondedPoint到最后的point是一个sub_sol_list,
    从倒数第二个bondedPoint到倒数第一个bondedPoint(不含)是一个sub_sol_list.
    @return: sub_sol_listlist: 二维list，每行存储一个platform序列
    """
    sub_sol_listlist = []
    end_pointer = len(sol_list)
    platform_listdict = allplatform_listdict
    for platform_index in reversed(sol_list):
        if platform_listdict[int(platform_index) - 1]["mustFirst"]:
            start_pointer = sol_list.index(platform_index)
            sub_sol_list = sol_list[start_pointer:end_pointer]
            sub_sol_listlist.append(sub_sol_list)
            end_pointer = start_pointer
    if platform_listdict[sol_list[0]-1]["mustFirst"] is not True:
        sub_sol_listlist.append(sol_list[0:end_pointer])
    return sub_sol_listlist

def sols_2_routes(sol_2dList, allplatform_listdict):
    """
    将sol_2dList中以platform索引表示的路径转化为以platformCode表示的route;
    :param sol_2dList: 每个元素都是一条以bonded warehouse开头的路径(路径中以platform在输入文件中的索引(这里是从1开始而不是从0开始表示第一个platform)表示该platform)
    """
    routes = []
    for sol in sol_2dList:
        route = []
        for index in sol:
            platformCode = allplatform_listdict[index-1]["platformCode"]
            route.append(platformCode)
        routes.append(route)
    return routes

def direct_route(platform_num, allplatform_listdict, pack_patterns_dictlistdictlistAlgorithmBox, bin):
        direct_route_truck_listdict = []  # 存储该order中所有的direct route
        for i in range(platform_num):
            platformCode = allplatform_listdict[i]["platformCode"]
            FLP_of_this_platform_dict =\
             pack_patterns_dictlistdictlistAlgorithmBox[platformCode][0]
            FLP_num_of_this_platform = FLP_of_this_platform_dict["1C-FLPNum"]
            for j in range(FLP_num_of_this_platform):
                truck = {}  # 每一辆truck都包含了一条route,所装载的boxes以及truck自己的数据
                truck["truckTypeId"] = bin.truck_type_id
                truck["truckTypeCode"] = bin.truck_type_code
                truck["piece"] = FLP_of_this_platform_dict["boxes_num"][j]  # 一共装载了多少boxes,platformCode定位某个点,0定位1C-FLP装载模式
                truck["volume"] = FLP_of_this_platform_dict["total_boxes_volume"][j]  # Total volume (mm3) of boxes packed in this truck.
                truck["weight"] = FLP_of_this_platform_dict["total_boxes_weight"][j]  # Total weight (kg) of the boxes packed in this truck.
                truck["innerLength"] = bin.length  # Truck length (mm). Same as the input file.
                truck["innerWidth"] = bin.width  # Truck width
                truck["innerHeight"] = bin.height  # Truck height
                truck["maxLoad"] = bin.max_weight  # Carrying capacity of the truck (kg).
                truck["platformArray"] = [platformCode]  # direct route. Huawei文件pack.py的_gen_res()中提供的方法可参考
                spu_list = []
                for algorithm_box in FLP_of_this_platform_dict["1C-FLPs"][j]:  # 当前没有考虑box的具体装载情况,后面应该还要改成packed_box,参考Huawei文件中pack.py中_gen_res()
                    spu = {}
                    spu["spuId"] = algorithm_box.box_id
                    spu["direction"] = 100  # 100 or 200,先随便填一个.这个在AlgorithmBox类中没有,存在于PackedBox中
                    spu["x"] = 0  # 同spu["direction"]
                    spu["y"] = 0
                    spu["z"] = 0
                    spu["order"] = 0  # 同spu["direction"]. Order of the box being packed.
                    spu["length"] = algorithm_box.length
                    spu["width"] = algorithm_box.width
                    spu["height"] = algorithm_box.height
                    spu["weight"] = algorithm_box.weight
                    spu["platformCode"] = algorithm_box.platform
                    spu_list.append(spu)
                # spu_list.sort(key=lambda box: box['order'])  # 按‘order’字段进行排序,当前暂不需要
                truck["spuArray"] = spu_list  # 存储装载的boxes
                direct_route_truck_listdict.append(truck)
        return direct_route_truck_listdict

def get_split_indication(
    platformCode, prev_platformCode,\
    truck_volume_load, truck_weight_load,\
    pack_patterns_dictlistdictlistAlgorithmBox,\
    distance_2dMatrix, platform_num, bin, allplatform_listdict,\
    max_split_costs=float('inf')):
    """
    当truck容量不足以装下下一个点的所有1C-SP货物时，根据max_split_costs确定split_indication值.
    依照此函数返回的split_indication来决定是否对下个点进行split，
    若split_indication = 1, 则卡车访问下个点; =0, 则不访问下个点
    """
    split_indication = float('inf')
    pack_patterns = pack_patterns_dictlistdictlistAlgorithmBox
    dist_map = distance_2dMatrix
    N = platform_num
    truck_remaining_v_capacity = bin.volume - truck_volume_load
    truck_remaining_w_capacity = bin.max_weight - truck_weight_load
    boxes_total_v = pack_patterns[platformCode][1]["total_boxes_volume"]
    boxes_total_w = pack_patterns[platformCode][1]["total_boxes_weight"]
    if truck_remaining_v_capacity < boxes_total_v or\
      truck_remaining_w_capacity < boxes_total_w:
        prev_platformCode_index = next(index for (index, d) in\
          enumerate(allplatform_listdict) if d["platformCode"] == prev_platformCode)
        platformCode_index = next(index for (index, d) in\
          enumerate(allplatform_listdict) if d["platformCode"] == platformCode)
        d_cprev_c = dist_map[prev_platformCode_index+1][platformCode_index+1]
        d_c_endpoint = dist_map[platformCode_index+1][N+1]
        d_cprev_endpoint = dist_map[prev_platformCode_index+1][N+1]
        if (d_cprev_c + d_c_endpoint) / d_cprev_endpoint <= max_split_costs:
            split_indication = 1
        else:
            split_indication = 0
    return split_indication

def get_total_boxes(sub_sol_list, allplatform_listdict, pack_patterns_dictlistdictlistAlgorithmBox, order, useFLP=None):
    """
        :param useFLP: True: use Full Load Pattern; False: not use FLP, platforms' SP(segment pattern) will include platforms' all boxes
    """
    remaining_boxes_list = []
    remaining_boxes_num = 0
    for platform_index in sub_sol_list:
        platformCode = allplatform_listdict[int(platform_index) - 1]["platformCode"]
        if useFLP:
            algoBoxes_list =\
                pack_patterns_dictlistdictlistAlgorithmBox[platformCode][1]["1C-SP"]
            # 对boxes按底面积和体积降序排列
            algoBoxes_list.sort(key=lambda box: (box.length*box.width, box.height, box.width, box.length), reverse=True)
        else:
            algoBoxes_list = \
                order.boxes_by_platform_dictlistAlgorithmBox[platformCode]
            # 对boxes按底面积和体积降序排列
            algoBoxes_list.sort(key=lambda box: (box.length*box.width, box.height, box.width, box.length), reverse=True)
        for algoBox in algoBoxes_list:
            remaining_boxes_list.append(algoBox)
        remaining_boxes_num += len(algoBoxes_list)
    return remaining_boxes_list, remaining_boxes_num

def decode_sub_sol(sub_sol_list, allplatform_listdict,\
                   pack_patterns_dictlistdictlistAlgorithmBox,\
                   order, max_split_costs, bin, distance_2dMatrix, platform_num,\
                   useFLP=None):
    """
        :param useFLP: True: use Full Load Pattern; False: not use FLP, platforms' SP(segment pattern) will include platforms' all boxes
    """
    total_boxes_listAlgorithmBox, total_boxes_num = get_total_boxes(sub_sol_list,\
                                                                    allplatform_listdict,\
                                                                    pack_patterns_dictlistdictlistAlgorithmBox,\
                                                                    order, useFLP)
    boxes_pointer = 0  # 记录下次应该装载的box的索引
    loaded_boxes_num = 0  # 记录实际已经装载的boxes数量
    truck_list = []
    max_split_costs = max_split_costs  # 先临时定一个来测试
    while boxes_pointer < total_boxes_num:
        truck = {}
        truck["truckTypeId"] = bin.truck_type_id
        truck["truckTypeCode"] = bin.truck_type_code
        truck["piece"] = 0  # 一共装载了多少boxes,platformCode定位某个点,0定位1C-FLP装载模式
        truck["volume"] = 0  # Total volume (mm3) of boxes packed in this truck.
        truck["weight"] = 0  # Total weight (kg) of the boxes packed in this truck.
        truck["innerLength"] = bin.length  # Truck length (mm). Same as the input file.
        truck["innerWidth"] = bin.width  # Truck width
        truck["innerHeight"] = bin.height  # Truck height
        truck["maxLoad"] = bin.max_weight  # Carrying capacity of the truck (kg).
        truck["platformArray"] = []
        truck["spuArray"] = []
        truck_volume_load = 0  # 装载的boxes的体积
        truck_weight_load = 0  # 装载的boxes的重量
        spu_list = []
        platform_list = []
        split_indication = float('inf')
        for i in range(boxes_pointer, total_boxes_num):  # 将boxes一个一个地装载
            algo_box = total_boxes_listAlgorithmBox[i]
          # split_indication>1表示当前truck的剩余容量可以容纳下个点的所有货物;
          # split_indication=1表示当前truck的剩余不足以容纳下个点的所有货物，
          # 但允许下个点的(SP模式的)货物进行split，将一部分放到该truck中.
            platformCode = algo_box.platform
            if i != boxes_pointer and platformCode != spu_list[-1]["platformCode"]:
                prev_platformCode = spu_list[-1]["platformCode"]
                split_indication = get_split_indication(platformCode, prev_platformCode,\
                                                        truck_volume_load, truck_weight_load,\
                                                        pack_patterns_dictlistdictlistAlgorithmBox,\
                                                        distance_2dMatrix, platform_num,\
                                                        bin, allplatform_listdict,\
                                                        max_split_costs)
          # 更新truck装载的volume和weight
            truck_volume_load += algo_box.volume
            truck_weight_load += algo_box.weight
          # 当truck可容纳下个点所有货物或者下个点允许split,且未超过装载能力则装载
            if (split_indication >= 1 and\
              truck_volume_load < bin.volume and\
              truck_weight_load < bin.max_weight):
              # 将该box载入该truck
                spu = {}
                spu["spuId"] = algo_box.box_id
                spu["direction"] = 100  # 100 or 200,先随便填一个.这个在AlgorithmBox类中没有,存在于PackedBox中
                spu["x"] = 0  # 同spu["direction"]
                spu["y"] = 0
                spu["z"] = 0
                spu["order"] = i  # 同spu["direction"]. Order of the box being packed.
                spu["length"] = algo_box.length
                spu["width"] = algo_box.width
                spu["height"] = algo_box.height
                spu["weight"] = algo_box.weight
                spu["platformCode"] = algo_box.platform
                spu_list.append(spu)
              # 更新路径platform_list
                if spu["platformCode"] not in platform_list:
                    platform_list.append(spu["platformCode"])
                loaded_boxes_num += 1
            else:  # 超过装载能力则进行下一辆车
                truck["spuArray"] = spu_list
                truck["platformArray"] = platform_list
                truck["volume"] = truck_volume_load - algo_box.volume
                truck["weight"] = truck_weight_load - algo_box.weight
                truck["piece"] = i - boxes_pointer
                if truck["piece"] != len(spu_list):
                    raise Exception("Wrong.\
                      The vaule of truck[\"piece\"] should equal to\
                        the value of (i - boxes_pointer).")
                truck_list.append(truck)
                boxes_pointer = i
                break  # 跳出for循环，执行后续while循环语句
        if loaded_boxes_num == total_boxes_num:
            truck["spuArray"] = spu_list
            truck["platformArray"] = platform_list
            truck["volume"] = truck_volume_load
            truck["weight"] = truck_weight_load
            truck["piece"] = len(spu_list)
            truck_list.append(truck)
            boxes_pointer = loaded_boxes_num
    return truck_list

def decode_sol_SDVRP(sol_list,\
                    allplatform_listdict,\
                    pack_patterns_dictlistdictlistAlgorithmBox,\
                    order, max_split_costs, bin, distance_2dMatrix, platform_num,\
                    useFLP=None):
    """
    我们将路径分成了direct route和split delivery route,
    该方法解码split delivery route
    注意将解码后的解添加进self.res["solutionArray"]时是哪一级list
    :param sol_list: 一个platform索引(在输入文件中的索引，代码是从1开始)的permutation,
                    e.g., [1,2,3,4,5]表示input文件中(即self.data)["algorithmBaseParamDto"]["platformDtoList"]中
                    第1,2,3,4,5个platform, 而第1个platform的platformCode可能是platform06(而不是platform01)
    :param useFLP: True: use Full Load Pattern; False: not use FLP, platforms' SP(segment pattern) will include platforms' all boxes
    """
    truck_list = []
    sub_sol_listlist = get_sub_sol_according2_mustfirst_points(sol_list, allplatform_listdict)
    for sub_sol_list in sub_sol_listlist:
        sub_truck_list = decode_sub_sol(sub_sol_list, allplatform_listdict,\
               pack_patterns_dictlistdictlistAlgorithmBox,\
               order, max_split_costs, bin, distance_2dMatrix, platform_num,\
               useFLP=None)
        for truck in sub_truck_list:
            truck_list.append(truck)
    return truck_list

def get_a_decodedsol(sol_list,\
                    platform_num, allplatform_listdict,\
                    pack_patterns_dictlistdictlistAlgorithmBox,\
                    bin, order, max_split_costs, distance_2dMatrix,\
                    useFLP=None):
    """ 
        既包含direct routes,也包含SDVRP routes
        :param sol_list: 一个platform索引(在输入文件中的索引，代码是从1开始)的permutation,
                    e.g., [1,2,3,4,5]表示input文件中(即self.data)["algorithmBaseParamDto"]["platformDtoList"]中
                    第1,2,3,4,5个platform, 而第1个platform的platformCode可能是platform06(而不是platform01)
        :param useFLP: True: use Full Load Pattern; False: not use FLP, platforms' SP(segment pattern) will include platforms' all boxes
    """
    truck_list = []
    if useFLP:
        truck_list_direct_trip = direct_route(platform_num,\
                                            allplatform_listdict,\
                                            pack_patterns_dictlistdictlistAlgorithmBox,\
                                            bin)
        truck_list_SDVRP = decode_sol_SDVRP(sol_list,\
                    allplatform_listdict,\
                    pack_patterns_dictlistdictlistAlgorithmBox,\
                    order, max_split_costs, bin, distance_2dMatrix, platform_num,\
                    useFLP=None)
        for truck in truck_list_direct_trip:
            truck_list.append(truck)
        for truck in truck_list_SDVRP:
            truck_list.append(truck)
    else:
        truck_list_SDVRP = decode_sol_SDVRP(sol_list,\
                    allplatform_listdict,\
                    pack_patterns_dictlistdictlistAlgorithmBox,\
                    order, max_split_costs, bin, distance_2dMatrix, platform_num,\
                    useFLP=None)
        for truck in truck_list_SDVRP:
            truck_list.append(truck)
    return truck_list

def get_f1f2_values_one_sol(truck_list,
                            allplatform_listdict,
                            distance_2dMatrix
                            ):
    truck_num = len(truck_list)
    loading_rate_sum = 0
    route_len_sum = 0
    for truck_dict in truck_list:
      # 计算f1
        v_rate = truck_dict["volume"] /\
            (truck_dict["innerLength"]*truck_dict["innerWidth"]*truck_dict["innerHeight"])
        w_rate = truck_dict["weight"] / (truck_dict["maxLoad"])
        loading_rate = max(v_rate, w_rate)
        loading_rate_sum += loading_rate
      # 计算f2
        route_len = 0
        start_platform_index = 0
        end_platform_index = 0
        platform_list = allplatform_listdict
        for platformCode in truck_dict["platformArray"]:
            platform_index = next(index for (index, d)\
              in enumerate(platform_list) if d["platformCode"] == platformCode)
            end_platform_index = platform_index + 1
            route_len += distance_2dMatrix[start_platform_index][end_platform_index]
            start_platform_index = end_platform_index
        route_len_sum += route_len
    f1 = 1 - loading_rate_sum / truck_num
    f2 = route_len_sum
    return f1, f2