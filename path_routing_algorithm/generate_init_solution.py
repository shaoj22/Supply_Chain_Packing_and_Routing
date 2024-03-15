'''
File: generate_init_solution.py
Project: Supply Chain Packing and Routing
Description:
-----------
get the init solution for the routing problem, not important.
-----------
Author: 626
Created Date: 2023-1009
'''


import numpy as np 


def gen_initial_sol(allplatform_listdict, distance_2dMatrix, platform_num):
    """产生初始解，从start point开始,找距离自己最近的没有参加排列的pickup point作为下一个 """
    init_sol_list = []  # 1~N的一个排列
    platform_flag_dict = {}
    for platform_dict in allplatform_listdict:
        platformCode = platform_dict["platformCode"]
        platform_flag_dict[platformCode] = 1
    i = 0
    dist_of_this_point_list = distance_2dMatrix[0].tolist()
    while len(init_sol_list) < platform_num:
        min_dist = min(dist_of_this_point_list)
        min_dist_index_of_list =\
            dist_of_this_point_list.index(min_dist)  # 在dist_of_this_point_list中的索引
        min_dist_index_array =\
            np.where(distance_2dMatrix[i] == min_dist)  # array类型,在原matrix中的索引
        if (int(min_dist_index_array[0]) in init_sol_list) or (int(min_dist_index_array[0]) == (platform_num + 1)):
            del dist_of_this_point_list[min_dist_index_of_list]
            continue
        else:
            init_sol_list.append(int(min_dist_index_array[0]))
            i = int(min_dist_index_array[0])
            dist_of_this_point_list = distance_2dMatrix[i].tolist()
    return init_sol_list

def get_random_init_sol(num, original_list, platform_num):
    """ 产生随机的初始解 """
    init_pop = self.sample_pop(num, original_list, platform_num)
    return init_pop

def sample_pop(num, original_list, platform_num):
    """
    生成num数量的个体
    :param num: 要生成的个体数
    :return indi_list: 包含num个元素，每个元素都是一个platform索引的排列
    """
    indi_list = []
    for i in range(num):
        indi = np.random.choice(original_list, size=platform_num, replace=False)
        indi_list.append(list(indi))
    return indi_list
