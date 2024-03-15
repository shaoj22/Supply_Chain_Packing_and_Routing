'''
File: 3DBinPackingConstrains.py
Project: Supply Chain Consolidation and Vehicle Routing Optimization
Description:
-----------
Judgment of several constraints in 3D bin packing problem
-----------
Author: 626
Created Date: 2023-1009
'''

from bin_packing_algorithm import general_utils as utils
import numpy as np
from bin_packing_algorithm.entity import Area
import math


def can_in_space(box_size, space_size):
    """
        判断空间是否能放下当前盒子
        :param box_size: 盒子的尺寸
        :param space_size: 空间的尺寸
    """
    for i in range(len(box_size)):
        if box_size[i] > space_size[i]:
            return False
    return True


def can_hold_box(box, space, direction):
    """
        判断空间是否可以支撑当前盒子
        :param box: 被放入的盒子
        :param space: 用来放盒子的空间
        :param direction: 箱子的摆放方向
    """
    # 如果支撑平面为空或者None则该空间为从地面起始的空间
    if not space.hold_surface:
        return True
    lx, ly, _ = utils.choose_box_direction_len(box.length, box.width,
                                               box.height, direction)
    min_coord = [space.min_coord[0], space.min_coord[1]]
    max_coord = [min_coord[0] + lx, min_coord[1] + ly]
    for area in space.hold_surface:
        # 底面一样
        if area.min_coord == min_coord and area.max_coord == max_coord \
           and area.max_layer >= 1:
            return True
        if utils.is_overlap(min_coord, max_coord, area.min_coord,
                            area.max_coord) and area.max_weight < box.weight:
            return False
    return False


def can_hold_block(block, space):
    """
        判断当前空间的支撑平面能否支撑放置的块
        :param block: 放置的块
        :param space: 放置块的空间
    """
    # 如果在地面上不需要判定
    if not space.hold_surface:
        return True
    # 支撑平面x,y方向箱子个数
    space_x_num = len(space.hold_surface[0])
    space_y_num = len(space.hold_surface)
    # 支撑平面的最大堆叠和最大承重
    max_layers = np.zeros([space_y_num, space_x_num])
    max_weights = np.zeros([space_y_num, space_x_num])
    for i in range(space_y_num):
        for j in range(space_x_num):
            max_layers[i][j] = space.hold_surface[i][j].max_layer
            max_weights[i][j] = space.hold_surface[i][j].max_weight
    # 装入块的支撑平面最大堆叠和最大承重
    new_max_layers = np.zeros([block.ny, block.nx])
    new_max_weights = np.zeros([block.ny, block.nx])
    for i in range(block.ny):
        for j in range(block.nx):
            new_max_layers[i][j] = block.hold_surface[i][j].max_layer
            new_max_weights[i][j] = block.hold_surface[i][j].max_weight
    # 支撑平面
    hold_surface = space.hold_surface
    # 块的支撑平面
    bottom_surface = block.bottom_surface
    for i in range(block.ny):
        for j in range(block.nx):
            gap_x = space.min_coord[0] - space.hold_surface[0][0].min_coord[0]
            gap_y = space.min_coord[1] - space.hold_surface[0][0].min_coord[1]
            # 获取当前放置箱子的最小和最大x,y坐标
            min_x = j * block.item_size[0] + gap_x
            min_y = i * block.item_size[1] + gap_y
            max_x = min_x + block.item_size[0]
            max_y = min_y + block.item_size[1]
            # 获取支撑平面支撑该箱子的箱子x,y方向的起始索引
            x_space = space.hold_surface[0][0].lx
            y_space = space.hold_surface[0][0].ly
            x_begin = int(min_x / x_space)
            x_end = int(math.ceil(max_x / x_space))
            y_begin = int(min_y / y_space)
            y_end = int(math.ceil(max_y / y_space))
            # 开始判断是否能支撑
            for m in range(y_begin, y_end):
                for n in range(x_begin, x_end):
                    # 如果支撑的箱子和被支撑的箱子底面积相同
                    if utils.is_same(hold_surface[m][n].min_coord,
                                     hold_surface[m][n].max_coord,
                                     bottom_surface[i][j].min_coord,
                                     bottom_surface[i][j].max_coord):
                        if max_layers[m][n] >= block.nz:
                            max_layers[m][n] -= block.nz
                            new_max_layers[i][j] = min(max_layers[m][n],
                                                       new_max_layers[i][j])
                        else:
                            return False
                    # 如果支撑的箱子和被支撑的箱子底面积不同
                    else:
                        if max_weights[m][n] >= \
                           bottom_surface[i][j].max_weight:
                            max_weights[m][n] -= bottom_surface[i][
                                j].max_weight
                            new_max_weights[i][j] = min(
                                max_weights[m][n],
                                new_max_weights[i][j],
                            )
                        else:
                            return False
    # 如果遍历完支撑平面上支撑被放置的箱子的箱子都可以支撑，则更新空间支撑平面和箱子支撑平面
    for i in range(space_y_num):
        for j in range(space_x_num):
            space.hold_surface[i][j].max_layer = max_layers[i][j]
            space.hold_surface[i][j].max_weight = max_weights[i][j]
    for i in range(block.ny):
        for j in range(block.nx):
            block.hold_surface[i][j].max_layer = new_max_layers[i][j]
            block.hold_surface[i][j].max_weight = new_max_weights[i][j]
    return True


def can_hold_block_by_area(block, space):
    """
        根据block的底面积和space的支撑面积判断是否满足支撑条件
        :param block: 放置的块
        :param space: 放置块的空间        
    """
    # 如果在地面上不需要判定
    if not space.hold_surface:
        return True
    # 计算block与space的支撑面的接触面积
    lx = min(space.min_coord[0] + block.item_size[0], space.max_coord[0])
    ly = min(space.min_coord[1] + block.item_size[1], space.max_coord[1])
    area_contact = (lx - space.min_coord[0]) * (ly - space.min_coord[1])
    # 计算block的底面积
    area_block = block.lx * block.ly
    if area_contact / area_block == 1:
        return True
    else:
        return False


def can_in_bin(bin_obj, amount, weight):
    """
        当容器有最大装载重量或金额限制时，需要判断箱子放入后能否满足约束
        :param bin_obj: 当前用来装载的容器
        :param amount: 当前箱子的金额
        :param weight: 当前箱子的重量
    """
    if weight + bin_obj.load_weight > bin_obj.max_weight:
        return False
    return True


def can_form_rectangle_block(block, residual_box_list_ind, box_list, bin_obj,
                             space):
    """
        判断能否构成满足支撑约束的块
        :param block: 需要判断的块
        :param residual_box_list_ind: 可以用来构成该块的箱子在box_list中的索引
        :param box_list: 箱子集合
        :param bin_obj: 用来装载的集装箱类
        :param space: 用来放置该块的空间
    """
    # 构成该块的集合
    packed_box_list = []
    total_weight, total_amount = 0, 0
    max_layer = np.zeros((block.ny, block.nx))
    max_weight = np.zeros((block.ny, block.nx))
    bottom_weight = np.zeros((block.ny, block.nx))
    # 已经使用的箱子索引
    used_box_ind = {}
    # 校验还未放置的箱子能否构成满足支撑约束的块
    """ begin: han add """
    box = None
    for value in residual_box_list_ind:
        # 如果该类箱子在被使用索引中，但已经使用个数量等于总数量，查看一下个
        if value in used_box_ind.keys(
        ) and box_list[value].box_num == used_box_ind[value]:
            continue
        # 如果该箱子还未在使用索引中但是已经没有可用箱子，查看下一个
        if value not in used_box_ind.keys(
        ) and box_list[value].box_num == 0:
            continue
        if box_list[value].max_layer >= block.nz - 0:
            box = box_list[value]
            box_num = used_box_ind.setdefault(value, 0)
            used_box_ind[value] = box_num + 1
            max_layer[0][0] = box.max_layer - 1
            break
    # 如果没有箱子可以使用了，则构建块失败
    if not box:
        return False, used_box_ind
    bottom_weight[0][0] += box.weight
    if 0 == block.nz - 1:
        max_weight[0][0] = box.max_weight
    packed_box_list.append(box)
    total_amount += box.amount
    total_weight += box.weight
    # 判断块是否满足集装箱的重量和金额约束
    if can_in_bin(bin_obj, total_amount, total_weight):
        hold_surface = []
        bottom_surface = []
        # 构成该块的支撑平面和底部平面
        for num_y in range(block.ny):
            temp_hold = []
            temp_hold.append(1)  # han add:表示不在地面上
            hold_surface.append(temp_hold)
        block.hold_surface = hold_surface
        block.bottom_surface = bottom_surface
        block.weight = total_weight
        block.amount = total_amount
        block.packed_box_list = packed_box_list
        return True, used_box_ind
    return False, used_box_ind


def can_get_hold_block(block, box_size_map, box_list, bin_obj, space):
    """
        是否能够获取一个块使其能够被当前空间支撑
        :param block: 当前校验的块
        :param box_size_map: 箱子提货点和尺寸与箱子索引的映射
        :param box_list: 箱子集合
        :param bin_obj: 当前用来装载的集装箱
        :param space: 当前用来装载的空间
    """
    residual_box_list_ind = box_size_map[block.item_size]
    # 判断构成的块是否满足支撑约束
    flag, used_box_ind = can_form_rectangle_block(block, residual_box_list_ind,
                                                  box_list, bin_obj, space)
    if flag:
        # 判断构成的块能否被空间支撑
        if can_hold_block_by_area(block, space):
            temp_residual_list_ind = []
            for ind in residual_box_list_ind:
                if ind in used_box_ind:
                    box_list[ind].box_num -= used_box_ind[ind]
            return True
    return False
