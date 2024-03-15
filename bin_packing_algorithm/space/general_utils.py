"""
   Copyright (c) 2020. Huawei Technologies Co., Ltd.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import bin_packing_algorithm.general_utils as utils
from bin_packing_algorithm.entity import PackedBox, AlgorithmBox


def assign_box_2_bin(box, space, packed_bin, box_direction):
    """
        将箱子放置到容器中
        :param box: the box that has been packed
        :param space: the space packs the box
        :param packed_bin: the bin packs the box
        :param box_direction: 箱子的摆放方向
    """
    packed_bin.order += 1
    lx, ly, lz = utils.choose_box_direction_len(box.length, box.width,
                                                box.height, box_direction)
    packed_box = PackedBox(*space.min_coord, lx, ly, lz, box, packed_bin.order,
                           box_direction, 0)
    copy_box = AlgorithmBox.copy_algorithm_box(box, 1)
    if packed_bin.packed_box_list:
        packed_bin.packed_box_list.append(packed_box)
        packed_bin.box_list.append(copy_box)
    else:
        packed_bin.packed_box_list = [packed_box]
        packed_bin.box_list = [copy_box]
    packed_bin.load_volume += lx * ly * lz
    packed_bin.load_weight += box.weight
    packed_bin.load_amount += box.amount


def assign_rectangle_box_in_block(space, block, packed_bin):
    """
        将块放入空间中
        :param space: 放入的空间
        :param block: 需要放入的块
        :param packed_bin: 放入的容器
    """
    lx, ly, lz = block.item_size
    base_x, base_y, base_z = space.min_coord
    order = packed_bin.order
    paceked_box_list = block.packed_box_list
    i = 0
    for num_z in range(block.nz):
        for num_x in range(block.nx):
            for num_y in range(block.ny):
                box = paceked_box_list[i]
                i += 1
                order += 1
                direction = utils.get_box_direction(box.length, box.width,
                                                    box.height, lx, ly, lz)
                copy_box = AlgorithmBox.copy_algorithm_box(box, 1)
                packed_box = PackedBox(
                    base_x + num_x * lx, base_y + num_y * ly,
                    base_z + num_z * lz, lx, ly, lz, box, order, direction, 0)
                if packed_bin.packed_box_list:
                    packed_bin.packed_box_list.append(packed_box)
                    packed_bin.box_list.append(copy_box)
                else:
                    packed_bin.packed_box_list = [packed_box]
                    packed_bin.box_list = [copy_box]
    packed_bin.order = order
    packed_bin.load_volume += block.vol
    packed_bin.load_weight += block.weight
    packed_bin.load_amount += block.amount
