
""" 与华为原先的方法相比较，主要不同在block,这里组成block时只用1个箱子，那实际上这里的block就是单个箱子，而原先的代码中的block是长宽高有多个相同尺寸的箱子组成一个长方体;与packing_algorithms2.py相比较，该源代码没有后面对地面空间的组合转化过程 """
import sys
sys.path.append("..")
import copy
from bin_packing_algorithm.sequence.others import BlockHeuristic
from bin_packing_algorithm.entity import PackedBin, Space, CoordinateItem
from bin_packing_algorithm.space.simple_space import SimpleSpace
from bin_packing_algorithm.space import general_utils as space_utils
import bin_packing_algorithm.general_utils as utils


def update_box_size_and_avail_map(box_size_map, box_list, avail, block):
    """
        更新box_size_map和avail两个字典
        :param box_size_map: 箱子尺寸（某一个方向）与该箱子在box_list中下标的映射
        :param box_list: 箱子集合
        :param avail: 箱子尺寸（某一个方向）与对应箱子数量的映射
        :param block: 此次装载的块
    """
    block_size = block.item_size
    sizes = []
    # for i in range(6):
    for i in range(2):
        temp_size = utils.choose_box_direction_len(*block_size, i)
        sizes.append(temp_size)
    for size in sizes:
        if size in box_size_map.keys():
            temp_ind = []
            total_num = 0
            for ind in box_size_map[size]:
                if box_list[ind].box_num > 0:
                    temp_ind.append(ind)
                    total_num += box_list[ind].box_num
            box_size_map[size] = temp_ind
            avail[size] = total_num


def filter_space(packed_bin: PackedBin):  # (Han:判断空间有没有被box遮蔽)
    space_list = packed_bin.space_obj.space_list
    box_list = packed_bin.packed_box_list
    new_space_list = []
    for space in space_list:
        shelter = False
        for box in box_list:
            if (box.x > space.min_coord[0]
                    and box.y < space.max_coord[1]
                    and box.y + box.ly > space.min_coord[1]
                    and box.z < space.max_coord[2]
                    and box.z + box.lz > space.min_coord[2]):
                shelter = True
                break
        if not shelter:
            new_space_list.append(space)
    packed_bin.space_obj.space_list = new_space_list


class SSBH:
    """
    Simple Space Block Heuristic.
    """
    @classmethod
    def pack(cls,
             bin_obj,
             box_list,
             only_one_box=False,
             last_bin=None):
        """
            优先装载卡车的接口，即一辆卡车不能装载了才装下一辆车
            :param bin_obj: 用来装载的容器
            :param box_list: 被装载的箱子list表示
            :param only_one_box: 是否只能一个箱子去装载
            :param last_bin: 上一次装载的容器
        """
        packed_bin_list = []
        # box_list为空则没有需要装载的箱子
        if not box_list:
            return packed_bin_list
        # 使用BlockHeuristic方法进行装载
        box_list = copy.deepcopy(box_list)
        bh = BlockHeuristic(bin_obj, box_list)  # Han: BlockHeuristic是一个类, 那么bh.box_size_map是干嘛的呢？
        # 生成块集合并排序
        block_list = bh.gen_rectangle_block()
        # block_list.sort(
        #     key=lambda x: (x.lz * x.ly, x.lz, x.ly, x.lx), reverse=True)  # 原始语句
        block_list.sort(
            key=lambda x: (x.lx * x.ly, x.lz, x.lx, x.ly), reverse=True)  # han add
        num_residual = 0
        for box in box_list:
            num_residual += box.box_num
        total_box_num = num_residual
        avail = {}
        for size, indexes in bh.box_size_map.items():
            total_num = 0
            for ind in indexes:
                total_num += box_list[ind].box_num
            avail[size] = total_num
        packed_box_num = 0
        # 若还有箱子没有装则持续装载
        last_num_residual = -1
        pack_fail_num = 0
        max_pack_fail_num = 2 if last_bin else 1
        gen_avail_block = bh.gen_avail_block
        assign_rectangle_box_in_block = space_utils.assign_rectangle_box_in_block
        while num_residual > 0:
            abandoned_space_list = []  # han add
            trans_space_list = []  # han add
            if num_residual == last_num_residual:
                pack_fail_num += 1
                if pack_fail_num >= max_pack_fail_num:
                    break  # han add
                    # raise Exception(
                    #     "Cannot pack all stuffs in the given containers.")  # 原始语句
            else:
                pack_fail_num = 0
            last_num_residual = num_residual
            # 若只能用一个容器装载且已装载了一个容器，则装载失败返回长为2的list
            if only_one_box and len(packed_bin_list) == 1:
                packed_bin_list.append(bh.bin_obj)
                return packed_bin_list
            # 如果上一次装载的容器不为空，使用上一次装载的容器
            if last_bin:
                bh.bin_obj = copy.deepcopy(last_bin)
                filter_space(bh.bin_obj)
                last_bin = None
            # 否则初始化一个新的容器
            else:
                bh.bin_obj = PackedBin.create_by_bin(bin_obj)
                # 使用SimpleSpace方法用来进行空间选取
                bh.bin_obj.space_obj = SimpleSpace([
                    Space(bh.bin_obj.length, bh.bin_obj.width,
                          bh.bin_obj.height)
                ])
            packed_bin = bh.bin_obj
            space_obj = packed_bin.space_obj
            space_list = space_obj.space_list
            space_list.sort(key=lambda s: (s.min_coord[0], s.min_coord[1], s.min_coord[2]), reverse=False)  # han add
            # 若该容器还有可以使用的装载空间且箱子没有装载完继续装载
            while len(space_list) > 0 and num_residual > 0:
                space = space_list.pop(0)
                # block = bh.gen_avail_block(block_list, bin_obj, space, avail)
                block = gen_avail_block(block_list, bin_obj, space, avail)
                if block:
                    assign_rectangle_box_in_block(
                        space, block, packed_bin)
                    update_box_size_and_avail_map(bh.box_size_map, box_list,
                                                   avail, block)
                    num_residual -= block.box_num
                    packed_box_num += block.box_num
                    space_obj.update_space(
                        CoordinateItem(block, space.min_coord), space)
                    space_list.sort(key=lambda s: (s.min_coord[0], s.min_coord[1], s.min_coord[2]), reverse=False)  # han add
                # 若没有块能装载进当前块，进行空间转换
                # 只用当剩余空间不为空时才进行空间转换
                else:
                    # if space_list:  # 原始语句
                    #     # space_obj.transfer_space(space)  # 原来语句
                    #     space = space_obj.transfer_space(space)  # han add
                    abandoned_space_list.append(space)  # han add
            ratio_volume = packed_bin.load_volume / packed_bin.volume
            ratio_weight = packed_bin.load_weight / packed_bin.max_weight
            packed_bin.ratio = max(ratio_volume, ratio_weight)
            # packed_bin.space_obj.space_list.extend(abandoned_space_list)   # han add
            space_list.extend(abandoned_space_list)   # han add
            packed_bin_list.append(packed_bin)
        return packed_bin_list
