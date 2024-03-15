'''
File: utils.py
Project: Supply Chain Packing and Routing
Description:
-----------
some important utils for routing algorithm.
-----------
Author: 626
Created Date: 2023-1009
'''


import os
import json
import matplotlib.pyplot as plt
from path_routing_algorithm import fitness_evaluate


def get_objective_values_of_all_sols(res):
    """获得所有解的适应度值"""
    sols_listlist = res["solutionArray"]
    f1_all_sols_list = []
    f2_all_sols_list = []
    for truck_list in sols_listlist:
        f1, f2 = fitness_evaluate.get_f1f2_values_one_sol(truck_list)
        f1_all_sols_list.append(f1)
        f2_all_sols_list.append(f2)
    return f1_all_sols_list, f2_all_sols_list

def get_Max_split_costs(distance_2dMatrix, platform_num):
    """获得最大分割成本"""
    Max_split_costs = []
    ric_list = []  # ric: relative insertions costs
    dist_map = distance_2dMatrix
    platform_num = platform_num
    for i in range(1, platform_num):
        for j in range(i+1, platform_num+1):
            ric_ij = (dist_map[i][j] + dist_map[j][platform_num+1]) /\
              dist_map[i][platform_num+1]
            ric_ji = (dist_map[j][i] + dist_map[i][platform_num+1])/\
              dist_map[j][platform_num+1]
            ric_list.append(ric_ij)
            ric_list.append(ric_ji)
    ric_min = min(ric_list)
    ric_max = max(ric_list)
    msc = ric_min  # max split cost
    while msc < ric_max:
        Max_split_costs.append(msc)
        msc *= 2
    Max_split_costs.append(ric_max)
    return Max_split_costs

def save_sols(output_path, file_name, res):
    """ 将解保存到输出文件 """
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_file = os.path.join(output_path, file_name)
    with open(
            output_file, 'w', encoding='utf-8', errors='ignore') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)  # 将解保存为json文件，无后缀

def save_experiments_results(output_path, file_name, experiments_results):
    """ 将实验结果保存到输出文件 """
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_file = os.path.join(output_path, file_name + "_results.json")
    with open(
            output_file, 'w', encoding='utf-8', errors='ignore') as f:
        json.dump(experiments_results, f, ensure_ascii=False, indent=4)  # 将解保存为json文件

def get_POF(paretoSolSet_f1f2_3dlist, hv_list=None, eval_num_list=None):
    """
    绘制每一代的Pareto optimal front (POF);
    :paretoSolSet_f1f2_3dlist: 第一维[i]表示每一代；第二维[i][j]表示每一代的POF上的每一个点；
    第三维[i][j][0]/[i][j][1]表示f1/f2值.
    """
    f1_min, f1_max, f2_min, f2_max = get_minmax_f1f2(paretoSolSet_f1f2_3dlist)
    f1_gap = f1_max - f1_min
    f2_gap = f2_max - f2_min
    plt.figure(figsize=(10,5),dpi=200)  #plt.figure(figsize=(50,25))
    gen_num = len(paretoSolSet_f1f2_3dlist)
    if gen_num > 3:
        for i in range(3): # 画3条线
            nomalized_f1f2_listlist = []
            if i == 2:
                paretoSolSet_f1f2_2dlist = paretoSolSet_f1f2_3dlist[-1]
                marker = 'r*:'
                label = 'Iteration ' + str(gen_num)
            elif i == 1:
                paretoSolSet_f1f2_2dlist = paretoSolSet_f1f2_3dlist[2]
                marker = 'gs--'
                label = 'Iteration ' + str(2)
            else:
                paretoSolSet_f1f2_2dlist = paretoSolSet_f1f2_3dlist[0]
                marker = 'bo-'
                label = 'Iteration ' + str(0)
            paretoSolSet_f1f2_2dlist.sort(key=lambda f1f2_list: f1f2_list[0])
            f1_list = [f1f2_list[0] for f1f2_list in paretoSolSet_f1f2_2dlist]
            f2_list = [f1f2_list[1]/100000 for f1f2_list in paretoSolSet_f1f2_2dlist]
            plt.subplot(121)
            plt.xlabel('f1')
            plt.ylabel('f2 (e+05)')
            plt.plot(f1_list, f2_list, marker, label=label)
            plt.legend()  # 显示图例
    elif gen_num == 1:
        paretoSolSet_f1f2_2dlist = paretoSolSet_f1f2_3dlist[-1]
        marker = 'r*:'
        label = 'POF'
        paretoSolSet_f1f2_2dlist.sort(key=lambda f1f2_list: f1f2_list[0])
        f1_list = [f1f2_list[0] for f1f2_list in paretoSolSet_f1f2_2dlist]
        f2_list = [f1f2_list[1]/100000 for f1f2_list in paretoSolSet_f1f2_2dlist]
        plt.subplot(121)
        plt.plot(f1_list, f2_list, marker, label=label)
        plt.legend()  # 显示图例
    else:
        raise Exception("The number of generations is not enougth.")

        """
      # 标准化
        for f1f2_list in paretoSolSet_f1f2_2dlist:
            f1 = f1f2_list[0]
            f2 = f1f2_list[1]
            nomal_f1 = (f1 - f1_min) / f1_gap
            nomal_f2 = (f2 - f2_min) / f2_gap
            nomalized_f1f2_listlist.append([nomal_f1, nomal_f2])
      # end:标准化
        # 按f1值从小到大排序
        nomalized_f1f2_listlist.sort(key=lambda f1f2_list: f1f2_list[0])
        nomal_f1_list = [f1f2_list[0] for f1f2_list in nomalized_f1f2_listlist]
        nomal_f2_list = [f1f2_list[1] for f1f2_list in nomalized_f1f2_listlist]
        plt.subplot(122)
        plt.plot(nomal_f1_list, nomal_f2_list, markeraverage, label=label)
        """
    # 画hv曲线
    if hv_list is not None:
        plt.subplot(122)
        plt.plot(eval_num_list, hv_list, 'ro-', label='Hyper volume (HV)')
    plt.xlabel('# evaluation')
    plt.ylabel('HV')
    plt.legend()  # 显示图例
    plt.savefig(figure_name)
    # plt.show()

def get_minmax_f1f2(nondomi_sols_f1f2_3dlist):
    """ 从产生的所有非支配解集中，寻找最大和最小的f1 f2值，以便我们的对hv的计算更准确 """
    f1_list = []
    f2_list = []
    for nondomi_f1f2_2dlist in nondomi_sols_f1f2_3dlist:
        sub_f1_list = [f1f2_list[0] for f1f2_list in nondomi_f1f2_2dlist]
        sub_f2_list = [f1f2_list[1] for f1f2_list in nondomi_f1f2_2dlist]
        f1_list.extend(sub_f1_list)
        f2_list.extend(sub_f2_list)
    min_f1 = min(f1_list)
    max_f1 = max(f1_list)
    min_f2 = min(f2_list)
    max_f2 = max(f2_list)
    # minmax_f1 = [min_f1, max_f1]
    # minmax_f2 = [min_f2, max_f2]
    return min_f1, max_f1, min_f2, max_f2

def calculate_HVs(nondomi_sols_3dlist, nondomi_sols_f1f2_3dlist):
    min_f1, max_f1, min_f2, max_f2 = get_minmax_f1f2(nondomi_sols_f1f2_3dlist)
    hv_list = []
    for i in range(len(nondomi_sols_3dlist)):
        if len(nondomi_sols_3dlist[i]) == 0:
            hv_list.append(0)
            continue
        hv = get_HV(nondomi_sols_3dlist[i], nondomi_sols_f1f2_3dlist[i], min_f1=min_f1, max_f1=max_f1, min_f2=min_f2, max_f2=max_f2)
        hv_list.append(hv)
    return hv_list

def get_HV(nondominated_solset_listlist, nondominated_solset_f1f2_listlist, min_f1=None, max_f1=None, min_f2=None, max_f2=None):
    """ 计算一个Pareto optimal set的hyper volume值 """
    HV = 0
    nomalized_f1f2_listlist = []
    sol_num = len(nondominated_solset_listlist)
    if min_f1 is None:
        if sol_num == 1:
            return -1
        f1_list = [i[0] for i in nondominated_solset_f1f2_listlist]
        f2_list = [i[1] for i in nondominated_solset_f1f2_listlist]
        f1_max = max(f1_list)
        f1_min = min(f1_list)
        f2_max = max(f2_list)
        f2_min = min(f2_list)
    else:
        f1_max = max_f1
        f1_min = min_f1
        f2_max = max_f2
        f2_min = min_f2
    # begin:先将各f1和f2值标准化
    f1_gap = f1_max - f1_min
    f2_gap = f2_max - f2_min
    if f1_gap == 0 or f2_gap == 0:
        return 1.5
    for f1f2_list in nondominated_solset_f1f2_listlist:
        f1 = f1f2_list[0]
        f2 = f1f2_list[1]
        nomal_f1 = (f1 - f1_min) / f1_gap
        nomal_f2 = (f2 - f2_min) / f2_gap
        nomalized_f1f2_listlist.append([nomal_f1, nomal_f2])
  # end:标准化
    # 按f1值从小到大排序
    nomalized_f1f2_listlist.sort(key=lambda f1f2_list: f1f2_list[0])
    HV += (1.2-nomalized_f1f2_listlist[0][0]) * (1.2-nomalized_f1f2_listlist[0][1])
    for i in range(1, sol_num):
        HV += (1.2-nomalized_f1f2_listlist[i][0]) *\
          (nomalized_f1f2_listlist[i-1][1]-nomalized_f1f2_listlist[i][1])
    return HV

