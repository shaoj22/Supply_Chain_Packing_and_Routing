'''
File: routing.py
Project: Supply Chain Packing and Routing
Description:
-----------
    提供了两种装载模式：
    装载模式1：二维装载     
    只考虑箱子的总体积和总总量是否超过车辆的承载能力\
    而不考虑三维装载的其它约束\
    实际上整个问题处理的是capacitated split delivery VRP\
    Routing.evaluate_sol_set()中应调用truck_list = self.get_a_decodedsol(sol_list)\
    装载模式2：三维装载
    考虑所有的三维装载约束，这才是真正的3L-SDVRP问题\
    实际上没有用到Order类中的装载模式1C-FLP(full load pattern)和1C-SP(split pattern)\
    Routing.evaluate_sol_set()中应调用truck_list = pack_obj.decode_sol_SDVRP(sol_list)\
-----------
Author: 626
Created Date: 2023-1009
'''


import sys
sys.path.append('..')
import copy
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from path_routing_algorithm import operator
from path_routing_algorithm import fitness_evaluate
from path_routing_algorithm import generate_init_solution
from path_routing_algorithm import utils


class Routing:
    def __init__(self, order, pack_obj=None, which_pack_obj=1, useFLP=None) -> None:
        """
            :param order: Order类对象
            :param pack_obj: Pack类对象
            :param which_pack_obj: 1:huawei packing方法; 2:自己的packing方法
            :param useFLP: True: use Full Load Pattern;
                           False: not use FLP, and platforms' SP(segment pattern) will include platforms' all boxes;
                           Setting useFLP makes sense only when pack_obj=None.
        """
        super().__init__()
        self.evaluated_num = 0 # 评估过解的数量
        self.truck_listdict = [] # 车辆列表字典
        self.order = order # 订单信息类
        self.platform_num = self.order.platform_num # 取货点数量
        self.allplatform_listdict = self.order.allplatform_listdict # 所有取货点信息
        if pack_obj:
            self.bin = pack_obj.bin 
        else:
            self.bin = self.order.bin # 体积最大的
        self.distance_2dMatrix = self.order.distanceMap_2dMatrix # 距离矩阵
        self.pack_patterns_dictlistdictlistAlgorithmBox =\
            self.order.pack_pattern_each_platform_dictlistdictlistAlgorithmBox
        self.res = {"estimateCode": self.order.data["estimateCode"], "solutionArray": []}  # 存储整个order的路径解决方案
        self.max_split_costs = float('inf')
        self.evaluated_sol_listlist = [] # 存储已经被评价过的解
        self.localsearched_sol_listlist = [] # 存储已经被局部搜索过的解
        self.niter = 1 # 算法中的niter, 每个partial search中迭代的次数
        self.history_indvd_listlist = [] # 可用于存储出现过的个体
        self.file_name = self.order.data["estimateCode"] # 最终保存的解的文件与输入文件名相同
        self.experiments_results = {} # 存储单次实验结果
        self.experiments_results["evaluation_num"] = []
        self.original_list = list(range(1, self.platform_num + 1))
        if pack_obj:
            self.pack_obj = pack_obj
        else:
            self.pack_obj = None
        self.sol_2_trucklist_Dict = {}
        self.which_pack_obj = which_pack_obj
        self.re_pack = False
        self.useFLP = useFLP

    def local_search(self, sol, eval_num=None):
        """ 对某个解进行局部搜索 """
        print("Algorithm step 3 local search start...")
        s_best = sol  # [1,2,3,4,5,6,7,8,9]
        nbh_size = 0
        n = self.platform_num
        for ips in range(1): # range(self.nps):
            self.max_split_costs = float('inf') # self.Max_split_costs_list[ips]
            for inbh in range(1): # range(2)
                if inbh == 0:
                    nbh_size = int(n/2)  # max(n/4, 3)
                else:
                    nbh_size = max(n, 3)
                # 进行partial_search并且获取解和适应度值
                nondominated_sol_listlist, nondominated_sol_f1f2_listlist =\
                  self.partial_search(s_best, nbh_size, eval_num)
        print("Algorithm step 5 local search end...")
        print("-"*100)
        return nondominated_sol_listlist, nondominated_sol_f1f2_listlist

    def partial_search(self, s_best, nbh_size, eval_num=None):
        """ 对某个解进行局部搜索 """      
        s_curr = s_best
        nbh_sample = [] # 解的样本
        for iter in range(self.niter):
            print("Algorithm step 3.1 local search's swap operator is working...")
            nbh_aft_swap_listlist = operator.swap_operator(s_curr, nbh_size) # 对解进行交换算子搜索
            print("Algorithm step 3.2 local search's 2opt operator is working...")
            nbh_aft_2opt_listlist = operator.get_nbh_sol_set_by_2opt(s_curr) # 对解进行2opt算子搜索
            print("Algorithm step 3.3 local search's 3opt operator is working...")
            nbh_aft_3opt_listlist = operator.get_nbh_sol_set_by_3opt(s_curr) # 对解进行3opt算子搜索
            nbh = nbh_aft_swap_listlist + nbh_aft_2opt_listlist + nbh_aft_3opt_listlist # 汇总解
            if eval_num is not None and len(nbh) > int(eval_num):
                nbh_sample_listtuple = random.sample(nbh, int(eval_num))  # 控制评价次数
                for nb_tuple in nbh_sample_listtuple:  
                    nbh_sample.append(list(nb_tuple)) # 添加新解到样本
            else:
                nbh_sample = nbh # 添加新解到样本
            if len(nbh_sample) == 0:
                raise Exception("nbh_sample is empty.")
            print("Algorithm step 3.4 search finished and get the sample:")
            nondominated_solset_listlist, nondominated_solset_f1f2_listlist =\
              self.get_nondominated_sol_set(nbh_sample, nbh=nbh) # 获取非支配解的集合和它们的适应度值
        return nondominated_solset_listlist, nondominated_solset_f1f2_listlist

    def individual_global_search(self, init_sol, eval_num):
        """ 不严格控制评价次数,即允许最后一代结束后总评价次数大于eval_num """
        t1 = time.process_time()
        # 用于存储搜索过程中的所有非支配解集, 以便选择hv值最大的进行输出
        # [i]表示索引为i的非支配解集,是二维list; [i][j]表示对应的一个解，是一个一维list
        paretoSolSet_3dlist = []
        # 用于绘图.每个元素是一个list,存储每一代的pareto最优解集两个目标函数值,
        # pareto集中的每个元素也是一个list,存储f1f2值
        paretoSolSet_f1f2_3dlist = []
        terminate_indication = 0
        hv_list = []
        eval_num_list = []
        many_nondomi_solset_listlist = []
        many_nondomi_solset_f1f2_listlist = []
        while init_sol in self.localsearched_sol_listlist:
            # 重新采样一个初始解
            # init_sol_listtuple = random.sample(self.full_pop_listtuple, 1)
            # init_sol = list(init_sol_listtuple[0])
            indi_list = generate_init_solution.sample_pop(1, self.original_list, self.platform_num)
            init_sol = indi_list[0]
        # 对初始解执行local_search(),得到一个非支配解集
        nondomi_solset_listlist, nondomi_solset_f1f2_listlist =\
          self.local_search(init_sol)
        self.localsearched_sol_listlist.append(init_sol)
        hv = utils.get_HV(nondomi_solset_listlist, nondomi_solset_f1f2_listlist)
        hv_list.append(hv)
        eval_num_list.append(self.evaluated_num)
      # Strt:local_search()中不对输入的解进行评价，所以需要单独评价一下初始解,并加入集合
      # 因为evaluate_sol_set()只接受二维list参数，所以要对单个的init_sol做转换
        init_sol_listlist, init_f1f2_listlist = self.evaluate_sol_set([init_sol])
        if len(init_sol_listlist) > 0:  # init_sol没被评价过
            nondomi_solset_listlist.append(init_sol_listlist[0])
            nondomi_solset_f1f2_listlist.append(init_f1f2_listlist[0])
            paretoSolSet_f1f2_3dlist.append(nondomi_solset_f1f2_listlist[:-1])
            paretoSolSet_3dlist.append(nondomi_solset_listlist[:-1])  # 用于最后输出
            self.evaluated_num += 1
        else:
            paretoSolSet_f1f2_3dlist.append(nondomi_solset_f1f2_listlist)
            paretoSolSet_3dlist.append(nondomi_solset_listlist)  # 用于最后输出
        many_nondomi_solset_listlist.extend(nondomi_solset_listlist)
        many_nondomi_solset_f1f2_listlist.extend(nondomi_solset_f1f2_listlist)
        if len(nondomi_solset_listlist) == 1:
            raise Exception("wrong.")
      # End
        while self.evaluated_num < eval_num and terminate_indication < 3:
            # 对非支配解集中的每个解进行local_search(),并将所有非支配解集合并
            for sol_list in nondomi_solset_listlist:
                # 只对没有进行过局部搜索的解进行局部搜索
                if sol_list not in self.localsearched_sol_listlist:
                    nondomi_sol_subset, nondomi_sol_f1f2_subset =\
                        self.local_search(sol_list)
                    # 将该解加入到局部搜索过的个体记录中
                    self.localsearched_sol_listlist.append(sol_list)
                    many_nondomi_solset_listlist.extend(nondomi_sol_subset)
                    many_nondomi_solset_f1f2_listlist.extend(nondomi_sol_f1f2_subset)
            # 从合并后的解集中再找出一个非支配解集
            nondomi_solset_listlist, nondomi_solset_f1f2_listlist =\
                self.get_nondominated_sol_set(
                    many_nondomi_solset_listlist, many_nondomi_solset_f1f2_listlist)
            many_nondomi_solset_listlist = copy.deepcopy(nondomi_solset_listlist)
            many_nondomi_solset_f1f2_listlist = copy.deepcopy(nondomi_solset_f1f2_listlist)
            hv = utils.get_HV(nondomi_solset_listlist, nondomi_solset_f1f2_listlist)
            if len(hv_list) > 0 and hv == hv_list[-1]:
                terminate_indication += 1
            eval_num_list.append(self.evaluated_num)
            hv_list.append(hv)  # 可删去
            if len(nondomi_solset_listlist) != 0:
                paretoSolSet_3dlist.append(nondomi_solset_listlist)  # 用于最后输出
                paretoSolSet_f1f2_3dlist.append(nondomi_solset_f1f2_listlist)  # 用于绘图
            else:
                raise Exception("nondomi_solset_listlist is empty.")
        new_hv_list = utils.calculate_HVs(paretoSolSet_3dlist, paretoSolSet_f1f2_3dlist)
        best_hv = max(new_hv_list)
        best_nondomi_set_index = new_hv_list.index(best_hv)
        best_nondomi_set_listlist = paretoSolSet_3dlist[best_nondomi_set_index]
        best_nondomi_set_f1f2_listlist = paretoSolSet_f1f2_3dlist[best_nondomi_set_index]
        t2 = time.process_time()
        print("Evaluation Number: ", eval_num_list, flush=True)
        print("HV: ", hv_list, flush=True)
        print("New HV: ", new_hv_list, flush=True)
        # self.get_POF(paretoSolSet_f1f2_3dlist, new_hv_list, eval_num_list)
        # 存储实验结果
        self.experiments_results["best_hv"] = best_hv
        self.experiments_results["best_nondomi_solset"] = best_nondomi_set_listlist
        self.experiments_results["hv_list"] = new_hv_list
        self.experiments_results["paretoSolSet_f1f2_3dlist"] = paretoSolSet_f1f2_3dlist
        self.experiments_results["evaluation_num"].append(eval_num_list)
        self.experiments_results["cpu_time"] = t2 - t1
        return best_nondomi_set_listlist, best_nondomi_set_f1f2_listlist

    def individual_global_search_limit_by_eval_num(self, init_sol, eval_num, gen_num):
        """
        通过把评价次数分配给每一次迭代，严格控制评价次数,不允许最后一代结束后总评价次数大于eval_num;
        每一代结束后的累计评价次数self.evaluated_num不要超过截至该代每代所分配的评价次数之和.
        """
        remaining_eval_num = 0
        eval_num_this_gen = 0
        # 用于存储搜索过程中的所有非支配解集, 以便选择hv值最大的进行输出
        # [i]表示索引为i的非支配解集,是二维list; [i][j]表示对应的一个解，是一个一维list
        paretoSolSet_3dlist = []
        # 用于绘图.每个元素是一个list,存储每一代的pareto最优解集,
        # pareto集中的每个元素也是一个list,存储f1f2值
        paretoSolSet_f1f2_3dlist = []
        terminate_indication = 0
        hv_list = []
        eval_num_list = []
        many_nondomi_solset_listlist = []
        many_nondomi_solset_f1f2_listlist = []
        # 如果输入的解已经被局部搜索过则需要重新采样一个解
        while init_sol in self.localsearched_sol_listlist:
            # 重新采样一个初始解
            indi_list = generate_init_solution.sample_pop(1, self.original_list, self.platform_num)
            init_sol = indi_list[0]
        # 第0代分配的评价次数
        eval_num_init_gen = int(eval_num/(gen_num+1))
        # 对初始解执行local_search(),得到一个非支配解集
        print("init the nondominate solution")
        nondomi_solset_listlist, nondomi_solset_f1f2_listlist =\
          self.local_search(init_sol, eval_num_init_gen)
        self.localsearched_sol_listlist.append(init_sol)
        # hv = self.get_HV(nondomi_solset_listlist, nondomi_solset_f1f2_listlist)
        # hv_list.append(hv)
        eval_num_list.append(self.evaluated_num)
      # Strt:local_search()中不对输入的解进行评价，所以需要单独评价一下初始解,并加入集合
      # 因为evaluate_sol_set()只接受二维list参数，所以要对单个的init_sol做转换
        init_sol_listlist, init_f1f2_listlist = self.evaluate_sol_set([init_sol])
        if len(init_sol_listlist) > 0:  # init_sol没被评价过
            nondomi_solset_listlist.append(init_sol_listlist[0])
            nondomi_solset_f1f2_listlist.append(init_f1f2_listlist[0])
            paretoSolSet_f1f2_3dlist.append(nondomi_solset_f1f2_listlist[:-1])
            paretoSolSet_3dlist.append(nondomi_solset_listlist[:-1])  # 用于最后输出
            self.evaluated_num += 1
        else:
            paretoSolSet_f1f2_3dlist.append(nondomi_solset_f1f2_listlist)
            paretoSolSet_3dlist.append(nondomi_solset_listlist)  # 用于最后输出
        many_nondomi_solset_listlist.extend(nondomi_solset_listlist)
        many_nondomi_solset_f1f2_listlist.extend(nondomi_solset_f1f2_listlist)
        remaining_eval_num = eval_num - self.evaluated_num  # 更新评价次数
      # End
        while self.evaluated_num < eval_num and terminate_indication < 3:
            print("local search once", "-"*50, self.evaluated_num, "times // ", eval_num)
            print("-"*100)
            # 确定该代的最大评价次数
            if gen_num > 0:
                eval_num_this_gen = remaining_eval_num / gen_num
            else:
                eval_num_this_gen = remaining_eval_num
            indi_num = len(nondomi_solset_listlist)
            # 对非支配解集中的每个解进行local_search(),并将所有非支配解集合并
            for sol_list in nondomi_solset_listlist:
                eval_num_this_indi_nbh =\
                    eval_num_this_gen / indi_num  # 该个体的邻域所分配到的最大评价次数
                # 只对没有进行过局部搜索的解进行局部搜索
                if eval_num_this_indi_nbh > 1 and (
                   sol_list not in self.localsearched_sol_listlist):
                    nondomi_sol_subset, nondomi_sol_f1f2_subset =\
                        self.local_search(sol_list, eval_num_this_indi_nbh)
                    # 将该解加入到局部搜索过的个体记录中
                    self.localsearched_sol_listlist.append(sol_list)
                    many_nondomi_solset_listlist.extend(nondomi_sol_subset)
                    many_nondomi_solset_f1f2_listlist.extend(nondomi_sol_f1f2_subset)
            # 更新剩余的评价次数
            remaining_eval_num = eval_num - self.evaluated_num
            gen_num -= 1
            # 从合并后的解集中再找出一个非支配解集
            nondomi_solset_listlist, nondomi_solset_f1f2_listlist =\
                self.get_nondominated_sol_set(
                    many_nondomi_solset_listlist, many_nondomi_solset_f1f2_listlist)
            many_nondomi_solset_listlist = copy.deepcopy(nondomi_solset_listlist)
            many_nondomi_solset_f1f2_listlist = copy.deepcopy(nondomi_solset_f1f2_listlist)
            # hv = self.get_HV(nondomi_solset_listlist, nondomi_solset_f1f2_listlist)
            # if len(hv_list) > 0 and hv == hv_list[-1]:
            if self.evaluated_num == eval_num_list[-1]:
                terminate_indication += 1
            eval_num_list.append(self.evaluated_num)
            # print("Evaluation Number: ", eval_num_list, flush=True)
            # hv_list.append(hv)
            if len(nondomi_solset_listlist) != 0:
                paretoSolSet_3dlist.append(nondomi_solset_listlist)  # 用于最后输出
                paretoSolSet_f1f2_3dlist.append(nondomi_solset_f1f2_listlist)  # 用于绘图
            else:
                raise Exception("nondomi_solset_listlist is empty.")
        new_hv_list = utils.calculate_HVs(paretoSolSet_3dlist, paretoSolSet_f1f2_3dlist)
        best_hv = max(new_hv_list)
        best_nondomi_set_index = new_hv_list.index(best_hv)
        best_nondomi_set_listlist = paretoSolSet_3dlist[best_nondomi_set_index]
        best_nondomi_set_f1f2_listlist = paretoSolSet_f1f2_3dlist[best_nondomi_set_index]
        # print("Evaluation Number: ", eval_num_list, flush=True)
        # print("HV: ", hv_list, flush=True)
        # print("New HV: ", new_hv_list, flush=True)
        # self.get_POF(paretoSolSet_f1f2_3dlist, new_hv_list, eval_num_list)
        # 存储实验结果
        # self.experiments_results["best_hv"] = best_hv
        # self.experiments_results["best_nondomi_solset"] = best_nondomi_set_listlist
        # self.experiments_results["hv_list"] = new_hv_list
        # self.experiments_results["paretoSolSet_f1f2_3dlist"] = paretoSolSet_f1f2_3dlist
        # self.experiments_results["evaluation_num"].append(eval_num_list)
        # self.experiments_results["cpu_time"] = t2 - t1
        return best_nondomi_set_listlist, best_nondomi_set_f1f2_listlist, best_hv

    def population_global_search(self, eval_num):
        """
        与两个individual_global_search相比，
        该搜索方法的第一步不是从单个解的邻域找pareto optimal set并往下搜索，
        而是从解空间中采样多个解，从邻域中找pareto optimal set,
        将多个帕雷托最优解集合并，找出新的帕雷托最优解集(实际上相当于将采样的多个解的邻域合并，
        从中找出一个帕雷托最优解集)，再对解集中的每个解往下搜索。
        """
        gen_num = 3
        t1 = time.process_time()
        # 用于存储搜索过程中的所有非支配解集, 以便选择hv值最大的进行输出
        # [i]表示索引为i的非支配解集,是二维list; [i][j]表示对应的一个解，是一个一维list
        paretoSolSet_3dlist = []
        # 用于绘图.每个元素是一个list,存储每一代的pareto最优解集,
        # pareto集中的每个元素也是一个list,存储f1f2值
        paretoSolSet_f1f2_3dlist = []
        terminate_indication = 0
        hv_list = []
        eval_num_list = []
        many_nondomi_solset_listlist = []
        many_nondomi_solset_f1f2_listlist = []
        # 得到初始种群
        # full_pop_listtuple = self.full_pop_listtuple
        # if len(full_pop_listtuple) > 10:
        #     init_pop_listtuple = random.sample(full_pop_listtuple, 10)
        # else:
        #     init_pop_listtuple = full_pop_listtuple
        init_pop_listtuple = generate_init_solution.ample_pop(10, self.original_list, self.platform_num)
        # 第0代分配的评价次数
        eval_num_init_gen = int(eval_num/(gen_num+1))
        # 第0代每个个体分配的评价次数
        eval_num_evey_init_indi = int(eval_num_init_gen/len(init_pop_listtuple))
        # 对初始种群中的每个个体,进行local_search(),得到非支配解集
        for indi_sol_tuple in init_pop_listtuple:
            indi_sol = list(indi_sol_tuple)
            nondomi_solset_listlist, nondomi_solset_f1f2_listlist =\
              self.local_search(indi_sol, eval_num_evey_init_indi)
            self.localsearched_sol_listlist.append(indi_sol)
            indi_sol_listlist, indi_f1f2_listlist = self.evaluate_sol_set([indi_sol])
            if len(indi_sol_listlist) > 0:
                self.evaluated_num += 1
                nondomi_solset_listlist.append(indi_sol_listlist[0])
                nondomi_solset_f1f2_listlist.append(indi_f1f2_listlist[0])
                many_nondomi_solset_listlist.extend(nondomi_solset_listlist[:-1])
                many_nondomi_solset_f1f2_listlist.extend(nondomi_solset_f1f2_listlist[:-1])
            else:  # indi_sol已经被评价过
                many_nondomi_solset_listlist.extend(nondomi_solset_listlist)
                many_nondomi_solset_f1f2_listlist.extend(nondomi_solset_f1f2_listlist)
        # 对合并后的解集，从中再找出一个非支配解集
        nondomi_solset_listlist, nondomi_solset_f1f2_listlist =\
            self.get_nondominated_sol_set(
                many_nondomi_solset_listlist, many_nondomi_solset_f1f2_listlist)
        many_nondomi_solset_listlist = copy.deepcopy(nondomi_solset_listlist)
        many_nondomi_solset_f1f2_listlist = copy.deepcopy(nondomi_solset_f1f2_listlist)
        hv = utils.get_HV(nondomi_solset_listlist, nondomi_solset_f1f2_listlist)
        hv_list.append(hv)
        eval_num_list.append(self.evaluated_num)
        paretoSolSet_3dlist.append(nondomi_solset_listlist)  # 用于最后输出
        paretoSolSet_f1f2_3dlist.append(nondomi_solset_f1f2_listlist)  # 用于绘图
        remaining_eval_num = eval_num - self.evaluated_num  # 更新评价次数
        # 对非支配解集中的解再进行局部搜索
        while self.evaluated_num < eval_num and terminate_indication < 3:
            # 确定该代的最大评价次数
            if gen_num > 0:
                eval_num_this_gen = remaining_eval_num / gen_num
            else:
                eval_num_this_gen = remaining_eval_num
            indi_num = len(nondomi_solset_listlist)
            # 对非支配解集中的每个解进行local_search(),并将所有非支配解集合并
            for sol_list in nondomi_solset_listlist:
                eval_num_this_indi_nbh =\
                    eval_num_this_gen / indi_num  # 该个体的邻域所分配到的最大评价次数
                # 只对没有进行过局部搜索的解进行局部搜索
                if eval_num_this_indi_nbh > 1 and (
                   sol_list not in self.localsearched_sol_listlist):
                    nondomi_sol_subset, nondomi_sol_f1f2_subset =\
                        self.local_search(sol_list, eval_num_this_indi_nbh)
                    # 将该解加入到局部搜索过的个体记录中
                    self.localsearched_sol_listlist.append(sol_list)
                    many_nondomi_solset_listlist.extend(nondomi_sol_subset)
                    many_nondomi_solset_f1f2_listlist.extend(nondomi_sol_f1f2_subset)
            # 更新剩余的评价次数
            remaining_eval_num = eval_num - self.evaluated_num
            gen_num -= 1
            # 从合并后的解集中再找出一个非支配解集
            nondomi_solset_listlist, nondomi_solset_f1f2_listlist =\
                self.get_nondominated_sol_set(
                    many_nondomi_solset_listlist, many_nondomi_solset_f1f2_listlist)

            """ # 从many_nondomi_solset_listlist中采样10个新个体加入到nondomi_solset_listlist
            disturbance_indis_listtuple = random.sample(many_nondomi_solset_listlist, 10) """

            many_nondomi_solset_listlist = copy.deepcopy(nondomi_solset_listlist)
            many_nondomi_solset_f1f2_listlist = copy.deepcopy(nondomi_solset_f1f2_listlist)
            hv = utils.get_HV(nondomi_solset_listlist, nondomi_solset_f1f2_listlist)
            if len(hv_list) > 0 and hv == hv_list[-1]:
                terminate_indication += 1
            eval_num_list.append(self.evaluated_num)
            hv_list.append(hv)
            if len(nondomi_solset_listlist) != 0:
                paretoSolSet_3dlist.append(nondomi_solset_listlist)  # 用于最后输出
                paretoSolSet_f1f2_3dlist.append(nondomi_solset_f1f2_listlist)  # 用于绘图
            else:
                raise Exception("nondomi_solset_listlist is empty.")

            """ # 从many_nondomi_solset_listlist中采样10个新个体加入到nondomi_solset_listlist
            for distb_indi_tuple in disturbance_indis_listtuple:
                distb_indi_list = list(distb_indi_tuple)
                if distb_indi_list not in nondomi_solset_listlist:
                    nondomi_solset_listlist.append(distb_indi_list) """

        new_hv_list = utils.calculate_HVs(paretoSolSet_3dlist, paretoSolSet_f1f2_3dlist)
        best_hv = max(new_hv_list)
        best_nondomi_set_index = new_hv_list.index(best_hv)
        best_nondomi_set_listlist = paretoSolSet_3dlist[best_nondomi_set_index]
        best_nondomi_set_f1f2_listlist = paretoSolSet_f1f2_3dlist[best_nondomi_set_index]
        t2 = time.process_time()
        print("Evaluation Number: ", eval_num_list, flush=True)
        print("HV: ", hv_list, flush=True)
        print("New HV: ", new_hv_list, flush=True)
        print("Best hv:", best_hv, flush=True)
        # self.get_POF(paretoSolSet_f1f2_3dlist, new_hv_list, eval_num_list)
        # 存储实验结果
        self.experiments_results["best_hv"] = best_hv
        self.experiments_results["best_nondomi_solset"] = best_nondomi_set_listlist
        self.experiments_results["hv_list"] = new_hv_list
        self.experiments_results["paretoSolSet_f1f2_3dlist"] = paretoSolSet_f1f2_3dlist
        self.experiments_results["evaluation_num"].append(eval_num_list)
        self.experiments_results["cpu_time"] = t2 - t1
        return best_nondomi_set_listlist, best_nondomi_set_f1f2_listlist

    def population_global_search2(self, eval_num, sample_num):
        """ 进行种群全局搜索解 """
        gen_num = 3
        # 用于绘图：每个元素是一个list,存储每一代的pareto最优解集, pareto集中的每个元素也是一个list,存储f1f2值
        paretoSolSet_f1f2_3dlist = []
        paretoSolSet_3dlist = []
        many_nondomi_solset_listlist = []
        many_nondomi_solset_f1f2_listlist = []
        # 得到初始种群
        init_pop_listtuple = generate_init_solution.sample_pop(sample_num, self.original_list, self.platform_num)
        print("Algorithm step 1 get init population:", init_pop_listtuple)
        # 对初始种群中的每个个体,进行local_search(),得到非支配解集
        indi_index = 0
        # global figure_name
        # init_figure_name = copy.deepcopy(figure_name)
        print("Algorithm step 2 use individual global search the init population...")
        for i in tqdm(range(len(init_pop_listtuple))):
            # figure_name = init_figure_name + 'indi' + str(indi_index) + '.png'
            indi_index += 1
            # print('--------The', indi_index, 'th individual', '---------------------', flush=True)
            self.evaluated_num = 0
            indi_sol = list(init_pop_listtuple[i])
            #nondomi_sol, nondomi_f1f2 = self.individual_global_search(indi_sol, eval_num/10)
            nondomi_sol, nondomi_f1f2, hv =\
                self.individual_global_search_limit_by_eval_num(indi_sol, eval_num/sample_num, gen_num=gen_num)
            if (nondomi_sol) != 0:
                paretoSolSet_3dlist.append(nondomi_sol)
                paretoSolSet_f1f2_3dlist.append(nondomi_f1f2)
            else:
                raise Exception("nondomi_sol is empty.")
            # hv_list.append(hv)
            many_nondomi_solset_listlist.extend(nondomi_sol)
            many_nondomi_solset_f1f2_listlist.extend(nondomi_f1f2)
        # 对合并解集再找出一个非支配解集
        nondomi_solset_listlist, nondomi_solset_f1f2_listlist =\
            self.get_nondominated_sol_set(
                many_nondomi_solset_listlist, many_nondomi_solset_f1f2_listlist)
        print("All things done")
        print("Algorithm step 6 get global search solution:", nondomi_solset_listlist)
        # hv = self.get_HV(nondomi_solset_listlist, nondomi_solset_f1f2_listlist)
        # hv_list.append(hv)
        if len(nondomi_solset_listlist) != 0:
            paretoSolSet_3dlist.append(nondomi_solset_listlist)  # 用于最后输出
            paretoSolSet_f1f2_3dlist.append(nondomi_solset_f1f2_listlist)  # 用于绘图
        else:
            raise Exception("nondomi_solset_listlist is empty.")
        new_hv_list = utils.calculate_HVs(paretoSolSet_3dlist, paretoSolSet_f1f2_3dlist)
        best_hv = max(new_hv_list)
        best_nondomi_set_index = new_hv_list.index(best_hv)
        best_nondomi_set_listlist = paretoSolSet_3dlist[best_nondomi_set_index]
        # t2 = time.process_time()
        # print("Best nondominated set:", best_nondomi_set_listlist, flush=True)
        # print("hv:", hv_list, flush=True)
        # print("New hv:", new_hv_list, flush=True)
        # print("Best hv:", best_hv, flush=True)
        # print("end.", flush=True)
        # 存储实验结果
        # self.experiments_results["best_hv"] = best_hv
        # self.experiments_results["best_nondomi_solset"] = best_nondomi_set_listlist
        # self.experiments_results["hv_list"] = new_hv_list
        # self.experiments_results["paretoSolSet_f1f2_3dlist"] = paretoSolSet_f1f2_3dlist
        # self.experiments_results["cpu_time"] = t2 - t1
        # self.get_POF(paretoSolSet_f1f2_3dlist)
        # self.get_POF([many_nondomi_solset_f1f2_listlist])
        return best_nondomi_set_listlist

    def get_nondominated_sol_set(self, sol_listlist, sol_f1f2_listlist=None, nbh=None):
        """ 从样本中获取非支配解的集合和它们的适应度值 """
        print("Algorithm step 4 get nondominated solution set start...")
        nondominated_solset_listlist = [] # 非支配解的集合
        nondominated_solset_f1f2_listlist = [] # 它们的适应度值
        if sol_f1f2_listlist is None: # 若没有获得适应度值则计算评估获得
            new_sol_listlist, f1f2_listlist = self.evaluate_sol_set(sol_listlist, nbh=nbh)
            sol_num = len(new_sol_listlist)
        else:
            f1f2_listlist = sol_f1f2_listlist
            sol_num = len(f1f2_listlist)
            new_sol_listlist = sol_listlist

        # nondominated_indication[i][j]=1表示sol_listlist[i] is dominated by sol_listlist[j];
        # nondominated_indication[i][j]=0表示sol_listlist[i] isn't dominated by sol_listlist[j];
        # 通过适应度值来获得支配和非支配的解
        nondominated_indication = np.zeros((sol_num, sol_num))
        for i in range(sol_num-1):
            f1_i = f1f2_listlist[i][0]
            f2_i = f1f2_listlist[i][1]
            for j in range(sol_num):
                f1_j = f1f2_listlist[j][0]
                f2_j = f1f2_listlist[j][1]
                if f1_j <= f1_i and f2_j <= f2_i and (f1_j < f1_i or f2_j < f2_i):
                    # i is dominated by j
                    nondominated_indication[i][j] = 1
                elif f1_i <= f1_j and f2_i <= f2_j and (f1_i < f1_j or f2_i < f2_j):
                    # j is dominated by i
                    nondominated_indication[j][i] = 1
        nondominated_indication_sum_by_line = np.sum(nondominated_indication, axis=1)
        for i in range(sol_num):
            if nondominated_indication_sum_by_line[i] == 0:
                nondominated_solset_listlist.append(new_sol_listlist[i])
                nondominated_solset_f1f2_listlist.append(f1f2_listlist[i])
        if len(nondominated_solset_listlist) == 0:
            raise Exception("nondominated_solset_listlist is empty.")
        print("Algorithm step 4.1 the nondominated solution set is:", nondominated_solset_listlist)
        print("Algorithm step 4 get nondominated solution set end...")
        return nondominated_solset_listlist, nondominated_solset_f1f2_listlist

    def evaluate_sol_set(self, sol_listlist, nbh=None):
        """ 输入解的集合并它们的适应度值 """
        new_sol_listlist = []
        f1f2_listlist = []  # 每个元素是一个list,存储一个sol的f1和f2的值
        for sol_list in sol_listlist:
            # 只对未评价过的个体进行评价;self.re_pack:用自己的packing方法进行重新装载时,都是已经评价过的个体
            if sol_list not in self.evaluated_sol_listlist or self.re_pack:
                if self.pack_obj:
                    if self.which_pack_obj == 1:
                        # 先从sol_list得到以bonded warehouse开头的子路径
                        sub_sol_listlist = fitness_evaluate.get_sub_sol_according2_mustfirst_points(sol_list, self.allplatform_listdict)
                        # sub_sol_listlist中存的是索引，转化为以platformCode表示的list
                        route_list = fitness_evaluate.sols_2_routes(sub_sol_listlist, self.allplatform_listdict)
                        # 将子路径的list传给pack_obj
                        self.pack_obj.routes = route_list
                        res, truck_listdict = self.pack_obj.run()
                        self.sol_2_trucklist_Dict[''.join([str(x) for x in sol_list])] = truck_listdict
                    elif self.which_pack_obj == 2:
                        truck_listdict = self.pack_obj.decode_sol_SDVRP(sol_list)
                        self.sol_2_trucklist_Dict[''.join([str(x) for x in sol_list])] = truck_listdict
                        self.pack_obj.bin = self.pack_obj.origin_bin
                else:
                    truck_listdict = fitness_evaluate.get_a_decodedsol(sol_list,\
                                                            self.platform_num, self.allplatform_listdict,\
                                                            self.pack_patterns_dictlistdictlistAlgorithmBox,\
                                                            self.bin, self.order, self.max_split_costs, self.distance_2dMatrix,\
                                                            useFLP=self.useFLP)
                f1, f2 = fitness_evaluate.get_f1f2_values_one_sol(truck_listdict, self.allplatform_listdict, self.distance_2dMatrix)
                f1f2_listlist.append([f1, f2])
                new_sol_listlist.append(sol_list)
                self.evaluated_sol_listlist.append(sol_list)  # 更新list
            elif nbh is not None:  # 如果个体被评价过了，那么从邻域中再采样来补充
                sample_num = 0
                sample_sol_list = list(random.sample(nbh, 1)[0])
                while(sample_sol_list in self.evaluated_sol_listlist and sample_num < 100):
                    sample_sol_list = list(random.sample(nbh, 1)[0])
                    sample_num += 1
                if sample_sol_list not in self.evaluated_sol_listlist:
                    if self.pack_obj:
                        if self.which_pack_obj == 1: 
                            # 先从sol_list得到以bonded warehouse开头的子路径
                            sub_sol_listlist = fitness_evaluate.get_sub_sol_according2_mustfirst_points(sample_sol_list, self.allplatform_listdict)
                            # sub_sol_listlist中存的是索引，转化为以platformCode表示的list
                            route_list = fitness_evaluate.sols_2_routes(sub_sol_listlist, self.allplatform_listdict)
                            # 讲子路径的list传给pack_obj
                            self.pack_obj.routes = route_list
                            res, truck_listdict = self.pack_obj.run()
                            self.sol_2_trucklist_Dict[''.join([str(x) for x in sol_list])] = truck_listdict
                        elif self.which_pack_obj == 2:
                            truck_listdict = self.pack_obj.decode_sol_SDVRP(sample_sol_list)
                            self.sol_2_trucklist_Dict[''.join([str(x) for x in sol_list])] = truck_listdict
                            self.pack_obj.bin = self.pack_obj.origin_bin
                    else:
                        truck_listdict = fitness_evaluate.get_a_decodedsol(sample_sol_list,\
                                                            self.platform_num, self.allplatform_listdict,\
                                                            self.pack_patterns_dictlistdictlistAlgorithmBox,\
                                                            self.bin, self.order, self.max_split_costs, self.distance_2dMatrix,\
                                                            useFLP=self.useFLP)
                    f1, f2 = fitness_evaluate.get_f1f2_values_one_sol(truck_listdict, self.allplatform_listdict, self.distance_2dMatrix)
                    f1f2_listlist.append([f1, f2])
                    new_sol_listlist.append(sol_list)
                    self.evaluated_sol_listlist.append(sample_sol_list)  # 更新list
        self.evaluated_num += len(new_sol_listlist)
        #print("evaluated_num: ", self.evaluated_num, flush=True)
        return new_sol_listlist, f1f2_listlist







