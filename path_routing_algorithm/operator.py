'''
File: operator.py
Project: Supply Chain Packing and Routing
Description:
-----------
three operator for the routing problem.
-----------
Author: 626
Created Date: 2023-1009
'''


import copy


def swap_operator(encoded_sol_list, nbh_size):
    """swap算子"""
    nbh_aft_swap_listlist = []
    sol_len = len(encoded_sol_list)
    for i in range(sol_len):
        encoded_sol_copy = copy.deepcopy(encoded_sol_list)
        swap_pointer = int((i + nbh_size - 1) % sol_len)
        encoded_sol_copy[i], encoded_sol_copy[swap_pointer] =\
          encoded_sol_copy[swap_pointer], encoded_sol_copy[i]
        if encoded_sol_copy not in nbh_aft_swap_listlist:
            nbh_aft_swap_listlist.append(encoded_sol_copy)
    return nbh_aft_swap_listlist

def get_nbh_sol_set_by_2opt(encoded_sol_list):
    """2-opt算子"""
    # e.g., [3,1,4,5,6,9,8,2,7] → [0,3,1,4,5,6,9,8,2,7,10]
    extend_sol = [0] + encoded_sol_list + [len(encoded_sol_list) + 1]
    edges_num = len(extend_sol) - 1
    nbh_sol_listlist = []
    for t1 in range(edges_num - 2):
        for t3 in range(t1+2, edges_num):
            nb_sol = two_opt_operator(extend_sol, t1, t3)
            nbh_sol_listlist.append(nb_sol)
    return nbh_sol_listlist

def get_nbh_sol_set_by_3opt(encoded_sol_list):
    """3-opt算子"""
    # e.g., [3,1,4,5,6,9,8,2,7] → [0,3,1,4,5,6,9,8,2,7,10]
    extend_sol = [0] + encoded_sol_list + [len(encoded_sol_list) + 1]
    edges_num = len(extend_sol) - 1
    nbh_sol_listlist = []
    for t1 in range(edges_num - 4):
        for t3 in range(t1+2, edges_num - 2):
            for t5 in range(t3+2, edges_num):
                sub_nbh_sol_list =\
                  three_opt_operator(extend_sol, t1, t3, t5)
                nbh_sol_listlist += sub_nbh_sol_list
    return nbh_sol_listlist

def two_opt_operator(encoded_sol_list, t1, t3):
    """
    该操作不会改变encoded_sol_list[0]和encoded_sol_lsit[-1]的位置
    t1,t3是索引,用于指示encoded_sol_list(下称ESL)中的2条breaking-edges;
    t2=t1+1; t4=t3+1;
    t3 > t2 = t1+1; 
    将ELS[t2:t4]反转得到一个邻域个体.
    """
    t2 = t1 + 1
    t4 = t3 + 1
    segment1 = encoded_sol_list[:t2]
    segment2 = encoded_sol_list[t2:t4]
    segment3 = encoded_sol_list[t4:]
    nb_sol = segment1 + list(reversed(segment2)) + segment3
    # 将首尾的start point和delivery point去掉
    return nb_sol[1:-1]

def three_opt_operator(encoded_sol_list, t1, t3, t5):
    """
    该操作不会改变encoded_sol_list[0]和encoded_sol_lsit[-1]的位置
    t1,t3,t5是索引,用于指示encoded_sol_list(下称ESL)中的3条breaking-edges;
    t2=t1+1; t4=t3+1; t6=t5+1;
    t3 > t2 = t1+1; t5 > t4 = t3+1;
    break 3条已存在的edges: (ESL[t1], ESL[t2]), (ESL[t3],ESL[t4]), (ESL[t5],ESL[t6]);
    再建立3条reconnecting-edges;
    3-opt语境下，共有8个cases for reconnection, 只有4个cases,
    其中所有reconnecting-dege are all new edges,我们只考虑这种情况;
    其他4种情况包含一个与原来一模一样的reconnection，以及3种实际上是2-opt的情况。
    """
    nbh_sol_listlist = []
    t2 = t1 + 1
    t4 = t3 + 1
    t6 = t5 + 1
    segment1 = encoded_sol_list[0:t2]
    segment2 = encoded_sol_list[t2:t4]
    segment3 = encoded_sol_list[t4:t6]
    segment4 = encoded_sol_list[t6:]
    nb_sol1 = segment1 + list(reversed(segment2)) + list(reversed(segment3)) + segment4
    nb_sol2 = segment1 + segment3 + list(reversed(segment2)) + segment4
    nb_sol3 = segment1 + segment3 + segment2 + segment4
    nb_sol4 = segment1 + list(reversed(segment3)) + segment2 + segment4
    # 将首尾的start point和delivery point去掉
    nbh_sol_listlist.append(nb_sol1[1:-1])
    nbh_sol_listlist.append(nb_sol2[1:-1])
    nbh_sol_listlist.append(nb_sol3[1:-1])
    nbh_sol_listlist.append(nb_sol4[1:-1])
    return nbh_sol_listlist