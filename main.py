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

 
import os
from multiprocessing.pool import Pool
from bin_packing_algorithm.Huawei.pack import Pack
from path_routing_algorithm import utils
from path_routing_algorithm import routing
from path_routing_algorithm import fitness_evaluate
from bin_packing_algorithm import mypack
from bin_packing_algorithm.order import Order


def main(input, output):
    # 输入文件夹和输出文件夹的路径
    input_dir = input
    output_dir = output
    # 处理输入文件夹中每一个算例
    for file_name in os.listdir(input_dir):
        print("The order ", file_name, " is processing...")
        input_path = os.path.join(input_dir, file_name)
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            message_str = f.read()
        order = Order(message_str)
        mypack_obj = mypack.Pack(order)
        hwpack_obj = Pack(message_str, mypack_obj=mypack_obj)
        route = routing.Routing(order, pack_obj=hwpack_obj)
        print("Algorithm start running...")
        nondominated_sol_listlist = route.population_global_search2(20, 1)
        
        for sol_list in nondominated_sol_listlist:
            route.res["solutionArray"].append(route.sol_2_trucklist_Dict[''.join([str(x) for x in sol_list])])
            print("The code index of the platform is: ", sol_list)
            # 先从sol_list得到以bonded warehouse开头的子路径
            sub_sol_listlist = fitness_evaluate.get_sub_sol_according2_mustfirst_points(sol_list, route.allplatform_listdict)
            # sub_sol_listlist中存的是索引，转化为以platformCode表示的list
            route_list = fitness_evaluate.sols_2_routes(sub_sol_listlist, route.allplatform_listdict)
            print("The real index of the platform is: ", route_list)

        utils.save_sols(output_dir, route.file_name, route.res)               
        print("The order ", file_name, " is done.")


if __name__ == "__main__":
    INPUT_PATH="D:\\Desktop\\python_code\\Supply_Chain_Packing_and_Routing\\inputs"
    OUTPUT_PATH='D:\\Desktop\\python_code\\Supply_Chain_Packing_and_Routing\\outputs'
    main(INPUT_PATH, OUTPUT_PATH)
