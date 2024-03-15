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


import sys
sys.path.append("..")
import os
import json
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from data_tools.result_analysis import read_result_file


def visualization_bin_packing(input_dir, output_dir, result_file, visualization_dir):
    # 定义Java程序的JAR文件路径
    jar_file_path = "D:\\Desktop\\java_code\\Simulator\out\\artifacts\\Simulator_jar\\Simulator.jar"
    # 调用Java程序并传递命令行参数
    subprocess.run(["java", "-jar", jar_file_path, input_dir, output_dir, result_file, visualization_dir])

def plot_paths_with_coordinates(coordinates, result_paths, save_path=None):
    """
    绘制路径规划结果的高级可视化图，包括颜色渐变、标签优化、线条样式和宽度、点的样式和大小、箭头等

    参数：
    coordinates (dict): 一个包含点名称和坐标的字典
    result_paths (list): 一个包含多条路径的二维列表
    save_path (str): 绘制的图保存的路径

    返回：
    None
    """
    plt.figure(figsize=(10, 10))
    # 使用颜色渐变绘制每条路径
    cmap = plt.get_cmap('viridis')
    colors = iter(cmap(np.linspace(0, 1, len(result_paths))))
    for i, path in enumerate(result_paths):
        color = next(colors)
        # 计算终点的索引
        end_point_index = len(coordinates) - 1
        path_with_start_end = [0] + path + [end_point_index]
        x_coords, y_coords = zip(*[coordinates[f'platform{point:02d}'] for point in path_with_start_end])
        
        # 用渐变的颜色绘制路径
        plt.plot(x_coords, y_coords, color=color, linewidth=2, marker='o', markersize=8, label=f'Path {i + 1}')
        
        # 添加箭头
        for j in range(len(x_coords) - 1):
            # 计算线段的中点坐标
            mid_x = (x_coords[j] + x_coords[j+1]) / 2
            mid_y = (y_coords[j] + y_coords[j+1]) / 2
            
            arrow_dx = x_coords[j+1] - x_coords[j]
            arrow_dy = y_coords[j+1] - y_coords[j]
            arrow_length = np.sqrt(arrow_dx**2 + arrow_dy**2)
            
            # 计算箭头方向
            arrow_dx /= arrow_length
            arrow_dy /= arrow_length
            
            # 添加箭头
            plt.arrow(mid_x, mid_y, arrow_dx * 200, arrow_dy * 200, color='black', 
                      head_width=100, head_length=100)
            
    # 绘制点的坐标和标签
    for platform, coord in coordinates.items():
        plt.scatter(coord[0], coord[1], color='black', s=80)
        plt.text(coord[0] + 50, coord[1] + 50, platform, fontsize=10, color='black', alpha=0.7)
        
    # 添加起点和终点
    plt.scatter(0, 0, color='green', s=150, label='Start (0, 0)')
    if end_point_index < 10:
        end_point_name = 'platform0' + str(end_point_index)
    else:
        end_point_name = 'platform' + str(end_point_index)
    plt.scatter(coordinates[end_point_name][0], coordinates[end_point_name][1], color='red', s=150, label=f'End ({coordinates[end_point_name][0]}, {coordinates[end_point_name][1]})')

    # 设置坐标轴范围
    plt.xlim(-500, 10500)
    plt.ylim(-500, 10500)

    # 设置坐标轴标签和标题
    plt.xlabel('X Coordinate', fontsize=14)
    plt.ylabel('Y Coordinate', fontsize=14)
    plt.title('Path Planning Visualization', fontsize=16)

    # 显示图例
    plt.legend(fontsize=12)

    # 设置背景颜色
    plt.gca().set_facecolor('#F7F7F7')

    # 隐藏坐标轴上的刻度
    plt.tick_params(axis='both', which='both', bottom=False, left=False)

    # 显示网格
    plt.grid(True, linestyle='--', alpha=0.6)
    # 保存结果
    if save_path:
        plt.savefig(save_path)
    # 显示图表
    # plt.show()
    plt.close()

def visualization_path_routing(input_dir, output_dir):
    # 处理每一个instance
    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        # 获取坐标
        with open(input_path) as f:
            j = json.load(f)
        platformDtoList = j["algorithmBaseParamDto"]["platformDtoList"]
        coordsInfo = {}
        # 添加起点和终点
        coordsInfo["platform00"] = [0, 0]
        for plat in platformDtoList:
            platCoords = [plat["x_coords"], plat["y_coords"]]
            coordsInfo[plat["platformCode"]] = (platCoords)
        if len(platformDtoList)+1 < 10:
            endPoint = "platform0" + str(len(platformDtoList)+1)
        else:
            endPoint = "platform" + str(len(platformDtoList)+1)
        coordsInfo[endPoint] = [10000, 10000]
        output_path = os.path.join(output_dir, file_name)
        # 获取路径
        pathResultInfoDict = read_result_file(input_path, output_path)
        # print("path:", pathResultInfoDict)
        # print("coords:", coordsInfo)
        for i in range(len(pathResultInfoDict["pathResult"])):
            save_path1 = "D:\\Desktop\\sz_result\\path_visualization\\" + j["estimateCode"]
            if not os.path.exists(save_path1):
                os.makedirs(save_path1)
            save_path2 = save_path1 + "\\solution" + str(i+1) + ".png"
            plot_paths_with_coordinates(coordsInfo, pathResultInfoDict["pathResult"][i], save_path2)

if __name__ == "__main__":
    # 定义输入目录、输出目录、结果文件路径和可视化目录
    input_dir = "D:\\Desktop\\python_code\\Supply_Chain_Packing_and_Routing\\inputs"
    output_dir = "D:\\Desktop\\python_code\\Supply_Chain_Packing_and_Routing\\outputs"
    result_file = "D:\\Desktop\\sz_result\\result\\checkResult.txt"
    visualization_dir = "D:\\Desktop\\sz_result\\visualization"
    visualization_bin_packing(input_dir, output_dir, result_file, visualization_dir)
    visualization_path_routing(input_dir, output_dir)
