#!/usr/bin/env python3

"""
目前的简化：
1. task起点不重复
2. task不再有释放时间
"""

import copy
import time
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import math
from matplotlib import pyplot as plt
from itertools import chain
from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent
    from .Token import Token


class Map:
    """
    Class for map instance.
    用于表示和管理环境地图，支持未知区域探索和任务规划。
    """
    def __init__(self, map_info, tasks, laser_range, screen=False):
        # 基本地图信息
        self.screen = screen
        self.map = self.gen_gridmap(map_info)
        # 地图尺寸参数
        self.width = self.map.shape[1]
        self.height = self.map.shape[0]
        self.total_area = self.width * self.height
        self.known_area = 0

        
        
        # 前沿和中心点
        self.frontiers = list()
        self.centroids = dict()
        self.laser_range = laser_range
        
        self.task_map = self.__gen_task_map(tasks) # 可以先让任务的起点不一样
        # 存储任务信息，后续必须和task_map保持一致
        self.tasks_seen = self.get_tasks_seen()

        
        # 初始化已知区域计算
        self.update_known_area()


    def gen_gridmap(self, map_info: dict):
        """
        根据地图信息生成网格地图
        ---------
        2: base_points 基地点：可用于放置任务的基地位置
        1: obstacle 障碍物
        0: free space 自由空间
        -1: unknown space 未知空间
        """
        dimensions = map_info['dimensions']
        base_points = map_info['base_points']
        obstacles = map_info['obstacles']

        self.base_points = base_points
        self.obstacles = obstacles
        # 初始化地图
        map_grid = np.full((dimensions[1], dimensions[0]), 0, dtype=np.int32)

        # 设置障碍物
        for obstacle in obstacles:
            x, y = obstacle
            map_grid[y][x] = 1
            
        # 设置基地点
        for base_point in base_points:
            x, y = base_point
            map_grid[y][x] = 2

        return map_grid
    
    def __gen_task_map(self, tasks):
        """
        生成任务地图，使用二维列表存储任务信息
        每个任务只需要记录起点，完成后统一送回基地点
        """
        # 初始化二维列表，默认值为None
        task_map = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        # 记录所有任务位置
        for task in tasks:
            task_name = task['task_name']
            start_x, start_y = task['start']
            start_time = task['start_time']
            
            # 在起点位置记录任务信息
            task_map[start_y][start_x] = task # task本身就是一个字典，包含任务名称、起点、终点等信息
        
        return task_map
    


    ##==================================================
    # 地图更新函数
    ##==================================================
    def update(self, realmap: "Map", senses, token: "Token"):
        """
        根据机器人的感知信息更新地图
        ----------
        Parameters:
            realmap: (Map), 真实地图的信息
            senses: (list of 2-tuple), [(x,y), ...], 感知到的坐标
        """
        ts = time.time()
        
        # 更新地图单元格
        for x, y in senses:
            self.map[y][x] = realmap.map[y][x]
            
            # 同步更新任务地图
            # 如果真实地图上这个位置有任务信息，而当前地图没有，则更新
            if realmap.task_map[y][x] is not None:
                if self.task_map[y][x] is None or self.task_map[y][x] != realmap.task_map[y][x]:
                    self.task_map[y][x] = realmap.task_map[y][x]
                    token.unassigned_tasks.add(realmap.task_map[y][x]['task_name']) #更新unassigned_tasks
        
        self.get_tasks_seen()
        # 更新已知区域
        self.update_known_area()
        
        print('[Map] 更新占用地图完成，耗时: %ss.' % round(time.time()-ts, 5))
        
        # 更新前沿和中心点
        ts = time.time()
        self.get_frontiers()
        self.gen_centroids()
        print('[Map] 找到前沿点并聚类完成，耗时: %ss.' % round(time.time()-ts, 5))

    def update_known_area(self):
        """
        更新已知区域的面积
        """
        # Count all cells that are not -1 (unknown)
        known_area = np.sum(self.map != -1)
        self.known_area = known_area
        return known_area

    ##==================================================
    # 前沿和中心点处理
    ##==================================================
    def get_frontiers(self):
        """
        获取前沿点，即已探索和未探索区域的边缘
        """
        self.frontiers = []
        for x in range(self.width):
            for y in range(self.height):
                if self.map[y][x] == 0:  # 只考虑自由空间
                    if self.has_unexplored_neighbors((x,y)):
                        self.frontiers.append((x,y))
        return self.frontiers

    def gen_centroids(self, method='mean_shift', bandwidth=3):
        """
        对前沿点进行聚类获取中心点，并计算各种权重
        """
        self.centroids = dict()
        #bandwidth = estimate_bandwidth(self.frontiers, quantile=0.2, n_jobs=-1) if bandwidth is None else bandwidth
        if len(self.frontiers) > 0:
            if method == 'mean_shift':
                centroids_dict = self.__mean_shift(self.frontiers, bandwidth)
                # 评估每个中心点周围的未知区域
                for centroid, frontiers in centroids_dict.items():
                    point = Centroid(centroid, frontiers, self.map, self.laser_range)
                    self.centroids[centroid] = {
                        'frontiers': point.frontiers,
                        'bound_box': point.bound_box,
                        'area': point.unknown_area,
                    }
                self.print('生成了新的中心点！')
        else:
            self.print('没有前沿点！')
            return None
        return self.centroids

    def __mean_shift(self, frontiers, bandwidth=3):
        """
        使用均值漂移(Mean Shift)方法聚类前沿点
        """
        # 转换为numpy数组
        frontiers = np.array(frontiers)
        ms = MeanShift(bandwidth=bandwidth)
        ms.fit(frontiers)
        labels = ms.labels_
        unique_labels = np.unique(labels)
        
        # 均值漂移聚类
        centroids = dict()
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            cluster_points = frontiers[cluster_indices]
            
            # 计算聚类的平均点作为前沿中心
            centroid = np.mean(cluster_points, axis=0)
            centroid = tuple(np.int32(centroid)) # 整数化
            
            # 如果中心点不是自由空间，使用最近的自由空间点
            if self.map[centroid[1]][centroid[0]] != 0:
                cluster_points_list = [tuple(p) for p in cluster_points]
                cluster_points_list = sorted(cluster_points_list, key=lambda p: math.dist(p, centroid))
                centroid = cluster_points_list[0]
                
            centroids[centroid] = cluster_points
        return centroids

    ##====================
    # 工具函数
    ##====================
    def get_tasks_seen(self):
        """
        获取当前已知的任务列表
        """
        tasks_seen = []
        for y in range(self.height):
            for x in range(self.width):
                task_info = self.task_map[y][x]
                if task_info is not None:
                    tasks_seen.append(task_info)
        self.tasks_seen = tasks_seen
        return tasks_seen

    def has_unexplored_neighbors(self, node):
        """
        检查一个单元格是否有未探索的邻居
        """
        # 检查四向邻居是否有未探索的单元格
        neighbor_shift = [
            (0,1), (1,0), (-1,0), (0,-1),
        ]
        for dx, dy in neighbor_shift:
            nx, ny = node[0] + dx, node[1] + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.map[ny][nx] == -1:
                    return True
        return False

    def back_to_unknown(self):
        """
        创建一个所有单元格都是未知的地图副本
        """
        map_unknown = copy.deepcopy(self)
        # 将整个地图标记为未知
        map_unknown.map = np.full((self.height, self.width), -1, dtype=np.int32)
        for (x,y) in self.base_points:
            map_unknown.map[y][x] = 2 # 基地点已知
        # 将任务地图也标记为未知
        map_unknown.task_map = [[None for _ in range(self.width)] for _ in range(self.height)]
        map_unknown.tasks_seen = []
        return map_unknown



    @classmethod
    def merge_maps(cls, maps_list: list["Map"]):
        """
        合并多个Map实例的信息, 会通过deepcopy给各个agent创建新的地图实例
        Parameters:
            maps_list: (list of Map), 要合并的Map实例列表
            comm_graph: 可以通信的一系列agents
        Returns:
            merged_map: (Map), 合并后的新Map实例
        """
        # 以第一个地图为基础
        base_map = maps_list[0]
        

        merged_map = copy.deepcopy(base_map)  # 创建一个新的地图实例
        
        # 合并所有地图的已知区域
        for idx, current_map in enumerate(maps_list):
            for y in range(merged_map.height):
                for x in range(merged_map.width):
                    # 如果当前地图的该点是已知的，则合并到新地图
                    if current_map.map[y][x] != -1:
                        merged_map.map[y][x] = current_map.map[y][x]
                    
                    # 合并任务信息，后续可以据此更新tasks_seen
                    if current_map.task_map[y][x] is not None:
                        merged_map.task_map[y][x] = current_map.task_map[y][x]
        

        # 更新已知区域数量
        merged_map.update_known_area()
        
        # 重新计算前沿点和中心点
        merged_map.get_frontiers()
        merged_map.gen_centroids()
        
        # 更新任务信息
        merged_map.get_tasks_seen()
        
        return merged_map

    @staticmethod
    def color_the_map(map_grid):
        """
        为地图着色以便可视化
        """
        height, width = map_grid.shape
        map_colored = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 为不同类型的单元格分配颜色
        for x in range(width):
            for y in range(height):
                if map_grid[y][x] == -1:  # 未知区域
                    map_colored[y][x] = [105, 105, 105]  # 灰色
                elif map_grid[y][x] == 0:  # 自由空间
                    map_colored[y][x] = [255, 255, 255]  # 白色
                elif map_grid[y][x] == 1:  # 障碍物
                    map_colored[y][x] = [0, 0, 0]       # 黑色
                elif map_grid[y][x] == 2:  # 基地
                    map_colored[y][x] = [47, 79, 79]    # 深青色
                    
        return map_colored

    ##=================
    # 控制台输出
    ##=================
    def print(self, string, line=False):
        """输出调试信息到控制台"""
        if self.screen:
            if line:
                print('----------')
            print(f'[Map] ' + string)

    ##=================
    # 可视化函数
    ##=================
    

    def show2(self, show_tasks=True, show_frontiers=False, show_centroids=False, show_bases=True, figsize=None):
        """
        显示地图，可选显示任务、前沿点、中心点和基地点
        使用散点图方式展示，具有更好的视觉效果和布局
        
        Parameters:
            show_tasks: 是否显示任务
            show_frontiers: 是否显示前沿点
            show_centroids: 是否显示中心点
            show_bases: 是否显示基地点
            figsize: 自定义图表大小，如 (width, height)
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.colors import LinearSegmentedColormap
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
        plt.rcParams['axes.unicode_minus'] = False  # 显示负号

        # 设置图表大小
        if figsize is None:
            # 根据地图尺寸动态调整图表大小
            aspect_ratio = self.height / self.width
            fig_width = min(15, max(8, self.width/5))
            fig_height = fig_width * aspect_ratio
            figsize = (fig_width, fig_height)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        ax.set_title('地图视图', fontsize=16, fontweight='bold')
        
        # 设置坐标轴范围，确保显示整个网格
        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)
        ax.set_aspect('equal')  # 保持网格单元为正方形
        
        # 设置刻度间隔
        tick_spacing = max(1, min(5, self.width // 10))
        x_ticks = range(0, self.width, tick_spacing)
        y_ticks = range(0, self.height, tick_spacing)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xlabel('X坐标', fontsize=12)
        ax.set_ylabel('Y坐标', fontsize=12)
        
        # 手动绘制网格线
        for x in range(self.width + 1):
            ax.axvline(x=x - 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        for y in range(self.height + 1):
            ax.axhline(y=y - 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # 收集不同类型的单元格
        unknown_cells = []
        free_cells = []
        obstacle_cells = []
        base_cells = []
        
        for y in range(self.height):
            for x in range(self.width):
                cell_value = self.map[y][x]
                if cell_value == -1:        # 未知区域
                    unknown_cells.append((x, y))
                elif cell_value == 0:       # 自由空间
                    free_cells.append((x, y))
                elif cell_value == 1:       # 障碍物
                    obstacle_cells.append((x, y))
                elif cell_value == 2:       # 基地点
                    base_cells.append((x, y))
        
        # 使用矩形绘制地图元素，而不是散点
        # 这样可以得到更好的填充效果
        
        # 绘制自由空间（白色背景）
        for x, y in free_cells:
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='white', 
                                    edgecolor='lightgray', linewidth=0.5))
        
        # 绘制未知区域（灰色）
        for x, y in unknown_cells:
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='gray', 
                                    alpha=0.5))
        
        # 绘制障碍物（黑色）
        for x, y in obstacle_cells:
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='black'))
        
        # 绘制基地点（深青色）
        if show_bases and base_cells:
            for x, y in base_cells:
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='darkslategray', 
                                        alpha=0.7))
                ax.text(x, y, "基", fontsize=10, ha='center', va='center', 
                    color='white', fontweight='bold')
        
        # 显示任务
        if show_tasks:
            # 收集未拾取和已拾取的任务
            unpicked_tasks = []
            picked_tasks = []
            task_labels = []
            
            for y in range(self.height):
                for x in range(self.width):
                    task_info = self.task_map[y][x]
                    if task_info is not None:

                        
                        # 添加任务标签
                        task_labels.append((x, y, task_info['task_name']))
            
            # 绘制未拾取的任务（绿色圆形标记）
            if unpicked_tasks:
                for x, y in unpicked_tasks:
                    circle = plt.Circle((x, y), 0.4, color='green', alpha=0.8, zorder=5)
                    ax.add_patch(circle)
            
            # 绘制已拾取的任务（黄色圆形标记，半透明）
            if picked_tasks:
                for x, y in picked_tasks:
                    circle = plt.Circle((x, y), 0.4, color='yellow', alpha=0.5, zorder=5)
                    ax.add_patch(circle)
            
            # 添加任务标签
            for x, y, label in task_labels:
                ax.text(x, y, label, fontsize=8, color='black', weight='bold', 
                    ha='center', va='center', zorder=6)
        
        # 显示前沿点
        if show_frontiers and self.frontiers:
            for x, y in self.frontiers:
                marker = plt.Circle((x, y), 0.15, color='yellow', alpha=0.7, zorder=4)
                ax.add_patch(marker)
        
        # 显示中心点及其边界框
        if show_centroids and self.centroids:
            for centroid, data in self.centroids.items():
                # 绘制中心点（品红色星形）
                star_marker = plt.scatter(centroid[0], centroid[1], c='magenta', marker='*', 
                                    s=200, zorder=7)
                
                # 绘制边界框
                try:
                    bbox = data['bound_box']
                    x_min, y_min = bbox[0]
                    x_max, y_max = bbox[2]
                    width = x_max - x_min
                    height = y_max - y_min
                    rect = patches.Rectangle((x_min - 0.5, y_min - 0.5), width + 1, height + 1, 
                                        linewidth=2, edgecolor='magenta', facecolor='none', 
                                        linestyle='--', zorder=6)
                    ax.add_patch(rect)
                    
                    # 显示未知面积
                    ax.text(centroid[0] + 0.5, centroid[1] + 0.5, f"面积: {data['area']}", 
                        fontsize=9, color='magenta', weight='bold', zorder=7,
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
                except (IndexError, KeyError) as e:
                    print(f"无法绘制中心点 {centroid} 的边界框: {e}")
        
        # 创建图例
        legend_elements = []
        
        # 为不同地图元素添加图例
        legend_elements.append(patches.Patch(color='gray', alpha=0.5, label='未知区域'))
        legend_elements.append(patches.Patch(color='white', edgecolor='lightgray', label='自由空间'))
        legend_elements.append(patches.Patch(color='black', label='障碍物'))
        
        if show_bases and base_cells:
            legend_elements.append(patches.Patch(color='darkslategray', alpha=0.7, label='基地点'))
        
        if show_tasks:
            if unpicked_tasks:
                legend_elements.append(patches.Circle((0, 0), radius=0.1, color='green', alpha=0.8, label='未拾取任务'))
            if picked_tasks:
                legend_elements.append(patches.Circle((0, 0), radius=0.1, color='yellow', alpha=0.5, label='已拾取任务'))
        
        if show_frontiers and self.frontiers:
            legend_elements.append(patches.Circle((0, 0), radius=0.1, color='yellow', alpha=0.7, label='前沿点'))
        
        if show_centroids and self.centroids:
            legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='magenta', 
                                        markersize=15, label='中心点'))
        
        # 添加图例
        if legend_elements:
            legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
            legend.get_frame().set_alpha(0.7)
        
        # 添加信息框显示地图信息
        info_text = f"地图大小: {self.width}×{self.height}\n"
        info_text += f"已知区域: {self.known_area}/{self.total_area} ({self.known_area/self.total_area*100:.1f}%)\n"
        info_text += f"前沿点数: {len(self.frontiers)}\n"
        info_text += f"中心点数: {len(self.centroids)}"
        
        # 在图表左上角添加信息框
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax  # 返回图表对象，方便进一步自定义

class Centroid():
    """
    用于表示探索任务中的中心点
    """
    def __init__(self, centroid, frontiers, gridmap, laser_range):
        self.centroid = centroid
        self.frontiers = self.__frontiers(frontiers)
        self.gridmap = gridmap
        self.width = gridmap.shape[1]
        self.height = gridmap.shape[0]
        self.laser_range = laser_range
        self.bound_all_frontiers()
        self.evaluate_unknown_area()

    def __frontiers(self, frontiers):
        """
        统一前沿点的数据格式
        """
        frontiers_tuple = set()
        for frontier in frontiers:
            frontiers_tuple.add(tuple(frontier))
        return frontiers_tuple

    def bound_all_frontiers(self):
        """
        使用矩形框包围所有前沿点
        """
        x, y = self.centroid
        x_dir_r, x_dir_l, y_dir_u, y_dir_l = 0, 0, 0, 0
        wide = round(self.laser_range/2)
        
        # 确定沿X方向的扩展方向
        if x + wide < self.width and self.gridmap[y][min(x+wide, self.width-1)] == -1:
            x_dir_r = 1
        if x - wide >= 0 and self.gridmap[y][max(x-wide, 0)] == -1:
            x_dir_l = 1
            
        # 确定沿Y方向的扩展方向
        if y + wide < self.height and self.gridmap[min(y+wide, self.height-1)][x] == -1:
            y_dir_u = 1
        if y - wide >= 0 and self.gridmap[max(y-wide, 0)][x] == -1:
            y_dir_l = 1
            
        # 生成X轴的边界框
        self.x_min = min([frontier[0] for frontier in self.frontiers])
        self.x_max = max([frontier[0] for frontier in self.frontiers])
        if x_dir_l == 1:
            self.x_min = min(self.x_min, x-self.laser_range)
            self.x_min = max(self.x_min, 0)
        if x_dir_r == 1:
            self.x_max = max(self.x_max, x+self.laser_range)
            self.x_max = min(self.x_max, self.width-1)
            
        # 生成Y轴的边界框
        self.y_min = min([frontier[1] for frontier in self.frontiers])
        self.y_max = max([frontier[1] for frontier in self.frontiers])
        if y_dir_l == 1:
            self.y_min = min(self.y_min, y-self.laser_range)
            self.y_min = max(self.y_min, 0)
        if y_dir_u == 1:
            self.y_max = max(self.y_max, y+self.laser_range)
            self.y_max = min(self.y_max, self.height-1)
            
        # 获取边界框
        self.bound_box = [
            (self.x_min, self.y_min),
            (self.x_min, self.y_max),
            (self.x_max, self.y_max),
            (self.x_max, self.y_min),
        ]

    def evaluate_unknown_area(self):
        """
        评估边界框内的未知区域面积
        """
        self.unknown_area = 0
        for x in range(self.x_min, self.x_max+1):
            for y in range(self.y_min, self.y_max+1):
                if 0 <= x < self.width and 0 <= y < self.height:
                    if self.gridmap[y][x] == -1:
                        self.unknown_area += 1