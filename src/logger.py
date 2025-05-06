import os
import time
import json
import copy
import traceback
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import networkx as nx
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, List, Set, Tuple, Any

if TYPE_CHECKING:
    from .simulation import Simulation
    from .agent import Agent, Base
    from .map import Map
    from .Token import Token

import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """处理NumPy数据类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super().default(obj)
            

    

class Logger:
    """
    用于记录和可视化分布式仿真过程中的数据
    """
    def __init__(self, simulation: "Simulation", log_dir: str = "logs"):
        """
        初始化Logger
        :param simulation: 仿真对象
        :param log_dir: 日志保存目录
        """
        self.simulation = simulation
        self.log_dir = log_dir
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(log_dir, f"session_{self.timestamp}")
        self.create_log_directories()
        
        # 设置matplotlib中文支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 初始化存储数据的变量
        self.agent_positions = {}  # 记录代理位置随时间变化: {time: {agent_name: (x, y)}}
        self.task_status = {}      # 任务状态: {task_name: {"released": t1, "picked": t2, "completed": t3}}
        self.communication_graphs = {}  # 通信图: {time: [graph1, graph2, ...]}
        self.agent_tasks = {}      # 代理任务分配: {time: {agent_name: [task_names]}}
        self.picked_tasks = {}     # 代理已拾取任务: {time: {agent_name: [task_names]}}
        self.agent_paths = {}      # 代理路径记录: {agent_name: [{t, pos}, ...]}
        self.frontier_history = {} # 边界历史: {time: [frontier_positions]}
        self.performance_metrics = {"completed_task_nums": []}

        # 添加新的存储结构，用于保存聚合地图信息
        self.aggregated_maps = {}  # 每个时间步的聚合地图: {time: merged_map_copy}
        self.known_area_history = {}  # 已知区域随时间变化: {time: known_area}
        self.frontier_positions = {}  # 每个时间步的前沿点位置: {time: [(x,y), ...]}
        self.centroid_positions = {}  # 每个时间步的中心点位置及面积: {time: {(x,y): {'area': area, 'bound_box': box}}}

        # 添加新字段存储已完成的任务
        self.completed_tasks_sets = {}  # 每个时间步的已完成任务集合     
        
        # 记录初始状态
        self.record_initial_state()
    
    def create_log_directories(self):
        """创建日志目录结构"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.session_dir):
            os.makedirs(self.session_dir)
        
        # 创建子目录
        self.data_dir = os.path.join(self.session_dir, "data")
        self.plots_dir = os.path.join(self.session_dir, "plots")
        self.animations_dir = os.path.join(self.session_dir, "animations")
        
        for directory in [self.data_dir, self.plots_dir, self.animations_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def record_initial_state(self):
        """记录初始状态"""
        # 记录地图信息
        map_data = {
            "dimensions": (self.simulation.real_map.width, self.simulation.real_map.height),
            "obstacles": list(self.simulation.real_map.obstacles),
            "base_points": self.simulation.base.positions
        }
        with open(os.path.join(self.data_dir, "map_info.json"), "w") as f:
            json.dump(map_data, f, indent=4)
        
        # 记录代理初始位置
        time_step = 0
        self.agent_positions[time_step] = {}
        for agent in self.simulation.agents:
            self.agent_positions[time_step][agent.name] = agent.pos
            self.agent_paths[agent.name] = [{"t": -1, "pos": agent.pos}]
        
        # 记录任务信息
        task_info = []
        for task in self.simulation.agents[0].token._tasks:
            task_data = {
                "name": task["task_name"],
                "start": task["start"],
                "goal": task["goal"],
                "start_time": task.get("start_time", 0)
            }
            task_info.append(task_data)
            # 初始化任务状态
            self.task_status[task["task_name"]] = {
                "released": task.get("start_time", 0), 
                "picked": None, 
                "completed": None
            }
        
        with open(os.path.join(self.data_dir, "tasks_info.json"), "w") as f:
            json.dump(task_info, f, indent=4)
            
        # 记录边界点初始状态
        self.frontier_history[0] = list(self.simulation.agents[0].map.centroids.keys()) if hasattr(self.simulation.agents[0].map, 'centroids') else []
        # 记录初始聚合地图
        initial_map = self._get_aggregated_map()
        self.aggregated_maps[0] = initial_map
        self.known_area_history[0] = initial_map.known_area
        self.frontier_positions[0] = initial_map.frontiers.copy() if hasattr(initial_map, 'frontiers') else []
        self.centroid_positions[0] = deepcopy(initial_map.centroids) if hasattr(initial_map, 'centroids') else {}


    def _get_aggregated_map(self):
        """获取所有agent地图的聚合视图"""
        agent_maps = [agent.map for agent in self.simulation.agents]
        if not agent_maps:
            return None
        
        # 使用Map类的merge_maps方法聚合所有地图
        merged_map = agent_maps[0].merge_maps(agent_maps)
        return merged_map

    def update(self, time_step):
        """每个时间步更新日志数据"""
        # 记录代理位置
        self.agent_positions[time_step] = {}
        for agent in self.simulation.agents:
            self.agent_positions[time_step][agent.name] = agent.pos
            # 更新代理路径历史
            self.agent_paths[agent.name].append({"t": time_step, "pos": agent.pos})
        
        # 记录通信图
        if self.simulation.comm_graphs:
            # 转换通信图为可序列化格式
            serializable_graphs = []
            for graph in self.simulation.comm_graphs:
                serializable_graph = []
                for node in graph:
                    if node.name == "base":
                        serializable_graph.append({"name": "base", "positions": node.positions})
                    else:
                        serializable_graph.append({"name": node.name, "position": node.pos})
                serializable_graphs.append(serializable_graph)
            
            self.communication_graphs[time_step] = serializable_graphs
        
        # 记录代理任务分配
        self.agent_tasks[time_step] = {}
        self.picked_tasks[time_step] = {}


        for agent in self.simulation.agents:
            self.agent_tasks[time_step][agent.name] = agent.token.agents_to_tasks.get(agent.name, []).copy()
            self.picked_tasks[time_step][agent.name] = agent.picked_tasks.copy()
        

        # 保存当前时间步的已完成任务
        completed_tasks_set = self.simulation._c_tasks
        self.completed_tasks_sets[time_step] = completed_tasks_set.copy()

        for task_name in completed_tasks_set:
            if task_name in self.task_status and self.task_status[task_name]["completed"] is None:
                self.task_status[task_name]["completed"] = time_step
        
        for agent in self.simulation.agents:
            for task_name in agent.picked_tasks:
                if task_name in self.task_status and self.task_status[task_name]["picked"] is None:
                    self.task_status[task_name]["picked"] = time_step
        
        # 记录性能指标
        self.performance_metrics["completed_task_nums"].append(len(completed_tasks_set))
        
        # 记录边界点
        self.frontier_history[time_step] = list(self.simulation.agents[0].map.centroids.keys()) if hasattr(self.simulation.agents[0].map, 'centroids') else []
         # 更新聚合地图信息
        merged_map = self._get_aggregated_map()
        self.aggregated_maps[time_step] = merged_map
        self.known_area_history[time_step] = merged_map.known_area
        self.frontier_positions[time_step] = merged_map.frontiers.copy() if hasattr(merged_map, 'frontiers') else []
        self.centroid_positions[time_step] = deepcopy(merged_map.centroids) if hasattr(merged_map, 'centroids') else {}

    def save_data(self):
        """保存所有记录的数据"""
        # 保存代理路径数据
        with open(os.path.join(self.data_dir, "agent_paths.json"), "w") as f:
            json.dump(self.agent_paths, f, indent=4, cls=NumpyEncoder)
        
        # 保存任务状态数据
        with open(os.path.join(self.data_dir, "task_status.json"), "w") as f:
            json.dump(self.task_status, f, indent=4, cls=NumpyEncoder)
        
        # 保存通信图数据
        with open(os.path.join(self.data_dir, "communication_graphs.json"), "w") as f:
            json.dump(self.communication_graphs, f, indent=4, cls=NumpyEncoder)
        
        # 保存代理任务分配
        with open(os.path.join(self.data_dir, "agent_tasks.json"), "w") as f:
            json.dump(self.agent_tasks, f, indent=4, cls=NumpyEncoder)
        
        # 保存已拾取任务
        with open(os.path.join(self.data_dir, "picked_tasks.json"), "w") as f:
            json.dump(self.picked_tasks, f, indent=4, cls=NumpyEncoder)
            
        # 保存性能指标
        with open(os.path.join(self.data_dir, "performance_metrics.json"), "w") as f:
            json.dump(self.performance_metrics, f, indent=4, cls=NumpyEncoder)
        
        # 保存边界点历史
        with open(os.path.join(self.data_dir, "frontier_history.json"), "w") as f:
            json.dump(self.frontier_history, f, indent=4, cls=NumpyEncoder)

         # 保存已知区域历史
        with open(os.path.join(self.data_dir, "known_area_history.json"), "w") as f:
            json.dump(self.known_area_history, f, indent=4, cls=NumpyEncoder)
            
        # 保存前沿点和中心点信息 (只保存时间和位置，不保存整个地图对象)
        frontier_data = {t: [list(pos) for pos in positions] for t, positions in self.frontier_positions.items()}
        with open(os.path.join(self.data_dir, "frontier_positions.json"), "w") as f:
            json.dump(frontier_data, f, indent=4, cls=NumpyEncoder)
            
        # 修改存储中心点数据的方式，增加类型检查
        centroid_data = {}
        for t, centroids in self.centroid_positions.items():
            centroid_data[t] = {}
            for pos, data in centroids.items():
                try:
                    # 将位置转换为字符串键
                    pos_key = str(pos)
                    centroid_data[t][pos_key] = {
                        'area': int(data.get('area', 0)) if 'area' in data else 0
                    }
                    
                    # 处理边界框数据，增加类型检查
                    if 'bound_box' in data:
                        bound_box = data['bound_box']
                        # 确保bound_box是列表或元组
                        if isinstance(bound_box, (list, tuple)):
                            try:
                                # 处理每个点 - 使用安全的方式
                                safe_bound_box = []
                                for point in bound_box:
                                    if isinstance(point, (list, tuple)) and len(point) == 2:
                                        # 正常的坐标点
                                        safe_bound_box.append([int(point[0]), int(point[1])])
                                    elif hasattr(point, 'item'):  # numpy标量
                                        # 单个数值，不是坐标点（这是处理错误的情况）
                                        safe_bound_box.append(int(point.item()))
                                    else:
                                        # 其他情况，尝试直接转换
                                        safe_bound_box.append(int(point) if isinstance(point, (int, float, np.number)) else point)
                                
                                centroid_data[t][pos_key]['bound_box'] = safe_bound_box
                            except (TypeError, ValueError) as e:
                                print(f"无法转换边界框数据: {e}, 使用空列表替代")
                                centroid_data[t][pos_key]['bound_box'] = []
                        else:
                            # bound_box不是列表或元组，可能是标量
                            centroid_data[t][pos_key]['bound_box'] = []
                    else:
                        centroid_data[t][pos_key]['bound_box'] = []
                except Exception as e:
                    print(f"处理中心点 {pos} 的数据时出错: {e}")
                    # 使用一个安全的默认值
                    centroid_data[t][str(pos)] = {'area': 0, 'bound_box': []}
        
        with open(os.path.join(self.data_dir, "centroid_positions.json"), "w") as f:
            json.dump(centroid_data, f, indent=4, cls=NumpyEncoder)

    def generate_summary(self):
        """生成仿真摘要"""
        max_time = max(self.agent_positions.keys())
        completed_tasks_count = len([t for t, status in self.task_status.items() if status["completed"] is not None])
        
        summary = {
            "total_time_steps": max_time,
            "number_of_agents": len(self.simulation.agents),
            "total_tasks": len(self.simulation.agents[0].token._tasks),
            "completed_task_nums": completed_tasks_count,
            "avg_task_completion_time": self._calculate_avg_completion_time(),
            "communication_events": self._count_communication_events()
        }
        
        with open(os.path.join(self.session_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=4)
            
        # 将摘要也打印到控制台
        print("\n===== 仿真摘要 =====")
        for key, value in summary.items():
            print(f"{key}: {value}")
        print("===================\n")
        
        return summary
    
    def _calculate_avg_completion_time(self):
        """计算平均任务完成时间"""
        completion_times = []
        for task_name, status in self.task_status.items():
            if status["completed"] is not None:
                completion_time = status["completed"] - status["released"]
                completion_times.append(completion_time)
                
        if not completion_times:
            return 0
        return sum(completion_times) / len(completion_times)
    
    def _count_communication_events(self):
        """计算通信事件次数"""
        # 简单统计每个时间步中所有通信组的数量总和
        comm_events = 0
        for t, graphs in self.communication_graphs.items():
            for graph in graphs:
                # 每个通信组中，代理之间的通信次数是n*(n-1)/2，其中n是代理数量
                n = len([node for node in graph if node["name"] != "base"])
                if n > 1:
                    comm_events += (n * (n - 1)) // 2
        return comm_events
    
    def plot_agent_paths(self):
        """绘制代理路径图"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 获取地图尺寸
        width, height = self.simulation.real_map.width, self.simulation.real_map.height
        
        # 绘制完整网格
        self._draw_complete_grid(ax, width, height)
        
        # 绘制障碍物
        obstacles = self.simulation.real_map.obstacles
        for obs in obstacles:
            ax.add_patch(plt.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, color='black'))
        
        # 绘制基站
        for base_pos in self.simulation.base.positions:
            ax.add_patch(plt.Rectangle((base_pos[0] - 0.5, base_pos[1] - 0.5), 1, 1, color='lightblue', alpha=0.7))
        
        # 为每个代理绘制路径
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.simulation.agents)))
        for i, agent in enumerate(self.simulation.agents):
            path = self.agent_paths[agent.name]
            xs = [p["pos"][0] for p in path]
            ys = [p["pos"][1] for p in path]
            ax.plot(xs, ys, '-o', color=colors[i], label=agent.name, alpha=0.7)
        
        # 添加标签和图例
        ax.set_title('Agent Paths')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "agent_paths.png"), dpi=300)
        plt.close()

    def plot_exploration_progress(self):
        """绘制地图探索进度"""
        if not self.known_area_history:
            print("没有可用的地图探索数据")
            return
            
        time_steps = sorted(self.known_area_history.keys())
        known_areas = [self.known_area_history[t] for t in time_steps]
        total_area = self.simulation.real_map.total_area
        
        # 计算探索百分比
        exploration_percentages = [area / total_area * 100 for area in known_areas]
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, exploration_percentages, '-o', linewidth=2)
        plt.title('地图探索进度')
        plt.xlabel('时间步')
        plt.ylabel('已探索区域百分比 (%)')
        plt.grid(True)
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "exploration_progress.png"), dpi=300)
        plt.close()
    
    def create_animation(self):
        """创建代理移动动画，包含地图未知区域、前沿点和中心点的可视化"""
        import matplotlib.animation as animation
        import matplotlib.patches as mpatches
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
        plt.rcParams['axes.unicode_minus'] = False    # 显示负号
        
        # 创建图形，使用GridSpec分割为上下两个区域
        fig = plt.figure(figsize=(12, 12))
        
        # 创建网格布局，上方为地图（高度为3），下方为任务统计图（高度为1）
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
        
        # 创建上下两个子图
        ax = fig.add_subplot(gs[0, 0])  # 地图子图
        task_ax = fig.add_subplot(gs[1, 0])  # 任务统计子图
        
        width, height = self.simulation.real_map.width, self.simulation.real_map.height
        
        # 设置地图坐标轴范围和标签
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)
        ax.set_aspect('equal')  # 保持网格单元为正方形
        ax.set_xlabel('X坐标', fontsize=12)
        ax.set_ylabel('Y坐标', fontsize=12)
        ax.set_title('仿真动态可视化', fontsize=16, fontweight='bold')
        ax.autoscale(False)  # 禁用自动缩放
        
        # 设置刻度间隔
        tick_spacing = max(1, min(5, width // 10))
        x_ticks = range(0, width, tick_spacing)
        y_ticks = range(0, height, tick_spacing)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        
        # 绘制初始网格线
        for x in range(width + 1):
            ax.axvline(x=x - 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        for y in range(height + 1):
            ax.axhline(y=y - 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # 绘制静态元素 - 障碍物
        obstacles = self.simulation.real_map.obstacles
        for obs in obstacles:
            ax.add_patch(plt.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, facecolor='black', zorder=2))
        
        # 绘制基地点（蓝色）
        for base_pos in self.simulation.base.positions:
            ax.add_patch(plt.Rectangle((base_pos[0] - 0.5, base_pos[1] - 0.5), 1, 1, 
                                    facecolor='royalblue', alpha=0.8, zorder=3))
            ax.text(base_pos[0], base_pos[1], "基", fontsize=10, ha='center', va='center', 
                    color='white', fontweight='bold', zorder=15)
        
        # 在右上角添加时间步显示
        time_step_text = ax.text(0.98, 0.98, 'Time: 0', transform=ax.transAxes,
                                fontsize=12, ha='right', va='top',
                                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'), zorder=15)
        
        # 在左上角添加信息显示
        info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                        fontsize=10, verticalalignment='top', 
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8), zorder=15)
        
        # 为每个代理创建一个散点
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.simulation.agents)))
        scatter_plots = []
        agent_names = []
        agent_labels = []
        
        for i, agent in enumerate(self.simulation.agents):
            scatter = ax.scatter([], [], s=120, color=colors[i], marker='o', 
                                edgecolors='black', linewidths=1.5, zorder=10, label=agent.name)
            scatter_plots.append(scatter)
            agent_names.append(agent.name)
            
            # 添加代理标签
            label = ax.text(0, 0, "", fontsize=9, ha='center', va='top', 
                        color='black', fontweight='bold', zorder=15)
            agent_labels.append(label)
        
        # 为任务创建散点（绿色圆形）
        task_markers = []
        task_texts = []
        
        # 为前沿点创建散点（黄色小圆点）
        frontier_markers = []
        
        # 为中心点创建散点（洋红色星形）
        centroid_markers = []
        centroid_boxes = []
        centroid_texts = []
        
        # 创建用于绘制地图的对象（未知区域、自由空间等）
        unknown_patches = []
        free_patches = []
        
        # 设置任务统计子图
        task_ax.set_title('任务统计', fontsize=14)
        task_ax.set_xlabel('时间步', fontsize=12)
        task_ax.set_ylabel('任务数量', fontsize=12)
        task_ax.grid(True, linestyle='--', alpha=0.7)
        
        # 初始化任务统计数据
        time_steps = []
        discovered_tasks = []
        completed_task_nums = []
        
        # 创建任务进度线图
        discovered_line, = task_ax.plot([], [], 'b-', marker='o', label='已发现任务')
        completed_line, = task_ax.plot([], [], 'r-', marker='s', label='已完成任务')
        
        # 添加任务进度图例
        task_ax.legend(loc='upper left')
        
        # 创建图例
        legend_elements = [
            mpatches.Patch(facecolor='gray', alpha=0.5, label='未知区域'),
            mpatches.Patch(facecolor='white', edgecolor='lightgray', label='自由空间'),
            mpatches.Patch(facecolor='black', label='障碍物'),
            mpatches.Patch(facecolor='royalblue', alpha=0.8, label='基地点'),
            mpatches.Patch(facecolor='green', alpha=0.8, label='任务'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', 
                    markersize=6, alpha=0.7, label='前沿点'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='magenta', 
                    markersize=15, label='中心点')
        ]
        
        # 为不同代理添加图例
        for i, agent in enumerate(self.simulation.agents):
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
                        markersize=10, label=agent.name)
            )
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8, 
                bbox_to_anchor=(1.0, 1.0))
        
        # 清除不需要的动态元素函数
        def clear_dynamic_elements():
            # 清除之前帧的动态元素
            for patch in unknown_patches + free_patches:
                if patch in ax.patches:
                    patch.remove()
            unknown_patches.clear()
            free_patches.clear()
            
            # 清除前沿点
            for marker in frontier_markers:
                if marker in ax.patches:
                    marker.remove()
            frontier_markers.clear()
            
            # 清除中心点
            for marker in centroid_markers:
                if marker in ax.collections:
                    marker.remove()
            centroid_markers.clear()
            
            # 清除中心点边界框
            for rect in centroid_boxes:
                if rect in ax.patches:
                    rect.remove()
            centroid_boxes.clear()
            
            # 清除中心点面积文本
            for text in centroid_texts:
                if text in ax.texts:
                    text.remove()
            centroid_texts.clear()
            
            # 清除任务标记
            for marker in task_markers:
                if marker in ax.patches:
                    marker.remove()
            task_markers.clear()
            
            # 清除任务文本
            for text in task_texts:
                if text in ax.texts:
                    text.remove()
            task_texts.clear()
        
        def update(frame):
            # 确保frame不超过最大时间步
            max_time = max(self.agent_positions.keys())
            current_frame = min(frame, max_time)
            
            # 清除动态元素
            clear_dynamic_elements()
            
            # 获取当前帧的聚合地图
            if current_frame in self.aggregated_maps:
                current_map = self.aggregated_maps[current_frame]
                
                # 绘制未知区域和自由空间
                unknown_cells = []
                free_cells = []
                
                for y in range(current_map.height):
                    for x in range(current_map.width):
                        cell_value = current_map.map[y][x]
                        if cell_value == -1:  # 未知区域
                            unknown_cells.append((x, y))
                        elif cell_value == 0:  # 自由空间
                            free_cells.append((x, y))
                
                # 绘制未知区域（灰色）- 设置较低的 zorder
                for x, y in unknown_cells:
                    patch = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='gray', alpha=0.5, zorder=1)
                    ax.add_patch(patch)
                    unknown_patches.append(patch)
                
                # 绘制自由空间（白色）- 设置较低的 zorder
                for x, y in free_cells:
                    patch = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='white', 
                                        edgecolor='lightgray', linewidth=0.5, zorder=1)
                    ax.add_patch(patch)
                    free_patches.append(patch)
                
                # 更新前沿点 (黄色圆点)
                if current_frame in self.frontier_positions:
                    for x, y in self.frontier_positions[current_frame]:
                        marker = plt.Circle((x, y), 0.15, facecolor='yellow', alpha=0.7, zorder=4)
                        ax.add_patch(marker)
                        frontier_markers.append(marker)
                
                # 更新中心点和边界框 (洋红色星形)
                if current_frame in self.centroid_positions:
                    for centroid, data in self.centroid_positions[current_frame].items():
                        # 绘制中心点（品红色星形）
                        star_marker = ax.scatter(centroid[0], centroid[1], c='magenta', marker='*', 
                                            s=200, zorder=5)
                        centroid_markers.append(star_marker)
                        
                        # 绘制边界框
                        try:
                            if 'bound_box' in data and data['bound_box']:
                                bbox = data['bound_box']
                                x_min, y_min = bbox[0]
                                x_max, y_max = bbox[2]
                                width = x_max - x_min
                                height = y_max - y_min
                                rect = mpatches.Rectangle((x_min - 0.5, y_min - 0.5), width + 1, height + 1, 
                                                    linewidth=2, edgecolor='magenta', facecolor='none', 
                                                    linestyle='--', zorder=5)
                                ax.add_patch(rect)
                                centroid_boxes.append(rect)
                                
                                # 显示未知面积
                                text = ax.text(centroid[0] + 0.5, centroid[1] + 0.5, f"面积: {data['area']}", 
                                            fontsize=9, color='magenta', weight='bold', zorder=15,
                                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
                                centroid_texts.append(text)
                        except (IndexError, KeyError) as e:
                            print(f"无法绘制中心点 {centroid} 的边界框: {e}")
                
                # 收集所有被拾取的任务
                all_picked_tasks = set()
                for agent_name, picked_tasks in self.picked_tasks.get(current_frame, {}).items():
                    all_picked_tasks.update(picked_tasks)
                
                # 使用历史记录的已完成任务集合，而不是直接访问simulation
                completed_tasks_set = set()
                if current_frame in self.completed_tasks_sets:
                    completed_tasks_set = self.completed_tasks_sets[current_frame]
                
                print(f"当前时间步: {current_frame}, 已完成任务: {completed_tasks_set}")
                # 显示未拾取的任务
                for task in current_map.tasks_seen:
                    task_name = task['task_name']
                    # 如果任务未完成且未被任何agent拾取
                    if task_name not in completed_tasks_set and task_name not in all_picked_tasks:
                        x, y = task['start']
                        # 绿色圆形表示任务，在中间显示任务名称
                        circle = plt.Circle((x, y), 0.4, facecolor='green', alpha=0.8, zorder=6)
                        ax.add_patch(circle)
                        task_markers.append(circle)
                        
                        # 添加任务名称文本
                        text = ax.text(x, y, task_name, fontsize=8, color='black', weight='bold', 
                                    ha='center', va='center', zorder=15)
                        task_texts.append(text)
                
                # 更新地图信息文本
                info_text_str = f"地图大小: {current_map.width}×{current_map.height}\n"
                info_text_str += f"已知区域: {current_map.known_area}/{current_map.total_area} "
                info_text_str += f"({current_map.known_area/current_map.total_area*100:.1f}%)\n"
                info_text_str += f"前沿点数: {len(self.frontier_positions.get(current_frame, []))}\n"
                info_text_str += f"中心点数: {len(self.centroid_positions.get(current_frame, {}))}"
                info_text.set_text(info_text_str)
            
            # 更新代理位置
            for i, agent_name in enumerate(agent_names):
                if current_frame in self.agent_positions and agent_name in self.agent_positions[current_frame]:
                    pos = self.agent_positions[current_frame][agent_name]
                    # 更新散点位置 - 使用NumPy数组格式
                    scatter_plots[i].set_offsets(np.array([pos]))
                    scatter_plots[i].set_visible(True)  # 确保散点可见
                    
                    # 更新代理携带的任务标签
                    if current_frame in self.picked_tasks and agent_name in self.picked_tasks[current_frame]:
                        carried_tasks = self.picked_tasks[current_frame][agent_name]
                        if carried_tasks:
                            task_text = f"{agent_name}\n[{','.join(carried_tasks)}]"
                        else:
                            task_text = agent_name
                        
                        agent_labels[i].set_text(task_text)
                        agent_labels[i].set_position((pos[0], pos[1] - 0.3))
                        agent_labels[i].set_alpha(1)  # 显示标签
                    else:
                        agent_labels[i].set_text(agent_name)
                        agent_labels[i].set_position((pos[0], pos[1] - 0.3))
                        agent_labels[i].set_alpha(1)
            
            # 更新时间步文本
            time_step_text.set_text(f"Time: {current_frame}")
            
            # 更新任务统计数据
            time_steps.append(current_frame)
            
            # 计算已发现和已完成的任务
            if current_frame in self.aggregated_maps:
                current_map = self.aggregated_maps[current_frame]
                discovered_count = len(current_map.tasks_seen)
                completed_count = len([t for t, status in self.task_status.items() 
                                    if status["completed"] is not None and status["completed"] <= current_frame])
                
                discovered_tasks.append(discovered_count)
                completed_task_nums.append(completed_count)
                
                # 更新线图数据
                discovered_line.set_data(time_steps, discovered_tasks)
                completed_line.set_data(time_steps, completed_task_nums)
                
                # 动态调整X轴范围，确保所有点都可见
                task_ax.set_xlim(0, max(current_frame + 2, 10))
                
                # 动态调整Y轴范围，确保所有点都可见，并留出一些空间
                task_ax.set_ylim(0, max(max(discovered_tasks, default=0) + 2, 10))
                
                # 设置x轴刻度，确保为整数
                task_ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            
            return scatter_plots + [discovered_line, completed_line, time_step_text, info_text] + agent_labels
        
        # 计算帧数
        max_time = max(self.agent_positions.keys())
        
        # 创建动画
        try:
            anim = animation.FuncAnimation(
                fig, 
                update, 
                frames=range(max_time + 1),
                interval=400,
                blit=False
            )
            
            # 保存动画
            anim_path = os.path.join(self.animations_dir, 'exploration_animation.mp4')
            writer = animation.FFMpegWriter(fps=1, bitrate=5000)  # 降低帧率以便更好观察
            anim.save(anim_path, writer=writer)
            print(f"动画已保存至 {anim_path}")
            
        except Exception as e:
            print(f"创建动画时出错: {e}")
            traceback.print_exc()
            # 尝试使用备用方法
            try:
                writer = animation.PillowWriter(fps=2)  # 使用较低的帧率
                anim_path = os.path.join(self.animations_dir, 'exploration_animation.gif')
                anim.save(anim_path, writer=writer)
                print(f"动画已保存为GIF: {anim_path}")
            except Exception as e2:
                print(f"保存GIF时出错: {e2}")
        finally:
            plt.close(fig)
    
    def plot_communication_graphs(self):
        """绘制通信图在不同时间步的快照"""
        import matplotlib.patches as mpatches
        
        max_time = max(self.communication_graphs.keys()) if self.communication_graphs else 0
        times_to_sample = min(10, max_time)  # 最多取10个时间点
        
        if max_time == 0:
            print("没有可用的通信图数据")
            return
        
        sample_times = sorted(list(self.communication_graphs.keys()))[::max(1, len(self.communication_graphs) // times_to_sample)][:times_to_sample]
        
        for t in sample_times:
            if t not in self.communication_graphs:
                continue
                
            fig, ax = plt.subplots(figsize=(10, 8))
            width, height = self.simulation.real_map.width, self.simulation.real_map.height
            
            # 绘制完整网格
            self._draw_complete_grid(ax, width, height)
            
            # 绘制障碍物
            for obs in self.simulation.real_map.obstacles:
                ax.add_patch(plt.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, color='black'))
            
            # 绘制基站
            for base_pos in self.simulation.base.positions:
                ax.add_patch(plt.Rectangle((base_pos[0] - 0.5, base_pos[1] - 0.5), 1, 1, color='lightblue', alpha=0.7))
            
            # 为每个通信组创建一个NetworkX图
            for i, graph in enumerate(self.communication_graphs[t]):
                G = nx.Graph()
                
                # 添加节点
                base_nodes = [node for node in graph if node["name"] == "base"]
                agent_nodes = [node for node in graph if node["name"] != "base"]
                
                for node in agent_nodes:
                    G.add_node(node["name"], pos=node["position"])
                
                # 为代理节点间添加边
                nodes = list(G.nodes())
                for j in range(len(nodes)):
                    for k in range(j+1, len(nodes)):
                        # 检查两个节点是否在通信范围内
                        pos1 = G.nodes[nodes[j]]["pos"]
                        pos2 = G.nodes[nodes[k]]["pos"]
                        distance = max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))
                        
                        if distance <= self.simulation.agents[0].comm_range:  # 通信范围
                            G.add_edge(nodes[j], nodes[k])
                
                # 添加代理与基站的连接（使用特殊边样式）
                if base_nodes:
                    base_positions = base_nodes[0]["positions"]
                    for agent_node in agent_nodes:
                        agent_pos = agent_node["position"]
                        # 检查代理是否与任何基站点在通信范围内
                        for base_pos in base_positions:
                            distance = max(abs(agent_pos[0] - base_pos[0]), abs(agent_pos[1] - base_pos[1]))
                            if distance <= self.simulation.agents[0].comm_range:  # 通信范围
                                # 在图中添加到最近的基站点的连线
                                if agent_node["name"] in G:
                                    nearest_base_pos = min(base_positions, 
                                                         key=lambda bp: max(abs(agent_pos[0] - bp[0]), 
                                                                            abs(agent_pos[1] - bp[1])))
                                    ax.plot([agent_pos[0], nearest_base_pos[0]], 
                                            [agent_pos[1], nearest_base_pos[1]], 
                                            'g--', alpha=0.6, linewidth=1.5)  # 绿色虚线表示与基站的连接
                                break
                
                # 绘制代理之间的通信图
                pos = nx.get_node_attributes(G, 'pos')
                if pos:  # 确保有节点才进行绘制
                    nx.draw(G, pos, with_labels=True, node_size=500, node_color=f'C{i}', alpha=0.8, ax=ax)
            
            # 创建图例
            base_patch = mpatches.Patch(color='lightblue', alpha=0.7, label='基站')
            obstacle_patch = mpatches.Patch(color='black', label='障碍物')
            agent_comm_line = plt.Line2D([0], [0], color='blue', linewidth=2, label='代理间通信')
            base_comm_line = plt.Line2D([0], [0], color='green', linestyle='--', linewidth=1.5, label='代理-基站通信')
            
            # 添加图例
            ax.legend(handles=[base_patch, obstacle_patch, agent_comm_line, base_comm_line], 
                    loc='upper right', bbox_to_anchor=(1, 1))
            
            ax.set_title(f'通信图 - 时间步: {t}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f"comm_graph_t{t}.png"), dpi=300)
            plt.close()
    
    def plot_task_completion_over_time(self):
        """绘制随时间的任务完成情况"""
        max_time = max(self.agent_positions.keys())
        completed_tasks = []
        time_steps = list(range(max_time + 1))
        
        for t in time_steps:
            completed = sum(1 for _, status in self.task_status.items() 
                           if status["completed"] is not None and status["completed"] <= t)
            completed_tasks.append(completed)
            
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, completed_tasks, '-o', linewidth=2)
        plt.title('任务完成情况')
        plt.xlabel('时间步')
        plt.ylabel('已完成任务数量')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "tasks_completion.png"), dpi=300)
        plt.close()
    
    def plot_tasks_gantt_chart(self):
        """
        绘制任务执行甘特图，展示各个代理的任务执行情况
        - 根据容量为每个代理分配多个插槽
        - 合理分配任务到不同插槽，避免重叠
        """
        import matplotlib.patches as mpatches
        
        # 计算所需的高度（基于代理数量和它们的容量）
        total_slots = sum(agent.capacity for agent in self.simulation.agents)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(15, 2 + total_slots * 0.5))
        
        # 为甘特图设置Y轴标签
        y_labels = []
        y_positions = []
        y_ticks = []
        
        # 随机生成颜色映射，对每个任务使用一致的颜色
        tasks = list(self.task_status.keys())
        colors = plt.cm.tab20(np.linspace(0, 1, len(tasks)))
        task_color_map = {task: colors[i] for i, task in enumerate(tasks)}
        
        current_y = 0
        agent_slots = {}  # 记录每个代理的插槽使用情况: {agent_name: {slot_index: [(start_time, end_time)]}}
        agent_y_start = {}  # 记录每个代理在甘特图上的起始Y坐标
        
        # 为每个代理分配Y坐标位置和初始化插槽
        for agent in self.simulation.agents:
            agent_y_start[agent.name] = current_y
            agent_slots[agent.name] = {slot: [] for slot in range(agent.capacity)}
            
            for slot in range(agent.capacity):
                y_label = f"{agent.name} (容量槽 {slot+1})"
                y_labels.append(y_label)
                y_positions.append(current_y)
                y_ticks.append(current_y + 0.5)
                current_y += 1
        
        # 收集任务执行数据
        max_time = max(self.agent_positions.keys())
        task_execution_data = {}
        
        # 第一步：收集每个任务的拾取和完成时间以及执行代理
        for task_name, status in self.task_status.items():
            if status["picked"] is not None:
                picked_time = status["picked"]
                completed_time = status["completed"] if status["completed"] is not None else max_time
                
                # 找出执行该任务的代理
                executing_agent = None
                for t in range(picked_time, completed_time + 1):
                    if t in self.picked_tasks:
                        for agent_name, tasks in self.picked_tasks[t].items():
                            if task_name in tasks:
                                executing_agent = agent_name
                                break
                        if executing_agent:
                            break
                
                if executing_agent:
                    task_execution_data[task_name] = {
                        "picked_time": picked_time,
                        "completed_time": completed_time,
                        "agent": executing_agent,
                        "duration": completed_time - picked_time
                    }
        
        # 第二步：为每个任务分配合适的插槽，避免时间重叠
        # 按任务持续时间从长到短排序，优先分配长任务
        sorted_tasks = sorted(task_execution_data.items(), key=lambda x: x[1]["duration"], reverse=True)
        
        for task_name, data in sorted_tasks:
            agent_name = data["agent"]
            if agent_name not in agent_slots:
                continue
                
            start_time = data["picked_time"]
            end_time = data["completed_time"]
            
            # 为任务找到合适的插槽
            assigned_slot = None
            for slot_index in range(len(agent_slots[agent_name])):
                slot_tasks = agent_slots[agent_name][slot_index]
                
                # 检查是否有时间重叠
                can_assign = True
                for task_start, task_end in slot_tasks:
                    # 检查两个时间范围是否重叠
                    if not (end_time <= task_start or start_time >= task_end):
                        can_assign = False
                        break
                
                if can_assign:
                    assigned_slot = slot_index
                    agent_slots[agent_name][slot_index].append((start_time, end_time))
                    # 更新任务数据，记录分配的插槽
                    task_execution_data[task_name]["slot"] = slot_index
                    break
            
            # 如果所有插槽都有重叠，则分配到第一个插槽（应该不会发生，因为容量限制）
            if assigned_slot is None:
                print(f"警告: 任务 {task_name} 无法找到可用插槽，可能超出容量限制")
                task_execution_data[task_name]["slot"] = 0
                agent_slots[agent_name][0].append((start_time, end_time))
        
        # 绘制甘特图的任务条
        for task_name, data in task_execution_data.items():
            # 计算任务条的Y位置
            if data["agent"] not in agent_y_start:
                continue
                
            agent_y = agent_y_start[data["agent"]]
            slot = data.get("slot", 0)  # 获取分配的插槽
            y_position = agent_y + slot  # 每个插槽对应一个Y坐标位置
            
            # 绘制任务条
            start_time = data["picked_time"]
            end_time = data["completed_time"]
            width = end_time - start_time
            
            # 绘制矩形表示任务
            rect = mpatches.Rectangle(
                (start_time, y_position), width, 1, 
                facecolor=task_color_map[task_name], 
                alpha=0.8,
                edgecolor='black',
                linewidth=1
            )
            ax.add_patch(rect)
            
            # 添加任务名称标签
            if width > 2:  # 只有当任务条足够宽时才添加标签
                ax.text(
                    start_time + width/2, 
                    y_position + 0.5, 
                    task_name, 
                    ha='center', 
                    va='center',
                    color='black',
                    fontsize=8
                )
        
        # 设置图表属性
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_ylabel("代理 (容量槽)")
        ax.set_xlabel("时间步")
        ax.set_title("任务执行甘特图")
        
        # 设置X轴范围
        ax.set_xlim(0, max_time)
        ax.set_ylim(0, current_y)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 添加最后一个任务完成时间的标记
        last_task_completion = max([status["completed"] for _, status in self.task_status.items() if status["completed"] is not None], default=0)
        if last_task_completion > 0:
            ax.axvline(x=last_task_completion, color='red', linestyle='--', linewidth=2)
            ax.text(
                last_task_completion + 0.5, 
                current_y * 0.98, 
                f"所有任务完成: t={last_task_completion}",
                ha='left',
                va='top',
                color='red',
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5')
            )
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(os.path.join(self.plots_dir, "task_gantt_chart.png"), dpi=300)
        plt.close()
    
    def plot_tasks_lifecycle_gantt(self):
        """
        绘制任务生命周期甘特图
        - 横轴为时间步
        - 纵轴为各个任务
        - 从任务被发布的时间开始绘制长条，到任务完成时结束
        - 在任务被拾取时用不同颜色区分未被拾取和已被拾取的状态
        """
        import matplotlib.patches as mpatches
        
        # 过滤出有完整生命周期数据的任务（已发布且已完成）
        completed_tasks = {
            task_name: status for task_name, status in self.task_status.items()
            if status["released"] is not None and status["completed"] is not None
        }
        
        if not completed_tasks:
            print("没有已完成的任务，无法生成任务生命周期图")
            return
        
        # 按发布时间排序任务
        sorted_tasks = sorted(completed_tasks.items(), key=lambda x: x[1]["released"])
        num_tasks = len(sorted_tasks)
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(15, max(6, num_tasks * 0.5)))
        
        # 设置Y轴标签（任务名称）
        task_names = [task[0] for task in sorted_tasks]
        y_positions = list(range(len(task_names)))
        
        # 颜色设置
        waiting_color = 'lightgray'    # 等待被拾取的颜色
        execution_color = 'royalblue'  # 被拾取后执行中的颜色
        
        # 绘制任务条
        for i, (task_name, status) in enumerate(sorted_tasks):
            # 任务发布到拾取的等待阶段
            released_time = status["released"]
            picked_time = status["picked"]
            completed_time = status["completed"]
            
            # 绘制等待阶段（发布到拾取）
            if picked_time and picked_time > released_time:
                waiting_width = picked_time - released_time
                waiting_rect = mpatches.Rectangle(
                    (released_time, i - 0.4), 
                    waiting_width, 
                    0.8,
                    facecolor=waiting_color,
                    alpha=0.7,
                    edgecolor='black',
                    linewidth=1
                )
                ax.add_patch(waiting_rect)
            
            # 绘制执行阶段（拾取到完成）
            if picked_time and completed_time and completed_time > picked_time:
                execution_width = completed_time - picked_time
                execution_rect = mpatches.Rectangle(
                                    (picked_time, i - 0.4), 
                                    execution_width, 
                                    0.8,
                                    facecolor=execution_color,
                                    alpha=0.8,
                                    edgecolor='black',
                                    linewidth=1
                                )
                ax.add_patch(execution_rect)
                
                # 在任务条上添加时间标注
                total_time = completed_time - released_time
                ax.text(
                    (released_time + completed_time) / 2,
                    i,
                    f"{total_time} 步",
                    ha='center',
                    va='center',
                    color='black',
                    fontsize=8,
                    fontweight='bold'
                )
        
        # 添加图例
        waiting_patch = mpatches.Patch(color=waiting_color, label='等待被拾取')
        execution_patch = mpatches.Patch(color=execution_color, label='执行中')
        ax.legend(handles=[waiting_patch, execution_patch], loc='upper right')
        
        # 设置轴标签和标题
        ax.set_yticks(y_positions)
        ax.set_yticklabels(task_names)
        ax.set_xlabel('时间步')
        ax.set_ylabel('任务')
        ax.set_title('任务生命周期甘特图')
        
        # 设置Y轴范围
        ax.set_ylim(-0.8, len(y_positions) - 0.2)
        
        # 设置X轴范围
        max_time = max(self.agent_positions.keys())
        ax.set_xlim(0, max_time)
        
        # 添加网格
        ax.grid(True, axis='x', alpha=0.3)
        
        # 找出最后一个任务完成的时间
        if completed_tasks:
            last_task_completion_time = max(status["completed"] for _, status in completed_tasks.items())
            
            # 在最后一个任务完成的时间处添加红色竖虚线
            ax.axvline(x=last_task_completion_time, color='red', linestyle='--', linewidth=2)
            
            # 添加文本标注
            ax.text(
                last_task_completion_time + 0.5,
                len(sorted_tasks) / 2,
                f"所有任务完成: t={last_task_completion_time}",
                ha='left',
                va='center',
                color='red',
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5')
            )
        
        # 增加子图的上下边距
        plt.tight_layout(pad=2.0)
        
        # 保存图表
        plt.savefig(os.path.join(self.plots_dir, "tasks_lifecycle_gantt.png"), dpi=300)
        plt.close()
        
        print("已生成任务生命周期甘特图")
    
    def plot_frontier_evolution(self):
        """绘制边界点随时间的变化"""
        max_time = max(self.agent_positions.keys())
        frontier_counts = []
        time_steps = sorted(self.frontier_history.keys())
        
        for t in time_steps:
            frontier_counts.append(len(self.frontier_history[t]))
            
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, frontier_counts, '-o', linewidth=2)
        plt.title('边界点数量变化')
        plt.xlabel('时间步')
        plt.ylabel('边界点数量')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "frontier_evolution.png"), dpi=300)
        plt.close()
    
    def _draw_complete_grid(self, ax, width, height):
        """
        在给定的坐标轴上绘制完整的网格
        :param ax: matplotlib坐标轴对象
        :param width: 地图宽度
        :param height: 地图高度
        """
        # 设置坐标轴范围，确保显示整个网格
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)
        
        # 绘制网格线
        for x in range(width):
            ax.axvline(x=x - 0.5, color='gray', linestyle='-', linewidth=0.8, alpha=0.3)
        for y in range(height):
            ax.axhline(y=y - 0.5, color='gray', linestyle='-', linewidth=0.8, alpha=0.3)
        
        # 设置网格刻度
        ax.set_xticks(range(width))
        ax.set_yticks(range(height))
        
        # 显示刻度标签
        ax.set_xticklabels(range(width))
        ax.set_yticklabels(range(height))
        
        # 设置网格线总是显示
        ax.grid(False)  # 关闭默认网格
    
    def generate_visualizations(self):
        """生成所有可视化图表"""
        print("正在生成可视化图表...")
        
        try:
            self.plot_agent_paths()
            print("已生成代理路径图")
            
            self.plot_task_completion_over_time()
            print("已生成任务完成情况图")
            
            self.plot_communication_graphs()
            print("已生成通信图快照")
            
            self.plot_tasks_gantt_chart()
            print("已生成任务甘特图")
            
            self.plot_tasks_lifecycle_gantt()
            print("已生成任务生命周期甘特图")
            
            self.plot_frontier_evolution()
            print("已生成边界点变化图")
            
            # 添加新的可视化函数
            self.plot_exploration_progress()
            print("已生成地图探索进度图")
            
            ts = time.time()
            self.create_animation()
            print(f"已生成地图探索动画, 耗时{time.time() - ts:.2f}秒")
        except Exception as e:
            print(f"生成可视化图表时出错: {e}")
            import traceback
            traceback.print_exc()
        
        print("可视化图表生成完成")

    def load_data(self, session_dir=r"D:\berker\graduation_design\code\POMAPD\logs"):
        """从指定会话目录加载保存的数据"""
        if session_dir:
            self.session_dir = session_dir
        
        data_dir = os.path.join(self.session_dir, "data")
        if not os.path.exists(data_dir):
            raise ValueError(f"数据目录不存在: {data_dir}")
        
        # 加载代理路径数据
        with open(os.path.join(data_dir, "agent_paths.json"), "r") as f:
            self.agent_paths = json.load(f)
        
        # 加载任务状态数据
        with open(os.path.join(data_dir, "task_status.json"), "r") as f:
            self.task_status = json.load(f)
        
        # 加载通信图数据
        with open(os.path.join(data_dir, "communication_graphs.json"), "r") as f:
            self.communication_graphs = json.load(f)
        
        # 加载代理任务分配
        with open(os.path.join(data_dir, "agent_tasks.json"), "r") as f:
            self.agent_tasks = json.load(f)
        
        # 加载已拾取任务
        with open(os.path.join(data_dir, "picked_tasks.json"), "r") as f:
            self.picked_tasks = json.load(f)
        
        # 加载性能指标
        with open(os.path.join(data_dir, "performance_metrics.json"), "r") as f:
            self.performance_metrics = json.load(f)
        
        # 加载地图信息
        with open(os.path.join(data_dir, "map_info.json"), "r") as f:
            self.map_info = json.load(f)
            
        # 加载已知区域历史
        with open(os.path.join(data_dir, "known_area_history.json"), "r") as f:
            self.known_area_history = json.load(f)
        
        # 加载前沿点位置
        with open(os.path.join(data_dir, "frontier_positions.json"), "r") as f:
            frontier_data = json.load(f)
            # 转换键为整数
            self.frontier_positions = {int(t): [tuple(pos) for pos in positions] 
                                    for t, positions in frontier_data.items()}
        
        # 加载中心点位置
        with open(os.path.join(data_dir, "centroid_positions.json"), "r") as f:
            centroid_data = json.load(f)
            # 处理中心点数据
            self.centroid_positions = {}
            for t, centroids in centroid_data.items():
                t = int(t)
                self.centroid_positions[t] = {}
                for pos_str, data in centroids.items():
                    # 将字符串键转回元组
                    pos = tuple(map(int, pos_str.strip('()').split(',')))
                    self.centroid_positions[t][pos] = data
        
        # 加载代理位置数据
        with open(os.path.join(data_dir, "agent_positions.json"), "r") as f:
            position_data = json.load(f)
            # 转换键为整数
            self.agent_positions = {int(t): {agent: tuple(pos) for agent, pos in positions.items()} 
                                for t, positions in position_data.items()}
        
        print("数据加载完成")

if __name__ == "__main__":
    print("Logger module imported. To use, create a Logger instance with your simulation object.")