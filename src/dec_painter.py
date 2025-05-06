"""
提供分布式仿真过程中的可视化功能，支持动态更新多个智能体的地图和位置
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import time
import numpy as np
from copy import deepcopy
from typing import TYPE_CHECKING, List, Dict, Tuple, Set, Optional, Union
import os
import colorsys
import math

if TYPE_CHECKING:
    from .agent import Agent, Base
    from .map import Map
    from .simulation import Simulation
    from .Token import Token

class PainterDec:
    """
    用于可视化分布式仿真过程中的地图和智能体状态
    支持实时更新多个智能体的地图视图和位置
    """
    def __init__(self, figsize=None, dpi=150, max_agents_to_display=2):
        """
        初始化PainterDec类
        
        Parameters:
            figsize: 图像大小，默认为根据地图大小自动调整
            dpi: 图像分辨率
            max_agents_to_display: 最多同时显示几个智能体的地图视图，默认为2
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
        plt.rcParams['axes.unicode_minus'] = False    # 显示负号
        
        self.figsize = figsize
        self.dpi = dpi
        self.fig = None
        self.axes = []  # 存储每个智能体的地图子图
        self.task_ax = None  # 任务统计子图
        
        # 最多显示的智能体数量
        self.max_agents_to_display = max_agents_to_display
        
        # 存储每个智能体视图的信息
        self.agent_views = {}  # {agent_name: {ax, scatter, time_text, info_text, laser_rect}}
        
        # 任务统计数据
        self.time_steps = []
        self.discovered_tasks = []
        self.completed_tasks = []
        
        # 记录通信图
        self.comm_graph_patches = {}  # 按智能体名称分类存储通信线
        
        # 初始化颜色映射
        self.agent_color_map = {}
        self.predefined_colors = [
            '#FF0000',  # 红
            '#00FF00',  # 绿
            '#0000FF',  # 蓝
            '#FFFF00',  # 黄
            '#FF00FF',  # 洋红
            '#00FFFF',  # 青
            '#FF8000',  # 橙
            '#8000FF',  # 紫
            '#0080FF',  # 天蓝
            '#FF0080',  # 粉
            '#80FF00',  # 黄绿
            '#00FF80',  # 青绿
        ]
        self.next_color_idx = 0
        
        # 保存所有智能体的引用
        self.all_agents = {}

    def setup(self, agents: List["Agent"], base: "Base", real_map: "Map", 
              agents_to_display: Optional[List[str]] = None):
        """
        设置初始图形
        
        Parameters:
            agents: 智能体列表
            base: 基站
            real_map: 真实地图(用于获取地图尺寸)
            agents_to_display: 要显示地图的智能体名称列表，默认显示前max_agents_to_display个
        """
        # 保存所有智能体的引用，便于后续访问
        self.all_agents = {agent.name: agent for agent in agents}
        self.base = base
        self.real_map = real_map
        
        # 确定要显示哪些智能体的地图
        if agents_to_display is None:
            # 默认显示前max_agents_to_display个智能体
            agents_to_display = [agent.name for agent in agents[:self.max_agents_to_display]]
        else:
            # 确保不超过最大数量限制
            agents_to_display = agents_to_display[:self.max_agents_to_display]
        
        # 保存要显示的智能体列表
        self.displayed_agents = [agent for agent in agents if agent.name in agents_to_display]
        
        # 如果未指定figsize，则根据地图尺寸和要显示的地图数量自动调整
        if self.figsize is None:
            map_aspect_ratio = real_map.height / real_map.width
            num_maps = len(self.displayed_agents)
            
            # 调整布局和大小
            if num_maps == 1:
                # 单地图时，使用更大的尺寸
                fig_width = min(18, max(10, real_map.width/3))
                fig_height = fig_width * map_aspect_ratio * 1.3  # 留出任务统计空间
            else:
                # 多地图时，减小单个地图的尺寸
                fig_width = min(20, max(12, real_map.width/2.5))
                # 高度取决于地图的布局方式
                rows = math.ceil(num_maps / 2)  # 每行最多2个地图
                fig_height = fig_width * map_aspect_ratio * rows * 0.6 + fig_width * 0.3  # 加上任务统计空间
            
            self.figsize = (fig_width, fig_height)
        
        # 保存初始token引用，用于任务统计
        self.initial_token = agents[0].token if agents else None
        
        # 创建图形
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # 创建网格布局：多个地图视图 + 一个任务统计视图
        num_maps = len(self.displayed_agents)
        
        # 确定布局：每行最多2个地图
        if num_maps == 1:
            # 单地图时用1行2列的布局，地图占第一列，任务统计占第二列
            gs = self.fig.add_gridspec(1, 2, width_ratios=[2, 1])
            self.axes = [self.fig.add_subplot(gs[0, 0])]
            self.task_ax = self.fig.add_subplot(gs[0, 1])
        else:
            # 多地图时，地图放在上方，任务统计放在下方
            rows = math.ceil(num_maps / 2)  # 计算需要的行数
            gs = self.fig.add_gridspec(rows + 1, 2, height_ratios=[1] * rows + [0.4])
            
            # 创建地图子图
            self.axes = []
            for i in range(num_maps):
                row = i // 2
                col = i % 2
                self.axes.append(self.fig.add_subplot(gs[row, col]))
            
            # 任务统计子图跨两列
            self.task_ax = self.fig.add_subplot(gs[rows, :])
        
        # 设置任务统计子图
        self.task_ax.set_title('任务统计', fontsize=14)
        self.task_ax.set_xlabel('时间步', fontsize=12)
        self.task_ax.set_ylabel('任务数量', fontsize=12)
        self.task_ax.grid(True, linestyle='--', alpha=0.7)
        
        # 初始化每个智能体的视图
        for i, agent in enumerate(self.displayed_agents):
            ax = self.axes[i]
            ax.set_title(f'智能体 {agent.name} 视图', fontsize=14, fontweight='bold')
            
            # 设置坐标轴范围
            ax.set_xlim(-0.5, real_map.width - 0.5)
            ax.set_ylim(-0.5, real_map.height - 0.5)
            ax.set_aspect('equal')  # 保持网格单元为正方形
            
            # 设置刻度间隔
            tick_spacing = max(1, min(5, real_map.width // 10))
            x_ticks = range(0, real_map.width, tick_spacing)
            y_ticks = range(0, real_map.height, tick_spacing)
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.set_xlabel('X坐标', fontsize=10)
            ax.set_ylabel('Y坐标', fontsize=10)
            
            # 绘制网格线
            for x in range(real_map.width + 1):
                ax.axvline(x=x - 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            for y in range(real_map.height + 1):
                ax.axhline(y=y - 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # 在右上角添加时间步显示
            time_text = ax.text(0.98, 0.98, 'Time: -1', transform=ax.transAxes,
                               fontsize=11, ha='right', va='top',
                               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
            
            # 在左上角添加信息显示
            info_text = f"地图大小: {real_map.width}×{real_map.height}\n"
            info_text += f"已知区域: {agent.map.known_area}/{real_map.total_area} ({agent.map.known_area/real_map.total_area*100:.1f}%)"
            
            info_text_obj = ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                                 fontsize=9, verticalalignment='top', 
                                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
            
            # 初始化智能体位置和探照范围
            agent_color = self._get_agent_color(agent.name)
            
            # 创建智能体的探照范围
            laser_half_width = agent.laser_range
            laser_rect = patches.Rectangle(
                (agent.pos[0] - laser_half_width - 0.5, agent.pos[1] - laser_half_width - 0.5),
                2 * laser_half_width + 1, 2 * laser_half_width + 1,
                linewidth=1.5, edgecolor=agent_color, facecolor=agent_color, alpha=0.1,
                linestyle='--', zorder=10
            )
            laser_rect._is_agent_laser = True
            ax.add_patch(laser_rect)
            
            # 创建智能体的散点标记
            scatter = ax.scatter(agent.pos[0], agent.pos[1], 
                               c=agent_color, marker='o', s=100, zorder=10,
                               edgecolors='black', linewidths=1.5)
            
            # 添加智能体名称标签
            agent_label = ax.text(agent.pos[0], agent.pos[1] - 0.3, agent.name, 
                               fontsize=9, ha='center', va='top', 
                               color='black', fontweight='bold', zorder=10)
            
            # 保存该智能体视图的所有元素
            self.agent_views[agent.name] = {
                'ax': ax,
                'scatter': scatter,
                'time_text': time_text,
                'info_text': info_text_obj,
                'laser_rect': laser_rect,
                'agent_label': agent_label,
                'frontier_markers': [],
                'centroid_markers': [],
                'centroid_boxes': [],
                'centroid_texts': [],
                'task_markers': [],
                'task_texts': [],
                'comm_lines': [],
                'other_agents': {}  # 存储在该视图中显示的其他智能体
            }
            
            # 初始化基站
            self._init_base(ax, base)
            
            # 初始化地图
            self._draw_agent_map(agent, t=-1)
            
            # 显示其他智能体的位置
            self._draw_other_agents(agent, agents)
            
            # 创建图例
            self._create_legend(ax, agent.map)
        
        # 初始化任务统计数据
        self.time_steps = [-1]
        self.discovered_tasks = [0]
        self.completed_tasks = [0]
        
        # 绘制初始任务统计图
        if self.initial_token:
            self._update_task_stats(-1, self.initial_token)
        
        # 紧凑布局
        plt.tight_layout()
        plt.ion()  # 打开交互模式
        plt.show(block=False)
    
    def _init_base(self, ax, base: "Base"):
        """在指定子图上初始化基站显示"""
        for pos in base.positions:
            ax.add_patch(plt.Rectangle(
                (pos[0] - 0.5, pos[1] - 0.5), 1, 1, 
                facecolor='royalblue', alpha=0.8, zorder=5
            ))
        
        # 在第一个基站位置显示标签
        if base.positions:
            ax.text(base.positions[0][0], base.positions[0][1], 'Base', 
                  fontsize=8, ha='center', va='center', 
                  color='white', fontweight='bold', zorder=10)

    def _draw_other_agents(self, current_agent: "Agent", all_agents: List["Agent"]):
        """在当前智能体视图中绘制其他智能体的位置"""
        if current_agent.name not in self.agent_views:
            return
        
        view_data = self.agent_views[current_agent.name]
        ax = view_data['ax']
        
        # 清除已存在的其他智能体显示
        for other_name, other_data in view_data['other_agents'].items():
            if other_data['scatter'] in ax.collections:
                other_data['scatter'].remove()
            if other_data['label'] in ax.texts:
                other_data['label'].remove()
        view_data['other_agents'] = {}
        
        # 绘制其他智能体
        for agent in all_agents:
            if agent.name != current_agent.name:
                # 为其他智能体使用不同的颜色
                color = self._get_agent_color(agent.name)
                
                # 创建散点标记
                scatter = ax.scatter(agent.pos[0], agent.pos[1], 
                                   c=color, marker='o', s=80, zorder=9,
                                   edgecolors='black', linewidths=1)
                
                # 添加标签
                label_text = agent.name
                if hasattr(agent, 'picked_tasks') and agent.picked_tasks:
                    label_text = f"{agent.name}\n[{','.join(agent.picked_tasks)}]"
                
                label = ax.text(agent.pos[0], agent.pos[1] - 0.3, label_text, 
                              fontsize=8, ha='center', va='top', 
                              color='black', fontweight='bold', zorder=9)
                
                # 保存引用，便于后续更新
                view_data['other_agents'][agent.name] = {
                    'scatter': scatter,
                    'label': label
                }

    def _adapt_map_for_display(self, local_map: "Map", token: "Token" = None):
        """为显示准备地图对象，确保它有必要的属性"""
        # 如果提供了token且地图没有token属性，则添加
        if token is not None and not hasattr(local_map, 'token'):
            local_map.token = token
        
        # 确保地图有tasks_seen属性
        if not hasattr(local_map, 'tasks_seen') or local_map.tasks_seen is None:
            # 尝试从token获取任务信息
            if hasattr(local_map, 'token') and hasattr(local_map.token, '_tasks'):
                # 创建任务列表，基于已知的任务
                tasks_seen = []
                for task in local_map.token._tasks:
                    # 检查任务是否在unassigned_tasks、c_tasks或已分配的任务中
                    if isinstance(task, dict) and 'task_name' in task:
                        task_name = task['task_name']
                        token = local_map.token
                        if (task_name in token.unassigned_tasks or 
                            task_name in token.c_tasks or
                            any(task_name in tasks for tasks in token.agents_to_tasks.values())):
                            tasks_seen.append(task)
                
                local_map.tasks_seen = tasks_seen
        
        return local_map

    def update(self, t: int, simulation: "Simulation"):
        """
        更新可视化：更新每个显示的智能体的地图和位置
        
        Parameters:
            t: 当前时间步
            simulation: 仿真对象，包含所有需要的状态
        """
        # 更新所有智能体的引用
        self.all_agents = {agent.name: agent for agent in simulation.agents}
        
        # 更新每个显示的智能体的地图视图
        for agent_name, view_data in self.agent_views.items():
            if agent_name in self.all_agents:
                agent = self.all_agents[agent_name]
                # 更新时间步显示
                view_data['time_text'].set_text(f'Time: {t}')
                
                # 绘制该智能体的地图和位置
                self._draw_agent_map(agent, t)
                
                # 更新该智能体视图中的通信图
                agent_comm_graphs = self._find_agent_comm_graphs(agent, simulation.comm_graphs)
                self._draw_agent_comm_graph(agent, agent_comm_graphs, simulation.agents, simulation.base)
                
                # 显示其他所有智能体的位置
                self._update_other_agents_in_view(agent, simulation.agents)
        
        # 更新任务统计
        if simulation.comm_graphs and len(simulation.comm_graphs) > 0 and len(simulation.comm_graphs[0]) > 0:
            display_token = simulation.comm_graphs[0][0].token
            self._update_task_stats(t, display_token)
        
        # 刷新图形
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except (ValueError, RuntimeError) as e:
            print(f"图形更新出错（可能是窗口已关闭）: {e}")
    
    def _find_agent_comm_graphs(self, agent: "Agent", comm_graphs: List[List["Agent"]]):
        """查找包含指定智能体的通信图"""
        agent_comm_graphs = []
        for comm_graph in comm_graphs:
            agent_names = [a.name for a in comm_graph]
            if agent.name in agent_names:
                agent_comm_graphs.append(comm_graph)
        return agent_comm_graphs
    
    def _draw_agent_comm_graph(self, agent: "Agent", comm_graphs: List[List["Agent"]], 
                               all_agents: List["Agent"], base: "Base"):
        """在指定智能体的视图中绘制通信图"""
        view_data = self.agent_views[agent.name]
        ax = view_data['ax']
        
        # 清除之前的通信图线条
        for line in view_data['comm_lines']:
            if line in ax.lines:
                line.remove()
        view_data['comm_lines'] = []
        
        # 为不同的通信图使用不同的颜色
        comm_line_colors = [
            (1.0, 0.0, 0.0, 0.6),  # 红色半透明
            (0.0, 1.0, 0.0, 0.6),  # 绿色半透明
            (0.0, 0.0, 1.0, 0.6),  # 蓝色半透明
            (1.0, 1.0, 0.0, 0.6),  # 黄色半透明
            (1.0, 0.0, 1.0, 0.6),  # 洋红色半透明
            (0.0, 1.0, 1.0, 0.6)   # 青色半透明
        ]
        
        # 绘制通信连接线
        for i, comm_graph in enumerate(comm_graphs):
            color_idx = i % len(comm_line_colors)
            line_color = comm_line_colors[color_idx]
            
            for agent1 in comm_graph:
                if agent1.name == agent.name:  # 我们只关心当前视图的智能体的连接
                    for agent2 in comm_graph:
                        if agent1 != agent2:
                            # 如果对方是基站
                            if agent2.name == 'base':
                                # 找到最近的基站位置
                                min_dist = float('inf')
                                closest_pos = None
                                for pos in agent2.positions:
                                    dist = max(abs(agent1.pos[0] - pos[0]), abs(agent1.pos[1] - pos[1]))
                                    if dist < min_dist:
                                        min_dist = dist
                                        closest_pos = pos
                                
                                if closest_pos and min_dist <= agent1.comm_range:
                                    line = plt.Line2D([agent1.pos[0], closest_pos[0]], 
                                                [agent1.pos[1], closest_pos[1]],
                                                color=line_color,
                                                linestyle='--', linewidth=1.5, zorder=3)
                                    ax.add_line(line)
                                    view_data['comm_lines'].append(line)
                            else:
                                # 两个普通智能体之间
                                dist = max(abs(agent1.pos[0] - agent2.pos[0]), 
                                         abs(agent1.pos[1] - agent2.pos[1]))
                                if dist <= agent1.comm_range:
                                    line = plt.Line2D([agent1.pos[0], agent2.pos[0]], 
                                                [agent1.pos[1], agent2.pos[1]],
                                                color=line_color,
                                                linestyle='--', linewidth=1.5, zorder=3)
                                    ax.add_line(line)
                                    view_data['comm_lines'].append(line)
    
    def _update_other_agents_in_view(self, agent: "Agent", all_agents: List["Agent"]):
        """更新某个智能体视图中所有其他智能体的位置"""
        if agent.name not in self.agent_views:
            return
        
        view_data = self.agent_views[agent.name]
        ax = view_data['ax']
        
        # 更新所有其他智能体的位置
        for other_agent in all_agents:
            if other_agent.name != agent.name and other_agent.name != 'base':
                if other_agent.name in view_data['other_agents']:
                    # 更新现有智能体的位置
                    other_data = view_data['other_agents'][other_agent.name]
                    other_data['scatter'].set_offsets([other_agent.pos[0], other_agent.pos[1]])
                    
                    # 更新标签位置和内容
                    text = other_agent.name
                    if hasattr(other_agent, 'picked_tasks') and other_agent.picked_tasks:
                        text = f"{text}\n[{','.join(other_agent.picked_tasks)}]"
                    
                    other_data['label'].set_text(text)
                    other_data['label'].set_position((other_agent.pos[0], other_agent.pos[1] - 0.3))
                else:
                    # 添加新的智能体
                    color = self._get_agent_color(other_agent.name)
                    scatter = ax.scatter(other_agent.pos[0], other_agent.pos[1], 
                                      c=color, marker='o', s=80, zorder=9,
                                      edgecolors='black', linewidths=1)
                    
                    # 添加标签
                    text = other_agent.name
                    if hasattr(other_agent, 'picked_tasks') and other_agent.picked_tasks:
                        text = f"{text}\n[{','.join(other_agent.picked_tasks)}]"
                    
                    label = ax.text(other_agent.pos[0], other_agent.pos[1] - 0.3, text, 
                                  fontsize=8, ha='center', va='top', 
                                  color='black', fontweight='bold', zorder=9)
                    
                    view_data['other_agents'][other_agent.name] = {
                        'scatter': scatter,
                        'label': label
                    }
        
        # 检查是否有需要移除的智能体（不再存在的）
        existing_agents = {agent.name for agent in all_agents}
        to_remove = []
        
        for other_name in view_data['other_agents']:
            if other_name not in existing_agents or other_name == 'base':
                # 移除不再存在的智能体或基站
                other_data = view_data['other_agents'][other_name]
                if other_data['scatter'] in ax.collections:
                    other_data['scatter'].remove()
                if other_data['label'] in ax.texts:
                    other_data['label'].remove()
                to_remove.append(other_name)
        
        # 从字典中删除已移除的智能体
        for name in to_remove:
            del view_data['other_agents'][name]

    def _update_task_stats(self, t: int, token: "Token"):
        """更新任务统计图"""
        # 确保token有任务信息
        if not hasattr(token, '_tasks'):
            print("警告: Token没有_tasks属性")
            return
        
        # 获取已发现和已完成的任务数量
        discovered_count = len(getattr(token, 'tasks_seen', []))
        if discovered_count == 0:
            # 如果没有tasks_seen属性，则尝试估计已发现的任务数量
            assigned_tasks = set()
            for tasks in token.agents_to_tasks.values():
                assigned_tasks.update(tasks)
            discovered_count = len(token.unassigned_tasks) + len(assigned_tasks) + len(token.c_tasks)
        
        completed_count = len(token.c_tasks)
        
        # 更新数据
        if t not in self.time_steps:
            self.time_steps.append(t)
            self.discovered_tasks.append(discovered_count)
            self.completed_tasks.append(completed_count)
        else:
            # 更新现有数据点
            idx = self.time_steps.index(t)
            self.discovered_tasks[idx] = discovered_count
            self.completed_tasks[idx] = completed_count
        
        # 清除当前任务统计图
        self.task_ax.clear()
        
        # 重新绘制统计图
        self.task_ax.set_title('任务统计', fontsize=14)
        self.task_ax.set_xlabel('时间步', fontsize=12)
        self.task_ax.set_ylabel('任务数量', fontsize=12)
        
        # 绘制折线图
        self.task_ax.plot(self.time_steps, self.discovered_tasks, 'b-', marker='o', label='已发现任务')
        self.task_ax.plot(self.time_steps, self.completed_tasks, 'r-', marker='s', label='已完成任务')
        
        # 添加网格和图例
        self.task_ax.grid(True, linestyle='--', alpha=0.7)
        self.task_ax.legend(loc='upper left')
        
        # 设置x轴刻度，确保为整数
        self.task_ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # 显示任务总数
        total_tasks = len(token._tasks)
        self.task_ax.text(0.98, 0.02, f'总任务数: {total_tasks}', 
                        transform=self.task_ax.transAxes, ha='right', va='bottom',
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

    def _draw_agent_map(self, agent: "Agent", t: int):
        """绘制指定智能体的地图和状态"""
        if agent.name not in self.agent_views:
            return  # 只处理设置了显示的智能体
        
        view_data = self.agent_views[agent.name]
        ax = view_data['ax']
        local_map = agent.map
        
        # 适配地图对象以确保它有所需的属性
        local_map = self._adapt_map_for_display(local_map, agent.token)
        
        # 更新智能体自身的位置和探照范围
        # 更新散点位置
        view_data['scatter'].set_offsets([agent.pos[0], agent.pos[1]])
        
        # 更新探照范围
        laser_rect = view_data['laser_rect']
        if laser_rect in ax.patches:
            laser_rect.remove()
        
        # 获取智能体颜色
        agent_color = self._get_agent_color(agent.name)
        
        # 创建新的探照范围矩形
        laser_half_width = agent.laser_range
        new_laser_rect = patches.Rectangle(
            (agent.pos[0] - laser_half_width - 0.5, agent.pos[1] - laser_half_width - 0.5),
            2 * laser_half_width + 1, 2 * laser_half_width + 1,
            linewidth=1.5, edgecolor=agent_color, facecolor=agent_color, alpha=0.1,
            linestyle='--', zorder=10
        )
        new_laser_rect._is_agent_laser = True
        ax.add_patch(new_laser_rect)
        view_data['laser_rect'] = new_laser_rect
        
        # 更新智能体标签
        view_data['agent_label'].set_position((agent.pos[0], agent.pos[1] - 0.3))
        if hasattr(agent, 'picked_tasks') and agent.picked_tasks:
            task_text = f"{agent.name}\n[{','.join(agent.picked_tasks)}]"
            view_data['agent_label'].set_text(task_text)
        else:
            view_data['agent_label'].set_text(agent.name)
        
        # 清除之前的地图元素（保留智能体的激光范围矩形）
        for patch in ax.patches[:]:
            if not hasattr(patch, '_is_agent_laser') and patch is not new_laser_rect:
                patch.remove()
        
        # 清除所有动态元素
        self._clear_agent_dynamic_elements(agent.name)
        
        # 收集不同类型的单元格
        unknown_cells = []
        free_cells = []
        obstacle_cells = []
        base_cells = []
        
        for y in range(local_map.height):
            for x in range(local_map.width):
                cell_value = local_map.map[y][x]
                if cell_value == -1:        # 未知区域
                    unknown_cells.append((x, y))
                elif cell_value == 0:       # 自由空间
                    free_cells.append((x, y))
                elif cell_value == 1:       # 障碍物
                    obstacle_cells.append((x, y))
                elif cell_value == 2:       # 基地点
                    base_cells.append((x, y))
        
        # 绘制自由空间
        for x, y in free_cells:
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                facecolor='white', 
                                edgecolor='lightgray', linewidth=0.5)
            ax.add_patch(rect)
        
        # 绘制未知区域
        for x, y in unknown_cells:
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                facecolor='gray', alpha=0.5)
            ax.add_patch(rect)
        
        # 绘制障碍物
        for x, y in obstacle_cells:
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                facecolor='black')
            ax.add_patch(rect)
        
        # 绘制基地点 - 使用蓝色
        for x, y in base_cells:
            rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                facecolor='royalblue', alpha=0.8)
            ax.add_patch(rect)
        
        # 绘制任务、前沿点和中心点
        self._update_agent_tasks_and_frontiers(agent.name, local_map)
        
        # 更新信息文本
        info_text = f"地图大小: {local_map.width}×{local_map.height}\n"
        info_text += f"已知区域: {local_map.known_area}/{local_map.total_area} ({local_map.known_area/local_map.total_area*100:.1f}%)\n"
        info_text += f"前沿点数: {len(local_map.frontiers)}\n"
        info_text += f"中心点数: {len(local_map.centroids)}"
        view_data['info_text'].set_text(info_text)

    def _update_agent_tasks_and_frontiers(self, agent_name: str, local_map: "Map"):
        """更新指定智能体视图中的任务、前沿点和中心点"""
        view_data = self.agent_views[agent_name]
        ax = view_data['ax']
        
        # 绘制前沿点
        if hasattr(local_map, 'frontiers') and local_map.frontiers:
            for x, y in local_map.frontiers:
                marker = plt.Circle((x, y), 0.15, color='yellow', alpha=0.7, zorder=4)
                ax.add_patch(marker)
                view_data['frontier_markers'].append(marker)
        
        # 绘制中心点
        if hasattr(local_map, 'centroids') and local_map.centroids:
            for centroid, data in local_map.centroids.items():
                # 绘制中心点
                star_marker = ax.scatter(centroid[0], centroid[1], c='magenta', marker='*', 
                                       s=200, zorder=7)
                view_data['centroid_markers'].append(star_marker)
                
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
                    view_data['centroid_boxes'].append(rect)
                    
                    # 显示未知面积
                    text = ax.text(centroid[0] + 0.5, centroid[1] + 0.5, f"面积: {data['area']}", 
                                 fontsize=8, color='magenta', weight='bold', zorder=7,
                                 bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
                    view_data['centroid_texts'].append(text)
                except (IndexError, KeyError) as e:
                    print(f"无法绘制中心点 {centroid} 的边界框: {e}")
        
        # 绘制任务
        self._draw_agent_tasks(agent_name, local_map)

    def _draw_agent_tasks(self, agent_name: str, local_map: "Map"):
        """在指定智能体的视图中绘制任务"""
        view_data = self.agent_views[agent_name]
        ax = view_data['ax']
        
        # 检查地图中是否有任务信息
        if not hasattr(local_map, 'tasks_seen') or not local_map.tasks_seen:
            return
        
        # 检查token信息，确定哪些任务已被分配或完成
        completed_tasks = set()
        if hasattr(local_map, 'token') and hasattr(local_map.token, 'c_tasks'):
            completed_tasks = set(local_map.token.c_tasks)
        
        # 分配给智能体但未完成的任务
        assigned_tasks = set()
        if hasattr(local_map, 'token') and hasattr(local_map.token, 'agents_to_tasks'):
            for name, tasks in local_map.token.agents_to_tasks.items():
                assigned_tasks.update(tasks)
        
        # 收集所有被拾取的任务 - 这里是关键修复点，我们需要收集所有智能体已拾取的任务
        picked_tasks = set()
        for agent_name, agent in self.all_agents.items():
            if hasattr(agent, 'picked_tasks') and agent.picked_tasks:
                picked_tasks.update(agent.picked_tasks)
        
        # 添加token中已完成的任务，确保它们不会重新显示
        for task_name in completed_tasks:
            picked_tasks.add(task_name)
        
        # 绘制所有可见任务，但排除已被拾取和已完成的任务
        for task in local_map.tasks_seen:
            if isinstance(task, dict) and 'task_name' in task and 'start' in task:
                task_name = task['task_name']
                
                # 如果任务已被拾取或已完成，则不显示
                if task_name in picked_tasks:
                    continue
                
                x, y = task['start']
                
                if task_name in completed_tasks:
                    # 已完成任务 - 理论上不应该走到这里，因为已经被排除了
                    continue
                elif task_name in assigned_tasks:
                    # 已分配任务 - 蓝色
                    circle = plt.Circle((x, y), 0.4, color='blue', alpha=0.7, zorder=5)
                    ax.add_patch(circle)
                    view_data['task_markers'].append(circle)
                    
                    text = ax.text(x, y, task_name, fontsize=8, color='white', weight='bold', 
                                ha='center', va='center', zorder=6)
                    view_data['task_texts'].append(text)
                else:
                    # 未分配任务 - 绿色
                    circle = plt.Circle((x, y), 0.4, color='green', alpha=0.8, zorder=5)
                    ax.add_patch(circle)
                    view_data['task_markers'].append(circle)
                    
                    text = ax.text(x, y, task_name, fontsize=8, color='black', weight='bold', 
                                ha='center', va='center', zorder=6)
                    view_data['task_texts'].append(text)

    def _clear_agent_dynamic_elements(self, agent_name: str):
        """清除指定智能体视图中的动态元素"""
        if agent_name not in self.agent_views:
            return
        
        view_data = self.agent_views[agent_name]
        ax = view_data['ax']
        
        # 清除前沿点
        for marker in view_data['frontier_markers']:
            try:
                if marker in ax.patches:
                    marker.remove()
            except (ValueError, RuntimeError) as e:
                pass
        view_data['frontier_markers'] = []
        
        # 清除中心点
        for marker in view_data['centroid_markers']:
            try:
                if marker in ax.collections:
                    marker.remove()
            except (ValueError, RuntimeError) as e:
                pass
        view_data['centroid_markers'] = []
        
        # 清除边界框
        for rect in view_data['centroid_boxes']:
            try:
                if rect in ax.patches:
                    rect.remove()
            except (ValueError, RuntimeError) as e:
                pass
        view_data['centroid_boxes'] = []
        
        # 清除中心点文本
        for text in view_data['centroid_texts']:
            try:
                if text in ax.texts:
                    text.remove()
            except (ValueError, RuntimeError) as e:
                pass
        view_data['centroid_texts'] = []
        
        # 清除任务标记
        for marker in view_data['task_markers']:
            try:
                if marker in ax.patches:
                    marker.remove()
            except (ValueError, RuntimeError) as e:
                pass
        view_data['task_markers'] = []
        
        # 清除任务文本
        for text in view_data['task_texts']:
            try:
                if text in ax.texts:
                    text.remove()
            except (ValueError, RuntimeError) as e:
                pass
        view_data['task_texts'] = []

    def _create_legend(self, ax, local_map: "Map"):
        """在指定子图上创建图例"""
        legend_elements = []
        
        # 地图元素图例
        legend_elements.append(patches.Patch(color='gray', alpha=0.5, label='未知区域'))
        legend_elements.append(patches.Patch(color='white', edgecolor='lightgray', label='自由空间'))
        legend_elements.append(patches.Patch(color='black', label='障碍物'))
        legend_elements.append(patches.Patch(color='royalblue', alpha=0.8, label='基地点'))
        
        # 任务图例
        legend_elements.append(patches.Circle((0, 0), radius=0.1, color='green', alpha=0.8, label='未分配任务'))
        legend_elements.append(patches.Circle((0, 0), radius=0.1, color='blue', alpha=0.7, label='已分配任务'))
        legend_elements.append(patches.Circle((0, 0), radius=0.1, color='gray', alpha=0.5, label='已完成任务'))
        
        # 前沿点图例
        if hasattr(local_map, 'frontiers') and local_map.frontiers:
            legend_elements.append(patches.Circle((0, 0), radius=0.1, color='yellow', alpha=0.7, label='前沿点'))
        
        # 中心点图例
        if hasattr(local_map, 'centroids') and local_map.centroids:
            legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='magenta', 
                                           markersize=12, label='中心点'))
        
        # 添加图例，使用较小的字体和简洁的布局
        if legend_elements:
            legend = ax.legend(handles=legend_elements, loc='lower right', fontsize=7, 
                             ncol=2, framealpha=0.7)
    
    def _get_agent_color(self, agent_name):
        """
        根据智能体名称分配唯一颜色
        使用预定义的明显不同的颜色，再根据需要动态生成
        """
        # 如果智能体已有颜色，则返回它
        if agent_name in self.agent_color_map:
            return self.agent_color_map[agent_name]
        
        # 分配新颜色
        if self.next_color_idx < len(self.predefined_colors):
            # 使用预定义颜色
            color = self.predefined_colors[self.next_color_idx]
            self.next_color_idx += 1
        else:
            # 如果预定义颜色用完，则动态生成
            # 使用Golden Ratio方法生成更多均匀分布的颜色
            golden_ratio = 0.618033988749895
            h = (self.next_color_idx * golden_ratio) % 1
            # HSV转RGB，使用饱和度和明度为0.95，以确保颜色鲜明
            h, s, v = h, 0.95, 0.95
            # 转换HSV到RGB
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
            self.next_color_idx += 1
        
        # 保存智能体的颜色映射
        self.agent_color_map[agent_name] = color
        return color

    def save_figure(self, filename):
        """保存当前图像为文件"""
        if self.fig:
            try:
                # 确保目录存在
                os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
                
                # 保存图像
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"图像已保存到: {filename}")
            except Exception as e:
                print(f"保存图像时出错: {e}")