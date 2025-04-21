import os
import json
import csv
import time
from typing import TYPE_CHECKING, Dict, List, Set, Tuple, Any
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
from copy import deepcopy
import numpy as np

if TYPE_CHECKING:
    from Simulation import Simulation
    from Agent import Agent, Base
    from Map import Map

class Logger:
    """
    用于记录和可视化仿真过程中的数据
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
        self.performance_metrics = {"algorithm_time": [], "completed_tasks": []}

        # 添加对agents_to_picked_tasks_t的存储
        self.agents_to_picked_tasks_t = {}  # 从simulation复制过来
        for agent_name, data in simulation.agents_to_picked_tasks_t.items():
            self.agents_to_picked_tasks_t[agent_name] = deepcopy(data)
        
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
            "dimensions": self.simulation.map.dimensions,
            "obstacles": list(self.simulation.map.obstacle_points),
            "base_points": list(self.simulation.map.base_points)
        }
        with open(os.path.join(self.data_dir, "map_info.json"), "w") as f:
            json.dump(map_data, f, indent=4)
        
        # 记录代理初始位置
        time_step = self.simulation.time
        self.agent_positions[time_step] = {}
        for agent in self.simulation.agents:
            self.agent_positions[time_step][agent.name] = agent.position
        
        # 记录任务信息
        task_info = []
        for task in self.simulation.tasks:
            task_data = {
                "name": task["task_name"],
                "start": task["start"],
                "goal": task["goal"],
                "start_time": task["start_time"]
            }
            task_info.append(task_data)
            # 初始化任务状态
            self.task_status[task["task_name"]] = {
                "released": task["start_time"], 
                "picked": None, 
                "completed": None
            }
        
        with open(os.path.join(self.data_dir, "tasks_info.json"), "w") as f:
            json.dump(task_info, f, indent=4)
    
    def update(self):
        """每个时间步更新日志数据"""
        time_step = self.simulation.time
        
        # 记录代理位置
        self.agent_positions[time_step] = {}
        for agent in self.simulation.agents:
            self.agent_positions[time_step][agent.name] = agent.position
        
        # 记录通信图
        comm_graphs, _ = self.simulation.create_comm_graphs()
        # 转换通信图为可序列化格式
        serializable_graphs = []
        for graph in comm_graphs:
            serializable_graph = []
            for node in graph:
                if node.name == "base":
                    serializable_graph.append({"name": "base", "positions": node.positions})
                else:
                    serializable_graph.append({"name": node.name, "position": node.position})
            serializable_graphs.append(serializable_graph)
        
        self.communication_graphs[time_step] = serializable_graphs
        
        # 记录代理任务分配
        self.agent_tasks[time_step] = {}
        self.picked_tasks[time_step] = {}
        for agent in self.simulation.agents:
            self.agent_tasks[time_step][agent.name] = agent.token.agents_to_tasks.get(agent.name, [])
            self.picked_tasks[time_step][agent.name] = agent.token.agents_picked_tasks.get(agent.name, [])
        
        # 更新任务状态
        for task_name, completion_time in self.simulation.completed_tasks_times.items():
            if task_name in self.task_status and self.task_status[task_name]["completed"] is None:
                self.task_status[task_name]["completed"] = completion_time
                
        for agent in self.simulation.agents:
            for task_name, pick_time in agent.token.pick_tasks_times.items():
                if task_name in self.task_status and self.task_status[task_name]["picked"] is None:
                    self.task_status[task_name]["picked"] = pick_time
                    
        # 记录性能指标
        self.performance_metrics["algorithm_time"].append(self.simulation.stats["algorithm_time"])
        self.performance_metrics["completed_tasks"].append(len(self.simulation.completed_tasks_times))
    
        # 更新agents_to_picked_tasks_t
        for agent_name, data in self.simulation.agents_to_picked_tasks_t.items():
            self.agents_to_picked_tasks_t[agent_name] = deepcopy(data)

    def save_data(self):
        """保存所有记录的数据"""
        # 保存代理路径数据
        with open(os.path.join(self.data_dir, "agent_paths.json"), "w") as f:
            json.dump(self.simulation.actual_paths, f, indent=4)
        
        # 保存任务状态数据
        with open(os.path.join(self.data_dir, "task_status.json"), "w") as f:
            json.dump(self.task_status, f, indent=4)
        
        # 保存通信图数据
        with open(os.path.join(self.data_dir, "communication_graphs.json"), "w") as f:
            json.dump(self.communication_graphs, f, indent=4)
        
        # 保存代理任务分配
        with open(os.path.join(self.data_dir, "agent_tasks.json"), "w") as f:
            json.dump(self.agent_tasks, f, indent=4)
        
        # 保存已拾取任务
        with open(os.path.join(self.data_dir, "picked_tasks.json"), "w") as f:
            json.dump(self.picked_tasks, f, indent=4)
            
        # 保存性能指标
        with open(os.path.join(self.data_dir, "performance_metrics.json"), "w") as f:
            json.dump(self.performance_metrics, f, indent=4)
        
        # 保存仿真统计
        with open(os.path.join(self.data_dir, "simulation_stats.json"), "w") as f:
            json.dump(self.simulation.stats, f, indent=4)
            
    def generate_summary(self):
        """生成仿真摘要"""
        summary = {
            "total_time_steps": self.simulation.time,
            "number_of_agents": len(self.simulation.agents),
            "total_tasks": len(self.simulation.tasks),
            "completed_tasks": len(self.simulation.completed_tasks_times),
            "avg_task_completion_time": self._calculate_avg_completion_time(),
            "communication_events": self.simulation.stats["communication_events"],
            "algorithm_time": self.simulation.stats["algorithm_time"]
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
    
    def plot_agent_paths(self):
        """绘制代理路径图"""
        fig, ax = plt.figure(figsize=(12, 10)), plt.subplot()
        # 设置图表边界
        width, height = self.simulation.map.dimensions
        
        # 绘制完整网格
        self._draw_complete_grid(ax, width, height)
        
        # 绘制障碍物
        obstacles = self.simulation.map.obstacle_points
        for obs in obstacles:
            ax.add_patch(plt.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, color='black'))
        
        # 绘制基站
        for base_pos in self.simulation.map.base_points:
            ax.add_patch(plt.Rectangle((base_pos[0] - 0.5, base_pos[1] - 0.5), 1, 1, color='lightblue', alpha=0.7))
        
        # 为每个代理绘制路径
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.simulation.agents)))
        for i, agent in enumerate(self.simulation.agents):
            path = self.simulation.actual_paths[agent.name]
            xs = [point['x'] for point in path]
            ys = [point['y'] for point in path]
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
    
    def create_animation(self):
        """创建代理移动动画，包含任务显示和代理携带任务信息"""
        import numpy as np
        import matplotlib.animation as animation
        
        fig, ax = plt.subplots(figsize=(12, 10))
        width, height = self.simulation.map.dimensions
        
        # 绘制完整网格
        self._draw_complete_grid(ax, width, height)
        
        # 绘制障碍物和基站（静态）
        for obs in self.simulation.map.obstacle_points:
            ax.add_patch(plt.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, color='black'))
        
        for base_pos in self.simulation.map.base_points:
            ax.add_patch(plt.Rectangle((base_pos[0] - 0.5, base_pos[1] - 0.5), 1, 1, color='lightblue', alpha=0.7))
        
        # 为每个代理创建一个散点
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.simulation.agents)))
        scatter_plots = []
        agent_names = []
        agent_labels = []  # 代理标签显示携带的任务
        
        for i, agent in enumerate(self.simulation.agents):
            scatter = ax.scatter([], [], s=100, color=colors[i], label=agent.name)
            scatter_plots.append(scatter)
            agent_names.append(agent.name)
            # 添加代理标签，用于显示携带的任务
            label = ax.text(0, 0, "", fontsize=8, ha='center', va='bottom', alpha=0)
            agent_labels.append(label)
        
        # 为任务创建散点（黄色三角形）
        task_scatter = ax.scatter([], [], marker='^', s=80, color='yellow', edgecolor='black')
        
        # 在左上角添加时间步显示
        time_step_text = ax.text(0.02, 0.98, "Time Step: 0", transform=ax.transAxes, 
                                fontsize=12, fontweight='bold', va='top')
        
        # 创建图例
        #ax.legend(loc='upper right')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(False)
        
        def update(frame):
            # 更新代理位置
            positions = []
            
            for i, agent_name in enumerate(agent_names):
                # 获取代理路径数据
                path_data = self.simulation.actual_paths[agent_name]
                current_frame = min(frame, len(path_data)-1)
                pos = path_data[current_frame]
                positions.append((pos['x'], pos['y']))
                
                # 获取代理携带的任务信息
                agent_tasks = []
                
                # 使用更精确的方式确定当前帧代理携带的任务
                # 1. 获取当前时间步
                current_time = pos['t']
                
                # 2. 查找对应时间步的已拾取任务记录
                picked_tasks_data = self.simulation.agents_to_picked_tasks_t[agent_name]
                
                # 找到最接近当前时间步的记录
                matching_record = None
                for record in picked_tasks_data:
                    if record['t'] == current_time:
                        matching_record = record
                        break
                    elif record['t'] < current_time:
                        matching_record = record
                    else:
                        break
                
                # 使用匹配的记录
                if matching_record:
                    agent_tasks = matching_record['picked_tasks']
                    
                # 确认这些任务在当前时间点仍然被代理携带(未被完成)
                confirmed_tasks = []
                for task in agent_tasks:
                    # 检查任务是否已经完成
                    if task in self.task_status:
                        completed_time = self.task_status[task]['completed']
                        # 如果任务尚未完成或完成时间晚于当前时间，则代理仍然携带它
                        if completed_time is None or completed_time > current_time:
                            confirmed_tasks.append(task)
                
                # 更新代理标签文本和位置
                if confirmed_tasks:
                    agent_labels[i].set_text(f"{agent_name}\n{', '.join(confirmed_tasks)}")
                else:
                    agent_labels[i].set_text(agent_name)
                agent_labels[i].set_position((pos['x'], pos['y'] + 0.4))
                agent_labels[i].set_alpha(1)  # 显示标签
            
            # 使用不同的变量名更新代理散点位置
            for j, scatter in enumerate(scatter_plots):
                if j < len(positions):  # 确保不越界
                    scatter.set_offsets([positions[j]])
                
            # 更新任务位置（只显示已发布但未被拾取的任务）
            task_positions = []
            current_time = min(frame, self.simulation.time)
            
            for task_name, status in self.task_status.items():
                # 任务已发布但未被拾取
                if status['released'] <= current_time and (status['picked'] is None or status['picked'] > current_time):
                    if task_name in self.simulation.task_dict:
                        task_start = self.simulation.task_dict[task_name]['start']
                        task_positions.append(task_start)
            
            # 使用空数组而不是空列表，以避免维度错误
            if task_positions:
                task_scatter.set_offsets(task_positions)
            else:
                task_scatter.set_offsets(np.zeros((0, 2)))  # 使用空的2D数组
            
            # 更新时间步文本
            time_step_text.set_text(f"Time Step: {min(frame, self.simulation.time)}")
            
            # 返回所有需要更新的artists
            return scatter_plots + [task_scatter, time_step_text] + agent_labels
        
        # 计算最大帧数并创建动画
        try:
            max_frames = max(len(self.simulation.actual_paths[agent.name]) for agent in self.simulation.agents)
            
            # 使用更安全的方式创建动画
            anim = animation.FuncAnimation(
                fig, 
                update, 
                frames=range(max_frames),
                interval=400,
                blit=True
            )
            
            # 保存动画
            anim_path = os.path.join(self.animations_dir, 'agent_movement.mp4')
            writer = animation.FFMpegWriter(fps=3)
            anim.save(anim_path, writer=writer)
            print(f"动画已保存至 {anim_path}")
            
        except Exception as e:
            print(f"创建动画时出错: {e}")
            # 尝试使用备用方法
            try:
                writer = animation.PillowWriter(fps=5)
                anim_path = os.path.join(self.animations_dir, 'agent_movement.gif')
                anim.save(anim_path, writer=writer)
                print(f"动画已保存为GIF: {anim_path}")
            except Exception as e2:
                print(f"保存GIF时出错: {e2}")
        finally:
            plt.close(fig)
        
    def plot_communication_graphs(self):
        """绘制通信图在不同时间步的快照"""
        import matplotlib.patches as mpatches
        
        times_to_sample = min(10, self.simulation.time)  # 最多取10个时间点
        sample_times = [t for t in range(1, self.simulation.time, max(1, self.simulation.time // times_to_sample))]
        print(f"采样时间点: {sample_times}")
        
        for t in sample_times:
            if t not in self.communication_graphs:
                continue
                
            fig, ax = plt.subplots(figsize=(10, 8))
            width, height = self.simulation.map.dimensions
            
            # 绘制完整网格
            self._draw_complete_grid(ax, width, height)
            
            # 绘制障碍物
            for obs in self.simulation.map.obstacle_points:
                ax.add_patch(plt.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, color='black'))
            
            # 绘制基站（统一用蓝色填充方格表示）
            for base_pos in self.simulation.map.base_points:
                ax.add_patch(plt.Rectangle((base_pos[0] - 0.5, base_pos[1] - 0.5), 1, 1, color='lightblue', alpha=0.7))
            
            # 为每个通信组创建一个NetworkX图
            for i, graph in enumerate(self.communication_graphs[t]):
                G = nx.Graph()
                
                # 添加节点（只添加代理节点，不添加基站节点）
                for node in graph:
                    if node["name"] != "base":
                        G.add_node(node["name"], pos=node["position"])
                
                # 处理基站点的连接
                base_nodes = [node for node in graph if node["name"] == "base"]
                agent_nodes = [node for node in graph if node["name"] != "base"]
                
                # 为代理节点间添加边
                nodes = list(G.nodes())
                for j in range(len(nodes)):
                    for k in range(j+1, len(nodes)):
                        # 检查两个节点是否在通信范围内
                        pos1 = G.nodes[nodes[j]]["pos"]
                        pos2 = G.nodes[nodes[k]]["pos"]
                        distance = max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))
                        
                        if distance <= 2:  # 通信范围
                            G.add_edge(nodes[j], nodes[k])
                
                # 添加代理与基站的连接（使用特殊边样式）
                if base_nodes:
                    base_positions = base_nodes[0]["positions"]
                    for agent_node in agent_nodes:
                        agent_pos = agent_node["position"]
                        # 检查代理是否与任何基站点在通信范围内
                        for base_pos in base_positions:
                            distance = max(abs(agent_pos[0] - base_pos[0]), abs(agent_pos[1] - base_pos[1]))
                            if distance <= 2:  # 通信范围
                                # 在NetworkX图中添加基站位置的边（特殊样式）
                                if agent_node["name"] in G:  # 确保代理节点已添加到图中
                                    # 在绘图时添加到最近的基站点的连线
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
            base_patch = mpatches.Patch(color='lightblue', alpha=0.7, label='Base Station')
            obstacle_patch = mpatches.Patch(color='black', label='Obstacle')
            agent_comm_line = plt.Line2D([0], [0], color='blue', linewidth=2, label='Agent-Agent Comm')
            base_comm_line = plt.Line2D([0], [0], color='green', linestyle='--', linewidth=1.5, label='Agent-Base Comm')
            
            # 添加图例
            ax.legend(handles=[base_patch, obstacle_patch, agent_comm_line, base_comm_line], 
                    loc='upper right', bbox_to_anchor=(1, 1))
            
            ax.set_title(f'Communication Graph - Time Step: {t}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f"comm_graph_t{t}.png"), dpi=300)
            plt.close()

    def plot_task_completion_over_time(self):
        """绘制随时间的任务完成情况"""
        completed_tasks = [0]
        time_steps = [0]
        
        for t in range(1, self.simulation.time + 1):
            completed = sum(1 for _, status in self.task_status.items() 
                            if status["completed"] is not None and status["completed"] <= t)
            completed_tasks.append(completed)
            time_steps.append(t)
            
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, completed_tasks, '-o', linewidth=2)
        plt.title('Completed Tasks Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Number of Completed Tasks')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "tasks_completion.png"), dpi=300)
        plt.close()
    
    def plot_algorithm_time(self):
        """绘制算法计算时间"""
        times = self.performance_metrics["algorithm_time"]
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(times)), times, '-o')
        plt.title('Algorithm Computation Time')
        plt.xlabel('Time Step')
        plt.ylabel('Time (seconds)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "algorithm_time.png"), dpi=300)
        plt.close()
    
    def plot_tasks_gantt_chart(self):
        """
        绘制任务执行甘特图，展示各个代理的任务执行情况
        - 每个代理对应其容量数量的时间轴
        - 任务从被拾取开始显示，到被完成结束
        - 在最后一个任务完成的时间处添加一条红色竖虚线
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
        agent_y_start = {}  # 记录每个代理在甘特图上的起始Y坐标
        
        # 为每个代理分配Y坐标位置
        for agent in self.simulation.agents:
            agent_y_start[agent.name] = current_y
            for slot in range(agent.capacity):
                y_label = f"{agent.name} (Slot {slot+1})"
                y_labels.append(y_label)
                y_positions.append(current_y)
                y_ticks.append(current_y + 0.5)
                current_y += 1
        
        # 收集每个时间步的任务分配情况
        task_execution_data = {}  # {task_name: {picked_time, completed_time, agent, slot}}
        
        # 首先确定每个任务的拾取和完成时间
        for task_name, status in self.task_status.items():
            picked_time = status["picked"]
            completed_time = status["completed"]
            
            if picked_time is not None:  # 任务已被拾取
                # 寻找是哪个代理拾取了这个任务
                for agent in self.simulation.agents:
                    # 检查代理的token记录
                    if task_name in agent.token.pick_tasks_times:
                        agent_name = agent.name
                        # 确定任务占用了哪个插槽
                        slot = self._find_task_slot(agent_name, task_name, picked_time)
                        
                        task_execution_data[task_name] = {
                            "picked_time": picked_time,
                            "completed_time": completed_time,
                            "agent": agent_name,
                            "slot": slot
                        }
                        break
        
        # 绘制甘特图的任务条
        for task_name, data in task_execution_data.items():
            if data["picked_time"] is None or data["completed_time"] is None:
                continue  # 跳过未完成的任务
                
            # 计算任务条的Y位置
            agent_y = agent_y_start[data["agent"]]
            y_position = agent_y + data["slot"]
            
            # 绘制任务条
            start_time = data["picked_time"]
            end_time = data["completed_time"]
            width = end_time - start_time
            
            # 绘制矩形表示任务
            rect = mpatches.Rectangle(
                (start_time, y_position), width, 1, 
                color=task_color_map[task_name], 
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
        ax.set_ylabel("Agent (Capacity Slot)")
        ax.set_xlabel("Time Step")
        ax.set_title("Task Execution Gantt Chart")
        
        # 设置X轴范围
        ax.set_xlim(0, self.simulation.time)
        ax.set_ylim(0, current_y)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 找出最后一个任务完成的时间
        last_task_completion_time = 0
        if self.simulation.completed_tasks_times:
            last_task_completion_time = max(self.simulation.completed_tasks_times.values())
            
            # 在最后一个任务完成的时间处添加红色竖虚线
            ax.axvline(x=last_task_completion_time, color='red', linestyle='--', linewidth=2)
            
            # 添加文本标注
            ax.text(
                last_task_completion_time + 0.5, 
                current_y * 0.98, 
                f"All tasks completed: t={last_task_completion_time}",
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
    
    def _find_task_slot(self, agent_name, task_name, picked_time):
        """
        确定任务在代理的哪个容量插槽中
        这是一个简化实现，实际上需要根据任务被拾取时代理中任务的情况来确定
        """
        # 找到任务被拾取时的时间点数据
        agent_data = self.simulation.agents_to_picked_tasks_t[agent_name]
        
        # 找到最接近拾取时间的记录
        closest_record = None
        for record in agent_data:
            if record['t'] <= picked_time:
                closest_record = record
            else:
                break
        
        if closest_record is None:
            return 0  # 默认放在第一个插槽
        
        # 获取当时代理已拾取的所有任务
        picked_tasks = closest_record['picked_tasks']
        
        # 确定task_name在任务列表中的位置，这将代表它的插槽
        if task_name in picked_tasks:
            return picked_tasks.index(task_name)
        
        # 如果没有找到任务（可能是刚被拾取），则放在下一个可用插槽
        return len(picked_tasks)
    
    # 添加一个辅助方法来绘制完整网格
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
        ax.grid(False)  # 关闭默认网格，因为我们已经手动添加了

    def plot_tasks_lifecycle_gantt(self):
        """
        绘制任务生命周期甘特图
        - 横轴为时间步
        - 纵轴为各个任务
        - 从任务被发布的时间开始绘制长条，到任务完成时结束
        - 在任务被拾取时用不同颜色区分未被拾取和已被拾取的状态
        """
        import matplotlib.patches as mpatches
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS'] 
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
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
        
        # 创建图表 - 增加上下边距来确保所有任务标签可见
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
                    color=waiting_color,
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
                    color=execution_color,
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
                    f"{total_time} steps",  # 使用英文避免中文显示问题
                    ha='center',
                    va='center',
                    color='black',
                    fontsize=8,
                    fontweight='bold'
                )
        
        # 添加图例
        waiting_patch = mpatches.Patch(color=waiting_color, label='Waiting for pickup')
        execution_patch = mpatches.Patch(color=execution_color, label='In execution')
        ax.legend(handles=[waiting_patch, execution_patch], loc='upper right')
        
        # 设置轴标签和标题
        ax.set_yticks(y_positions)
        ax.set_yticklabels(task_names)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Tasks')
        ax.set_title('Task Lifecycle Gantt Chart')
        
        # 设置Y轴范围，确保留有足够空间显示所有标签
        ax.set_ylim(-0.8, len(y_positions) - 0.2)
        
        # 设置X轴范围
        ax.set_xlim(0, self.simulation.time)
        
        # 添加网格
        ax.grid(True, axis='x', alpha=0.3)
        
        # 找出最后一个任务完成的时间
        if self.simulation.completed_tasks_times:
            last_task_completion_time = max(self.simulation.completed_tasks_times.values())
            
            # 在最后一个任务完成的时间处添加红色竖虚线
            ax.axvline(x=last_task_completion_time, color='red', linestyle='--', linewidth=2)
            
            # 添加文本标注到图表中间位置，避免与图题重叠
            ax.text(
                last_task_completion_time + 0.5,
                len(sorted_tasks) / 2,  # 放在图表中间位置
                f"All tasks completed: t={last_task_completion_time}",
                ha='left',
                va='center',
                color='red',
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5')
            )
        
        # 增加子图的上下边距，确保所有元素可见
        plt.tight_layout(pad=2.0)
        
        # 保存图表
        plt.savefig(os.path.join(self.plots_dir, "tasks_lifecycle_gantt.png"), dpi=300)
        plt.close()
        
        print("已生成任务生命周期甘特图")
        
    def generate_visualizations(self):
        """生成所有可视化图表"""
        print("正在生成可视化图表...")
        
        # 导入必要的库
        import numpy as np
        
        try:
            self.plot_agent_paths()
            print("已生成代理路径图")
            
            self.plot_task_completion_over_time()
            print("已生成任务完成情况图")
            
            self.plot_algorithm_time()
            print("已生成算法计算时间图")
            
            self.plot_communication_graphs()
            print("已生成通信图快照")

            self.plot_tasks_gantt_chart()  # 新增：生成任务甘特图
            print("已生成任务甘特图")

            self.plot_tasks_lifecycle_gantt()  # 新增：生成任务生命周期甘特图
            

            self.create_animation()
            print("已生成代理移动动画")
        except Exception as e:
            print(f"生成可视化图表时出错: {e}")
        
        print("可视化图表生成完成")

# 以下是Logger类的使用示例，你可以在主程序中使用
if __name__ == "__main__":
    print("Logger module imported. To use, create a Logger instance with your simulation object.")