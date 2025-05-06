from copy import deepcopy
import os
from itertools import chain
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .map import Map
    from .agent import Agent, Base
    from .path_planner import Path_Planner
    from .task_planner import Task_Planner
    from .dec_painter import PainterDec
    from .logger import Logger

class Simulation():
    """
    self 相关的逻辑
    """
    def __init__(self, task_planner: "Task_Planner", path_planner: "Path_Planner", agents: list["Agent"], real_map: "Map", base: "Base", screen=True):
        self.screen = screen

        # dependecy injection
        self.task_planner = task_planner
        self.path_planner = path_planner
        self.agents = agents
        self.real_map = real_map
        self.base = base
        
        # 用于处理agent要去一个任务，但是已经被其他agent拾取的情况
        self._picked_tasks = []  # 
        self._c_tasks = [] # 已完成的任务名字列表
        # 存储当前的通信图，供绘图使用
        self.comm_graphs = None
        self.comm_graphs_no_base = None
        
        # 可视化绘图器，默认为None
        self.painter = None
        
        # 数据记录器，默认为None
        self.logger = None
    
    def setup_visualization(self, painter: "PainterDec" = None):
        """设置可视化组件"""
        if painter:
            self.painter = painter
            # 初始化绘图器
            self.painter.setup(self.agents, self.base, self.real_map)
    
    def setup_logger(self, logger: "Logger" = None):
        """设置日志记录器"""
        if logger:
            self.logger = logger
    
    # ==========================
    # process control
    # ==========================
    def simulate_step_dec(self, t, render=False, save_path=None):
        """执行一个仿真步骤并返回是否应该结束仿真, 在分布式且包含通信限制的情况下"""
        print(f'\n** 仿真步骤 {t} **')

        messages = {agent.name: agent.message for agent in self.agents}

        # 1. 构建通信子图
        self.comm_graphs, self.comm_graphs_no_base = self.make_comm_graphs(comm_range=self.agents[0].comm_range)


        # 2. 通信子图内同步信息
        for comm_graph, comm_graph_no_base in zip(self.comm_graphs, self.comm_graphs_no_base):
            if len(comm_graph) != len(comm_graph_no_base):
                # 通信图中包含基站
                if len(comm_graph_no_base) == 0:
                    # 只有基站，直接跳过
                    continue
            b_token = self.sync_token_map(comm_graph, messages, t) # 同步token和map信息，可以处理带基站的情况

            # 任务分配和路径规划，过程中会直接更新b_token
            self.task_planner.assign_tasks(comm_graph_no_base, b_token, t)
            self.task_planner.assign_frontiers(comm_graph_no_base, b_token, t)
            self.path_planner.plan_paths(comm_graph_no_base, comm_graph[0].map, b_token)
            #self.print(f"b_token.agents_to_paths : {b_token.agents_to_paths}")

            for agent in comm_graph:
                agent.token.update(b_token) # 更新token信息
        
        # 3. 更新智能体
        for agent in self.agents:
            agent.update(t)
        
        # 4. 执行仿真步骤
        for agent in self.agents:
            agent.execute(self.real_map, t, self)
            messages[agent.name] = agent.publish(t)

        self._picked_tasks = set(chain.from_iterable([agent.picked_tasks for agent in self.agents]))
        self._c_tasks = set(chain.from_iterable([agent.token.c_tasks for agent in self.agents])) # 已完成的任务名字列表

        # 5. 如果需要渲染，更新可视化
        if render and self.painter:
            self.painter.update(t, self)
        
        # 6. 记录日志数据
        if self.logger:
            self.logger.update(t)
        
        # 7. 检查是否应该结束仿真
        if len(self._c_tasks) == len(self.agents[0].token._tasks):
            print("所有任务已完成，仿真结束！")
            return True
        
        return False
    
    # ==========================
    # update functions
    # ==========================

    def sync_token_map(self, comm_graph: list["Agent", "Base"], messages, t):
        """
        同步token和地图信息
        """
        # 同步地图的信息
        base_map = comm_graph[0].map
        map_list = [agent.map for agent in comm_graph]
        merged_map = base_map.merge_maps(map_list) # 代表整个通信图的地图信息
        #self.print(f"merged_map: {merged_map.map}")
        # 同步token的信息
        b_token = comm_graph[0].token
        token_list = [agent.token for agent in comm_graph]
        b_token = b_token.merge_tokens(token_list, comm_graph, t)

        # 更新map信息
        all_senses = set()
        a_names_cg = [agent.name for agent in comm_graph if agent.name != 'base'] # 通信图内的agent名字
        for name, info in messages.items():
            if name in a_names_cg: # 收集通讯图内的信息
                all_senses.update(info['sense']) 
        merged_map.update(self.real_map, all_senses, b_token) # 更新地图信息, 同时更新token中的unassigned_tasks

        # 地图不会在任务分配的过程中更新，可以给各个agent在此处更新地图
        for agent in comm_graph:
            agent.map = deepcopy(merged_map)
        self.print(f"u_tasks: {b_token.unassigned_tasks}, c_tasks: {b_token.c_tasks}")

        return b_token # token后续在任务分配过程中会再次更新
        
        


    def make_comm_graphs(self, comm_range=3):  
        """
        创建通信图集合，基于无穷范数距离
        使用广度优先搜索找到所有连通的通信组，包括基站
        :param comm_range: 最大通信距离
        :return: 通信图列表，每个通信图是一个相邻agent的列表，可能包含基站
        """
        # 构建邻接表表示通信关系，包括基站
        all_nodes = self.agents[:]
        all_nodes.append(self.base)  # 复制所有代理的列表
        adjacency = {node: [] for node in all_nodes}
        # 确定哪些节点可以直接通信
        for i, node1 in enumerate(all_nodes):
            for j, node2 in enumerate(all_nodes):
                if i == j:  # 避免自己与自己比较
                    continue
                # 如果节点1是基站
                if node1.name == 'base':
                    # 基站有多个位置，检查所有位置与node2的距离
                    for pos in node1.positions:
                        distance = max(abs(pos[0] - node2.pos[0]), 
                                    abs(pos[1] - node2.pos[1]))
                        if distance <= comm_range:
                            adjacency[node1].append(node2)  # 如果在通信范围内，建立连接
                            break  # 只要有一个基站点能连接就可以
                # 如果节点2是基站
                elif node2.name == 'base':
                    # 检查node1与所有基站位置的距离
                    for pos in node2.positions:
                        distance = max(abs(node1.pos[0] - pos[0]), 
                                    abs(node1.pos[1] - pos[1]))
                        if distance <= comm_range:
                            adjacency[node1].append(node2)  # 如果在通信范围内，建立连接
                            break  # 只要有一个基站点能连接就可以            
                # 两个普通代理之间的通信
                else:
                    distance = max(abs(node1.pos[0] - node2.pos[0]),
                                abs(node1.pos[1] - node2.pos[1]))
                    if distance <= comm_range:
                        adjacency[node1].append(node2)  # 如果在通信范围内，建立连接
        
        # 使用BFS构建连通分量（通信图）
        visited = set()
        comm_graphs = []
        for start_node in all_nodes:
            if start_node in visited:
                continue  # 如果该节点已经被处理，跳过
            # 开始BFS
            comm_graph = []
            queue = [start_node]
            visited.add(start_node)
            while queue:
                current_node = queue.pop(0)
                comm_graph.append(current_node)
                # 查找所有相邻且未访问的节点
                for neighbor in adjacency[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            comm_graphs.append(comm_graph)
        
        comm_graphs_without_base = [list(filter(lambda node: node != self.base, comm_g)) for comm_g in comm_graphs]

        return comm_graphs, comm_graphs_without_base  # 返回通信图和不包含基站的通信图

    def print(self, string):
        if self.screen:
            print(f'[Simulation] ' + string)

    def save_visualization(self, filename):
        """保存当前可视化状态为图像文件"""
        if self.painter:
            self.painter.save_figure(filename)
            
    def run_simulation(self, max_steps, render=False, save_path=None):
        """
        运行整个仿真过程
        :param max_steps: 最大步数
        :param render: 是否渲染
        :param save_path: 保存路径
        :return: 最终步数和是否正常完成
        """
        # 初始化扫描
        print('\n')
        print('[Simulation] Init scanning ---------------')
        messages = {worker.name: worker.message for worker in self.agents}
        for worker in self.agents:
            worker.execute(self.real_map, -1)  # 执行一次，更新地图信息
            messages[worker.name] = worker.publish(-1)
            
        # 如果有Logger，记录初始状态
        if self.logger:
            self.logger.record_initial_state()
            
        # 主循环
        for t in range(max_steps):
            print('\n')
            print(f'[Simulation] ** Time {t} **')
            should_stop = self.simulate_step_dec(t, render=render)
            
            # 如果需要保存当前帧
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                filename = os.path.join(save_path, f"frame_{t:03d}.png")
                self.save_visualization(filename)
            
            # 如果仿真应该结束
            if should_stop:
                print(f"仿真在步骤 {t} 结束")
                
                # 如果有Logger，保存数据和生成可视化
                if self.logger:
                    self.logger.save_data()
                    self.logger.generate_summary()
                    self.logger.generate_visualizations()
                    
                return t + 1, True  # 正常完成
        
        # 如果达到最大步数仍未完成
        if self.logger:
            self.logger.save_data()
            self.logger.generate_summary()
            self.logger.generate_visualizations()
            
        return max_steps, False  # 达到最大步数