import heapq
from collections import defaultdict
from typing import TYPE_CHECKING
import random

if TYPE_CHECKING:
    from .agent import Agent
    from .map import Map

DEBUG = False

class Path_Planner:
    def __init__(self):
        """
        初始化路径规划器
        """
        self.reservation = {
            'nodes': defaultdict(set),   # key: time, value: set of (x, y)
            'edges': defaultdict(set)     # key: time, value: set of (x1, y1, x2, y2)
        }
    
    def plan_paths(self, workers: list["Agent"], map_obj: "Map"):
        """
        为所有工作者规划无碰撞路径
        :param workers: 工作者列表
        :param map_obj: 地图对象
        :return: dict {agent_name: full_path}，其中 full_path 为一系列 (x,y) 坐标的列表
        """
        # 重置预约表
        self.reservation = {
            'nodes': defaultdict(set),   
            'edges': defaultdict(set)     
        }
        
        # 获取地图尺寸和障碍物
        dimensions = (map_obj.width, map_obj.height)
        obstacles = set()
        
        # 从地图中提取障碍物和未知区域，都视为不可通行
        for y in range(map_obj.height):
            for x in range(map_obj.width):
                # 1表示障碍物，-1表示未知区域，都视为障碍
                if map_obj.map[y][x] == 1 or map_obj.map[y][x] == -1:
                    obstacles.add((x, y))
        
        planned_paths = {}
        
        # 按照任务计划长度排序工作者，计划点少的在前
        sorted_workers = sorted(workers, key=lambda w: len(w.schedule))
        
        for worker in sorted_workers:
            waypoints = [worker.pos]  # 从当前位置开始
            
            # 添加所有任务点
            for task in worker.schedule:
                waypoints.append(task['pos'])
                
            if len(waypoints) == 1:  # 如果没有任务点，只有当前位置
                planned_paths[worker.name] = waypoints
                self.reserve_path(waypoints)
                worker.planned_path = waypoints
                continue
                
            full_path = []
            current = waypoints[0]
            start_time = 0
            full_path.append(current)
            
            # 对于每个连续的目标点，规划一段路径
            for goal in waypoints[1:]:
                segment = self.a_star(current, goal, start_time, dimensions, obstacles)
                
                if segment is None:
                    # 如果找不到路径，保持原地不动
                    if DEBUG:
                        print(f"Agent {worker.name}: 无法从 {current} 到 {goal} 规划路径！")
                    segment = [current]
                
                # 如果不是第一段，避免重复最后一个点
                if full_path and full_path[-1] == segment[0]:
                    full_path.extend(segment[1:])
                else:
                    full_path.extend(segment)
                
                # 更新起始时间和当前位置
                start_time += len(segment) - 1
                current = segment[-1]
                
            planned_paths[worker.name] = full_path
            worker.planned_path = full_path
            # 预定该 agent 的路径，防止后续 agent 产生冲突
            self.reserve_path(full_path)
            
        return planned_paths

    def a_star(self, start, goal, start_time, dimensions, obstacles):
        """
        基于时间扩展的 A* 算法规划路径，状态为 (x, y, t)
        :param start: (x,y) 起始坐标
        :param goal: (x,y) 目标坐标
        :param start_time: 规划起始时刻
        :param dimensions: 地图尺寸 (width, height)
        :param obstacles: 障碍物集合
        :return: 从 start 到 goal 的路径（列表，每个元素为 (x,y)）
        """
        open_list = []
        # 节点格式：(f, t, (x,y), path)，其中 path 是从 start_time 到当前状态的 (x,y) 序列
        h_start = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        heapq.heappush(open_list, (h_start + start_time, start_time, start, [start]))
        closed = set()  # 保存 (x, y, t)

        max_iterations = 5000  # 避免无限循环
        iterations = 0

        while open_list and iterations < max_iterations:
            f, t, current, path = heapq.heappop(open_list)

            if current == goal:
                return path
                
            # 考虑 4 个方向以及等待动作
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0), (0,0)]:
                nx, ny = current[0] + dx, current[1] + dy
                nt = t + 1

                if not (0 <= nx < dimensions[0] and 0 <= ny < dimensions[1]):
                    continue
                if (nx, ny) in obstacles:
                    continue
                # 检查该时间步目标位置是否已被预定
                if (nx, ny) in self.reservation['nodes'].get(nt, set()):
                    continue
                # 检查边冲突：即检查是否有 agent 在前一时刻从 (nx, ny) 移动到当前位置
                if (nx, ny, current[0], current[1]) in self.reservation['edges'].get(t, set()):
                    continue
                state = (nx, ny, nt)
                if state in closed:
                    continue
                closed.add(state)
                new_path = path + [(nx, ny)]
                cost = nt - start_time
                h = abs(nx - goal[0]) + abs(ny - goal[1])
                heapq.heappush(open_list, (cost + h, nt, (nx, ny), new_path))
            iterations += 1

        # 如果达到最大迭代次数或无法找到路径，返回None
        return None

    def reserve_path(self, path):
        """
        将规划好的路径预定到 reservation 表中，防止后续规划发生冲突。
        同时，在 agent 到达终点后，将终点持续预定一段时间以避免其他 agent 与其碰撞。
        :param path: agent 的完整路径列表 [(x,y), ...]
        """
        # 预定路径中每个时间步的位置和边
        for t, pos in enumerate(path):
            self.reservation['nodes'][t].add(pos)
            if t < len(path) - 1:
                next_pos = path[t + 1]
                self.reservation['edges'][t].add((pos[0], pos[1], next_pos[0], next_pos[1]))
        # 为终点持续预定后续若干时间步（例如 20 步）
        final_pos = path[-1]
        final_time = len(path) - 1
        for t in range(final_time + 1, final_time + 21):
            self.reservation['nodes'][t].add(final_pos)