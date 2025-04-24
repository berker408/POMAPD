"""
实现任务分配的相关逻辑
"""
from typing import TYPE_CHECKING
import random
if TYPE_CHECKING:
    from .agent import Agent
    from .map import Map

from .utils import *


class Task_Planner:
    """
    集合任务分配的相关逻辑
    task_planner会修改agent类的
    schedule和assigned_tasks属性
    l_map的u_tasks属性
    """
    def __init__(self):
        pass
    
    def assign_frontiers(self, agents: list["Agent"], l_map: "Map", t):
        """
        将前沿区域的中心点分配给各个agent
        优先分配面积大的区域，分配给距离最近的agent
        每个agent最多分配一个frontier访问点，并加入schedule
        
        Parameters:
            agents: 代理列表
            l_map: 地图对象
            t: 当前时间步
        """
        # 获取并排序中心点
        centroids = list(l_map.centroids.items())
        # 创建未分配的agent列表
        # unassigned_agents = [agent for agent in agents if agent.schedule == [] or agent.schedule[0]['name'] != 'frontier']
        unassigned_agents = [agent for agent in agents if agent.schedule == []] # 只有没事做的agent去探索
        if not centroids or not unassigned_agents:
            return
        centroids_to_nearest_agent = {centroid_pos: min(unassigned_agents, key=lambda agent: self._get_distance(agent.pos, centroid_pos)) 
                                      for centroid_pos, _ in centroids}
        
        centroids_to_utility = {centroid_pos: centroid_data['area'] / self._get_distance(centroids_to_nearest_agent[centroid_pos].pos, centroid_pos)
                                 for centroid_pos, centroid_data in centroids}
        # 按照utility从大到小排序
        centroids.sort(key=lambda x: centroids_to_utility[x[0]], reverse=True)

        agents_to_frontiers = {agent: agent.schedule[0]['pos'] for agent in agents if agent.schedule and agent.schedule[0]['name'] == 'frontier'}
        frontiers_to_agents = {v: k for k, v in agents_to_frontiers.items()}

        # 按utility顺序分配中心点
        for centroid_pos, centroid_data in centroids:
            if not unassigned_agents:  # 如果所有agent都已分配，则结束
                break
            # 找到距离该中心点最近的未分配agent
            nearest_agent = min(unassigned_agents, 
                            key=lambda agent: self._get_distance(agent.pos, centroid_pos))
            nearest_agent.schedule.insert(0, {'name': 'frontier', 'pos': centroid_pos})

            if centroid_pos in frontiers_to_agents:
                # 如果该centroid已经分配给了其他agent，则将其schedule中的frontier去掉
                other_agent = frontiers_to_agents[centroid_pos]
                if other_agent != nearest_agent:
                    other_agent.schedule.pop(0)  # 去掉frontier
            frontiers_to_agents[centroid_pos] = nearest_agent  # 更新分配关系

            # 从未分配列表中移除该agent
            unassigned_agents.remove(nearest_agent)
    # def assign_frontiers(self, agents: list["Agent"], l_map: "Map", t):
    #     """
    #     将前沿区域的中心点分配给各个agent
    #     优先分配面积大的区域，分配给距离最近的agent
    #     每个agent最多分配一个frontier访问点，并加入schedule
        
    #     Parameters:
    #         agents: 代理列表
    #         l_map: 地图对象
    #         t: 当前时间步
    #     """
    #     # 获取并排序中心点
    #     centroids = list(l_map.centroids.items())
    #     # 按照面积从大到小排序
    #     centroids.sort(key=lambda x: x[1]['area'], reverse=True)
        
    #     # 创建未分配的agent列表
    #     # unassigned_agents = [agent for agent in agents if agent.schedule == [] or agent.schedule[0]['name'] != 'frontier']
    #     unassigned_agents = [agent for agent in agents if agent.schedule == []] # 只有没事做的agent去探索

    #     agents_to_frontiers = {agent: agent.schedule[0]['pos'] for agent in agents if agent.schedule and agent.schedule[0]['name'] == 'frontier'}
    #     frontiers_to_agents = {v: k for k, v in agents_to_frontiers.items()}

    #     # 按面积优先顺序分配中心点
    #     for centroid_pos, centroid_data in centroids:
    #         if not unassigned_agents:  # 如果所有agent都已分配，则结束
    #             break
    #         # 找到距离该中心点最近的未分配agent
    #         nearest_agent = min(unassigned_agents, 
    #                         key=lambda agent: self._get_distance(agent.pos, centroid_pos))
    #         nearest_agent.schedule.insert(0, {'name': 'frontier', 'pos': centroid_pos})

    #         if centroid_pos in frontiers_to_agents:
    #             # 如果该centroid已经分配给了其他agent，则将其schedule中的frontier去掉
    #             other_agent = frontiers_to_agents[centroid_pos]
    #             if other_agent != nearest_agent:
    #                 other_agent.schedule.pop(0)  # 去掉frontier
    #         frontiers_to_agents[centroid_pos] = nearest_agent  # 更新分配关系

    #         # 从未分配列表中移除该agent
    #         unassigned_agents.remove(nearest_agent)

    def random_destroy(self, agents: list["Agent"], l_map: "Map", t, rate=0.1):
        """
        random destory agents' tasks 
        """
        for agent in agents:
            actionable_task_names = [task_name for task_name in agent.assigned_tasks if task_name not in agent.picked_tasks]
            if actionable_task_names and random.random() < rate:
                random_task_name = random.choice(actionable_task_names)
                agent.assigned_tasks.remove(random_task_name)
                agent.schedule = [s for s in agent.schedule if s['name'] != random_task_name]
                l_map.u_tasks.add(random_task_name)  # 将任务重新加入未分配任务列表






    def assign_tasks(self, agents: list["Agent"], l_map: "Map", t):
        """
        给agents分配任务
        """
        # self.random_destroy(agents, l_map, t, 0.1) # 随机删除任务
        while not self.is_full(agents, l_map):
            for agent in agents:
                if not self.is_full(agents, l_map) and len(agent.assigned_tasks) < agent.capacity:
                    # 如果能分配任务，先把去frontier的schedule去掉
                    if agent.schedule and agent.schedule[0]['name'] == 'frontier':
                        agent.schedule.pop(0)
                    # 如果要插入任务，先把base去掉，因为现在的逻辑没有
                    if agent.schedule and agent.schedule[-1]['name'] == 'base':
                        agent.schedule.pop(-1)
                    new_schedule, best_task_name = self._get_best_task(agent, l_map)
                    agent.assigned_tasks.append(best_task_name)
                    agent.schedule = new_schedule
                    l_map.u_tasks.remove(best_task_name)
            # 分配完任务后，给每个agent分配一个base点 todo全局信息时ok
            self.assign_base(agents, l_map)
    

    def assign_base(self, agents: list["Agent"], l_map: "Map"):
        """
        给每个agent分配一个base点
        现在直接让每个agent回起点,这样可以保证在分布式情况下不会冲突
        """
        for agent in agents:
            if agent.assigned_tasks and agent.schedule[-1]['name'] != 'base':
                # 未分配base
                new_schedule = agent.schedule + [{'name': 'base', 'pos': agent.start}]
                agent.schedule = new_schedule
            
    # def assign_base(self, agents: list["Agent"], l_map: "Map"):
    #     base_points = l_map.base_points
    #     occupied_base_points = set([agent.schedule[-1]['pos'] for agent in agents if agent.schedule and agent.schedule[-1]['name'] == 'base'])
    #     for agent in agents:
    #         if agent.assigned_tasks and agent.schedule[-1]['name'] != 'base':
    #             # 未分配base
    #             base_points.sort(key=lambda x: self._get_distance(agent.schedule[-1]['pos'], x))
    #             for base_point in base_points:
    #                 if base_point not in occupied_base_points:
    #                     min_base_point = base_point
    #                     break
    #             new_schedule = agent.schedule + [{'name': 'base', 'pos': min_base_point}]
    #             agent.schedule = new_schedule


    ##=======================================
    # task assignment
    #========================================
    def _get_best_task(self, agent: "Agent", local_map: "Map"):
        min_cost = float('inf')
        best_task_name = None
        best_schedule = None


        for task_name in local_map.u_tasks:
            task = local_map._tasks_dict[task_name]
            tmp_schedule, tmp_cost = self._insert_task(agent, agent.schedule, task, local_map.base_points)
            if tmp_cost < min_cost:
                min_cost = tmp_cost
                best_task_name = task_name
                best_schedule = tmp_schedule
        return best_schedule, best_task_name

    def _insert_task(self, agent, schedule, task, base_points):
        """
        插入任务到schedule中，返回新的schedule和cost
        """
        new_schedule = None
        new_cost = None
        sch_len = len(schedule)
        for i in range(sch_len+1): # 插入在第i个位置之前

            if i == sch_len:
                tmp_schedule = schedule + [{'name': task['task_name'], 'pos': task['start']}]
                tmp_cost = self._get_sche_cost(agent, tmp_schedule, base_points)
                if new_cost is None or tmp_cost < new_cost:
                    new_schedule = tmp_schedule
                    new_cost = tmp_cost
                break
            tmp_schedule = schedule[:i] + [{'name': task['task_name'], 'pos': task['start']}] + schedule[i:]
            tmp_cost = self._get_sche_cost(agent, tmp_schedule, base_points)
            if new_cost is None or tmp_cost < new_cost:
                new_schedule = tmp_schedule
                new_cost = tmp_cost
        return new_schedule, new_cost
        
            
    
    def _get_sche_cost(self, agent, schedule, base_points):
        """
        计算schedule的cost
        """
        cost = self._get_distance(agent.pos, schedule[0]['pos'])
        for i in range(len(schedule)-1):
            cost += self._get_distance(schedule[i]['pos'], schedule[i+1]['pos'])
        
        if schedule[-1]['name'] != 'base':
            min_base_point = min(base_points, key=lambda x: self._get_distance(schedule[-1]['pos'], x))
            cost += self._get_distance(schedule[-1]['pos'], min_base_point)
 
        return cost
    
    def _get_distance(self, pos1, pos2):
        """
        计算两点之间的距离
        """
        return q_distance(pos1, pos2)

    ##==========================
    # axiliary functions
    ##==========================
    def is_full(self, agents: list["Agent"], local_map: "Map"):
        """
        判断agents是否满载
        """
        if not local_map.u_tasks:
            return True
        for agent in agents:
            if len(agent.assigned_tasks) < agent.capacity:
                return False
        return True
    
