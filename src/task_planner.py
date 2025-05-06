"""
实现任务分配的相关逻辑
"""
from typing import TYPE_CHECKING
import random
if TYPE_CHECKING:
    from .agent import Agent
    from .map import Map
    from .Token import Token

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
    
    def assign_frontiers(self, comm_g: list["Agent"], token: "Token", t): #todo!!!!!!!!!!!!!!!!
        """
        将前沿区域的中心点分配给各个agent
        优先分配面积大的区域，分配给距离最近的agent
        每个agent最多分配一个frontier访问点，并加入schedule
        
        Parameters:
            comm_g: 代理列表
            l_map: 地图对象
            t: 当前时间步
        """
        # 获取中心点
        l_map = comm_g[0].map                    # 这里假设所有agent的地图是一样的
        centroids = list(l_map.centroids.items())
        # 创建未分配的agent列表
        unassigned_agents = [agent for agent in comm_g if agent.assigned_tasks == []] # 只有没事做的agent去探索
        if not centroids or not unassigned_agents:
            return
        centroids_to_nearest_agent = {centroid_pos: min(unassigned_agents, key=lambda agent: self._get_distance(agent.pos, centroid_pos)) 
                                      for centroid_pos, _ in centroids}
        
        centroids_to_utility = {centroid_pos: centroid_data['area'] / self._get_distance(centroids_to_nearest_agent[centroid_pos].pos, centroid_pos)
                                 for centroid_pos, centroid_data in centroids}
        # 按照utility从大到小排序
        centroids.sort(key=lambda x: centroids_to_utility[x[0]], reverse=True)

        #agents_to_frontiers = {agent_n : schedule[0]['pos'] for agent_n, schedule in token.agents_to_schedules.items() if schedule and schedule[0]['name'] == 'frontier'}
        agents_to_frontiers = {agent: agent.schedule[0]['pos'] for agent in comm_g if agent.schedule and agent.schedule[0]['name'] == 'frontier'}
        frontiers_to_agents = {v: k for k, v in agents_to_frontiers.items()}

        # 按utility顺序分配中心点
        for centroid_pos, centroid_data in centroids:
            if not unassigned_agents:  # 如果所有agent都已分配，则结束
                break
            # 找到距离该中心点最近的未分配agent
            nearest_agent = min(unassigned_agents, 
                            key=lambda agent: self._get_distance(agent.pos, centroid_pos))
            nearest_agent.schedule.insert(0, {'name': 'frontier', 'pos': centroid_pos})
            token.agents_to_schedules[nearest_agent.name].insert(0, {'name': 'frontier', 'pos': centroid_pos})

            if centroid_pos in frontiers_to_agents: # 只考虑了comm_g内的agent
                # 如果该centroid已经分配给了其他agent，则将其schedule中的frontier去掉
                other_agent = frontiers_to_agents[centroid_pos]
                if other_agent != nearest_agent:
                    other_agent.schedule.pop(0)  # 去掉frontier
                    token.agents_to_schedules[other_agent.name].pop(0) 
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
                l_map.unassigned_tasks.add(random_task_name)  # 将任务重新加入未分配任务列表



    def task_conflict_check(self, comm_g: list["Agent"], b_token: "Token"):
        """检查任务的冲突"""
        if len(comm_g) < 2:
            return
        
        # 1. 收集通信图中所有智能体已分配的任务
        agent_names = [agent.name for agent in comm_g]
        agent_dict = {agent.name: agent for agent in comm_g}
        task_assignments = {}  # {task_name: [agent_names]}
        l_map = comm_g[0].map # 这里假设所有agent的地图是一样的
        
        for agent_name in agent_names:
            # 获取该智能体的任务列表
            agent_tasks = b_token.agents_to_tasks.get(agent_name, [])
            for task_name in agent_tasks:
                task_assignments.setdefault(task_name, []).append(agent_name)
        
        # 2. 找出所有被多个智能体分配的任务（冲突任务）
        conflict_tasks = {task_name: agents for task_name, agents in task_assignments.items() if len(agents) > 1}

        # 3. 处理每个冲突任务
        for task_name, conflicting_agents in conflict_tasks.items():
            # 收集已拾取该任务的智能体
            picked_agents = [agent_name for agent_name in conflicting_agents 
                            if task_name in agent_dict[agent_name].picked_tasks]
            
            # 获取任务信息（起点和终点）
            task_info = None
            # 从任务字典中查找任务信息
            if  task_name in b_token._tasks_dict:
                task_dict_entry = b_token._tasks_dict[task_name]
                task_info = [task_dict_entry['start'], task_dict_entry['goal']]
            
            # 如果仍然无法获取任务信息，记录日志并跳过
            if not task_info:
                print(f"Warning: Cannot find information for conflict task_name {task_name}")
                continue
            
            # 4. 处理冲突：从未拾取任务的智能体中移除任务分配
            for agent_name in conflicting_agents:
                # 如果智能体已经拾取任务，则不移除
                if agent_name in picked_agents:
                    continue
                    
                # 从智能体的任务列表中移除任务
                if agent_name in b_token.agents_to_tasks and task_name in b_token.agents_to_tasks[agent_name]:
                    b_token.agents_to_tasks[agent_name].remove(task_name)
                    agent_dict[agent_name].assigned_tasks.remove(task_name) # 更新agent的assigned_tasks

                    b_token.agents_to_schedules[agent_name] = [s for s in b_token.agents_to_schedules[agent_name] if s['name'] != task_name]
                    agent_dict[agent_name].schedule = [s for s in agent_dict[agent_name].schedule if s['name'] != task_name] # 更新agent的schedule
        
            # 5. 将任务加回到未分配任务列表
            if not picked_agents and task_info:
                b_token.unassigned_tasks.add(task_name)  # 将任务重新加入未分配任务列表


    def assign_tasks(self, comm_g: list["Agent"], b_token: "Token", t):
        """
        这里的comm_g可能只是一个comm_g内的agents
        给agents分配任务
        """
        self.task_conflict_check(comm_g, b_token) # 检查任务冲突
        while not self.is_full(comm_g, b_token):
            change_flag = False
            for agent in comm_g:
                if not self.is_full(comm_g, b_token) and len(agent.assigned_tasks) < agent.capacity:
                    # 如果能分配任务，先把去frontier的schedule去掉
                    if agent.schedule and agent.schedule[0]['name'] == 'frontier':
                        agent.schedule.pop(0)
                    # 如果要插入任务，先把base去掉，因为现在的计算cost时不考虑base在schedule里
                    if agent.schedule and agent.schedule[-1]['name'] == 'base':
                        agent.schedule.pop(-1)
                    new_schedule, best_task_name = self._get_best_task(agent, b_token)
                    agent.assigned_tasks.append(best_task_name)
                    agent.schedule = new_schedule
                    # 一个一个agent式地更新token里的相关信息
                    b_token.unassigned_tasks.remove(best_task_name)
                    change_flag = True
            # 分配完任务后，给每个agent分配一个base点 todo全局信息时ok
            if self.assign_base(comm_g): change_flag = True

            if change_flag:
                for agent in comm_g:
                    # 更新token里的信息
                    b_token.agents_to_tasks[agent.name] = agent.assigned_tasks.copy()
                    b_token.agents_to_schedules[agent.name] = agent.schedule.copy()
                    
   

    

    def assign_base(self, agents: list["Agent"]):
        """
        给每个agent分配一个base点
        现在直接让每个agent回起点,这样可以保证在分布式情况下不会冲突
        """
        flag = False
        for agent in agents:
            if agent.assigned_tasks and agent.schedule[-1]['name'] != 'base':
                # 未分配base
                agent.schedule = agent.schedule + [{'name': 'base', 'pos': agent.start}]
                flag = True
        return flag

            

    ##=======================================
    # task assignment
    #========================================
    def _get_best_task(self, agent: "Agent", b_token: "Token"):
        min_cost = float('inf')
        best_task_name = None
        best_schedule = None


        for task_name in b_token.unassigned_tasks:
            task = b_token._tasks_dict[task_name]
            tmp_schedule, tmp_cost = self._insert_task(agent, agent.schedule, task)
            if tmp_cost < min_cost:
                min_cost = tmp_cost
                best_task_name = task_name
                best_schedule = tmp_schedule
        return best_schedule, best_task_name

    def _insert_task(self, agent, schedule, task):
        """
        插入任务到schedule中，返回新的schedule和cost
        """
        new_schedule = None
        new_cost = None
        sch_len = len(schedule)
        for i in range(sch_len+1): # 插入在第i个位置之前

            if i == sch_len:
                tmp_schedule = schedule + [{'name': task['task_name'], 'pos': task['start']}]
                tmp_cost = self._get_sche_cost(agent, tmp_schedule)
                if new_cost is None or tmp_cost < new_cost:
                    new_schedule = tmp_schedule
                    new_cost = tmp_cost
                break
            tmp_schedule = schedule[:i] + [{'name': task['task_name'], 'pos': task['start']}] + schedule[i:]
            tmp_cost = self._get_sche_cost(agent, tmp_schedule)
            if new_cost is None or tmp_cost < new_cost:
                new_schedule = tmp_schedule
                new_cost = tmp_cost
        return new_schedule, new_cost
        
            
    
    def _get_sche_cost(self, agent: "Agent", schedule):
        """
        计算schedule的cost
        """
        cost = self._get_distance(agent.pos, schedule[0]['pos'])
        for i in range(len(schedule)-1):
            cost += self._get_distance(schedule[i]['pos'], schedule[i+1]['pos'])
        
        if schedule[-1]['name'] != 'base':
            base_point = agent.start # 每个agent固定回自己的起点就行
            cost += self._get_distance(schedule[-1]['pos'], base_point)
 
        return cost
    
    def _get_distance(self, pos1, pos2):
        """
        计算两点之间的距离
        """
        return q_distance(pos1, pos2)

    ##==========================
    # axiliary functions
    ##==========================
    def is_full(self, agents: list["Agent"], b_token: "Token"):
        """
        判断agents是否满载
        """
        if not b_token.unassigned_tasks:
            return True
        for agent in agents:
            if len(agent.assigned_tasks) < agent.capacity:
                return False
        return True
    
