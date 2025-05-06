"""
至少需要token类来记录其他agent现在的任务分配, 路径分配情况
map类记录地图情况, 也统计现在的任务情况
Token和Map基本同时更新
"""
from  itertools import chain
from copy import deepcopy
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .agent import Agent, Base
    from .map import Map


class Token:
    def __init__(self, name, tasks):
        """
        记录任务分配，路径分配情况
        """
        self.name = name  # 所绑定到的agent的名字，或者是base
        self.agents_to_tasks = {}  # agent_name:task_name 记录每个agent的任务分配情况
        self.agents_to_schedules = {}  # 记录每个agent的路径关键点分配情况
        self.agents_to_paths = {}  # 记录每个agent的路径分配情况
        self.agents_to_comm_time = {}  # 记录和各个agent的通信时间

        self.unassigned_tasks = set() # 未分配的任务名字列表
        self.c_tasks = set() # 已完成的任务名字列表
        # 任务全局信息
        self._tasks = tasks
        self._tasks_dict = {t['task_name']: t for t in tasks}
    
    def _init_token(self, agents: list["Agent"]):
        pass# 初始化token的状态

    def update(self, info_dict):
        if isinstance(info_dict, Token):
            self.c_tasks = info_dict.c_tasks.copy()
            self.unassigned_tasks = info_dict.unassigned_tasks.copy()
            self.agents_to_tasks = deepcopy(info_dict.agents_to_tasks)
            self.agents_to_schedules = deepcopy(info_dict.agents_to_schedules)
            self.agents_to_paths = deepcopy(info_dict.agents_to_paths)
            self.agents_to_comm_time = info_dict.agents_to_comm_time.copy()
        else:
            # 更新Token的状态
            self.c_tasks = info_dict.get('c_tasks', set()).copy()
            self.unassigned_tasks = info_dict.get('unassigned_tasks', set()).copy()
            self.agents_to_tasks = deepcopy(info_dict.get('agents_to_tasks', {})) # item is mutable
            self.agents_to_schedules = deepcopy(info_dict.get('agents_to_schedules', {}))
            self.agents_to_paths = deepcopy(info_dict.get('agents_to_paths', {}))
            self.agents_to_comm_time = info_dict.get('agents_to_comm_time', {}).copy()

    
    @staticmethod
    def merge_tokens(token_list: list["Token"], comm_graph: list["Agent", "Base"], cur_time):
        """
        合并token信息,直接更改token_list中的token
        不会创建新的token对象
        """
        base_token = token_list[0]
        # 合并任务基础信息
        c_tasks = set()
        unassigned_tasks = set()
        cur_assigned_tasks = set()
        for token in token_list:
            c_tasks.update(token.c_tasks)
            cur_assigned_tasks.update(chain.from_iterable(token.agents_to_tasks.values()))
            unassigned_tasks.update(token.unassigned_tasks)
        unassigned_tasks.difference_update(c_tasks)
        unassigned_tasks.difference_update(cur_assigned_tasks)
        
        agent_to_tasks = {}
        agent_to_schedules = {}
        agent_to_paths = {}
        agent_to_comm_time = {}
        # 对于在通信图内的agents, 可以直接参考
        for agent in comm_graph:
            if agent.name != 'base':
                agent_to_tasks[agent.name] = agent.token.agents_to_tasks.get(agent.name, []).copy()
                agent_to_schedules[agent.name] = agent.token.agents_to_schedules.get(agent.name, []).copy()
                agent_to_paths[agent.name] = agent.token.agents_to_paths.get(agent.name, []).copy()
                agent_to_comm_time[agent.name] = cur_time
        
        # 对于不在通信图内的agents, 需要判断谁的信息更新
        all_agent_names = token_list[0].agents_to_tasks.keys()
        local_agent_names = [agent.name for agent in comm_graph if agent.name != 'base']
        for agent_name in all_agent_names:
            if agent_name in local_agent_names:
                continue
            latest_token = max(token_list, key=lambda x: x.agents_to_comm_time.get(agent_name, -1)) # 最新的token

            agent_to_tasks[agent_name] = latest_token.agents_to_tasks.get(agent_name, []).copy()
            agent_to_schedules[agent_name] = latest_token.agents_to_schedules.get(agent_name, []).copy()
            agent_to_paths[agent_name] = latest_token.agents_to_paths.get(agent_name, []).copy()
            agent_to_comm_time[agent_name] = latest_token.agents_to_comm_time.get(agent_name, -1)
        
        # 更新所有token的信息
        info_dict = {
            'c_tasks': c_tasks,
            'unassigned_tasks': unassigned_tasks,
            'agents_to_tasks': agent_to_tasks,
            'agents_to_schedules': agent_to_schedules,
            'agents_to_paths': agent_to_paths,
            'agents_to_comm_time': agent_to_comm_time
        }
        base_token.update(info_dict) # 更新第一个token的信息
        return base_token
        for token in token_list: # todo, 等任务分配完了再统一更新
            token.update(info_dict)



