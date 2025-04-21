#!/usr/bin/env python3

import copy
import random
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .map import Map

from .utils import *

LASER_RANGE = 3
COMM_RANGE = 3

class Agent():
    """
    Class for the mobile robot
    perceive and move in the environment
    """
    def __init__(self, info, map: "Map", screen=True):
        self.screen = screen
        self.__init_basic_info(info)
        # Environment and behavior
        self.senses = set()

        # Schedules
        self.schedule = [] # schedule 设计得复杂一点,[{'name':task_name, 'pos':pos},...]
        self.planned_path = [] # 由plan规划出的路径
        self.message = self.__init_message()
        self.picked_tasks = [] # [task_name]
        self.assigned_tasks = [] # [task_name]

        # 地图信息
        self.map = map # agent所看到的地图，任务的信息也直接从地图上得到

        # 记录信息
        self.actual_path = [{'t':'-1', 'pos': self.pos}] # 记录agent的实际路径

    ##==========================
    # Initialization functions
    ##==========================
    def __init_basic_info(self, info):
        # Extract from info
        # 静态参数
        self.name = info['name']
        self.capacity = info['capacity']
        self.start = info['start']
        self.laser_range = LASER_RANGE # 现在全是同质机器人
        self.comm_range = COMM_RANGE

        # 动态参数
        self.pos = info['start'] # 机器人初始位置
        self.next_pos = info['start'] # 机器人下一个位置




    def __init_message(self):
        """
        The structure of message attribute.
        """
        message = {
            'ct': 0,
            'pos': self.pos,
            'sense': set(),
            'picked_tasks': [],  # 记录在手上的任务
        }
        return message

    ##==================
    # Update functions
    ##==================
    def update(self, ct):
        """
        Update informations, such as  schedule, etc.
        """
        self.ct = ct
        self.print(f"updated schedule: {self.schedule}")

    def publish(self, ct):
        """
        Publish message that contains action, senses and etc.
        """
        self.message['ct'] = ct
        return self.message

    ##===================
    # Planing functions
    ##===================
    


    ##=====================
    # Execution functions
    ##=====================
    def execute(self, realmap, ct):
        """
        Simulate the execution of the agent.
        first move then scan the env
        """
        # Execution
        self.move(ct)
        self.scan(realmap)
        self.behave(ct)
        self.message['pos'] = self.pos
        self.message['sense'] = self.sense
        # Print the action
   


    def move(self, ct):
        """
        Move according to the planned path
        """
        if ct >= 0:
            if len(self.planned_path) > 1:
                self.pos = self.planned_path[1]
                self.planned_path = self.planned_path[1:] # 更新位置
            else:
                self.pos = self.planned_path[0]
        
        self.print(f"move to {self.pos}")
        self.actual_path.append({'t': ct, 'pos': self.pos}) # 记录实际路径
        
    def scan(self, realmap: "Map"):
        """
        Scan the real map to get the observed map
        ----------
        Parameters:
            @realmap: the real environment.
        """
        senses = set()
        obstacles = set()
        
        # 扫描agent周围laser_range范围内的所有格子
        for dx in range(-self.laser_range, self.laser_range + 1):
            for dy in range(-self.laser_range, self.laser_range + 1):
                x = self.pos[0] + dx
                y = self.pos[1] + dy
                
                # 检查坐标是否在地图范围内，且在激光范围内
                if (0<=x<realmap.width and  0<=y<realmap.height and 
                    q_distance(self.pos, (x, y)) <= self.laser_range):
                    
                    if realmap.map[y][x] != 1:
                        senses.add((x, y))
                    else:
                        obstacles.add((x, y))
        
        self.sense = senses.union(obstacles)

    def behave(self, ct):
        """
        进行抓取或放置, 其实也就是检查是否到达任务点
        """
        self.scan_frontier()
        self.pick()
        self.drop()

    def scan_frontier(self):
        if self.schedule and self.schedule[0]['name'] == 'frontier':
            if self.schedule[0]['pos'] not in self.map.centroids.keys() or self.pos == self.schedule[0]['pos']: # 如果消失了就重新分配
                self.schedule.pop(0)


    def pick(self):
        """模拟抓取任务的行为"""
        waypoints  = [s['pos'] for s in self.schedule]
        if self.pos in waypoints:
            for s in self.schedule:
                if self.is_task(s):
                    task_name = s['name']
                    self.picked_tasks.append(task_name)
                    self.schedule.remove(s)
    
    def drop(self):
        """模拟放置任务的行为"""
        x, y = self.pos
        if self.map.map[y][x] == 2:
            # base点
            self.map.u_tasks.difference_update(self.picked_tasks)
            self.map.c_tasks.update(self.picked_tasks)
            self.print(f"drop tasks: {self.picked_tasks}")
            self.assigned_tasks = []
            self.picked_tasks = []
            self.schedule = []

            

    ##=====================
    # Auxiliary functions
    ##=====================
    @staticmethod
    def is_task(s):
        """
        Check if the schedule is a task
        """
        if s['name'][:4] == 'task':
            return True
        return False 




    ##====================
    # Checking functions
    ##====================





    @staticmethod
    def random_locate(init_pos, ran=(0,0)):
        pos = [0 for k in range(len(init_pos))]
        for k in range(len(init_pos)):
            pos[k] = random.randint(init_pos[k]-ran[k], init_pos[k]+ran[k])
        return tuple(pos)



    def path2input(self, path, ct):
        
        pass

    ##=================
    # Print to screen
    ##=================
    def print(self, string, line=False):
        if self.screen:
            if line:
                print('----------')
            print(f'[Agent {self.name}] ' + string)