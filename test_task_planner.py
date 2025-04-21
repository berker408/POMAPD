import argparse
import yaml
import json
import os
import random
from copy import deepcopy

import RoothPath
from src.agent import Agent
from src.map import Map
from src.utils import *
from src.task_planner import Task_Planner

def init_print(string):
    """输出带有格式的初始化信息"""
    print(f"[Init] {string}")

def test_print(string):
    """输出带有格式的测试信息"""
    print(f"[Test] {string}")

if __name__ == '__main__':
    # 设置随机种子以保证结果可重复
    random.seed(1234)
    
    # 设置参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('-sim_steps', type=int, default=10, help='simulation steps')
    args = parser.parse_args()
    
    # 加载配置文件
    with open(os.path.join(RoothPath.get_root(), 'config', 'config.json'), 'r') as json_file:
        config = json.load(json_file)
    args.param = os.path.join(RoothPath.get_root(), os.path.join(config['input_path'], config['input_name']))
    
    # 从输入文件读取参数
    with open(args.param, 'r', encoding="utf-8") as param_file:
        try:
            param = yaml.load(param_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    
    # 提取地图相关参数
    dimensions = param['map']['dimensions']
    obstacles = param['map']['obstacles']
    base_points = param['map']['base_points']
    agents_info = param['agents']
    tasks = param['tasks']
    
    # 初始化地图
    init_print("生成真实地图")
    REAL_MAP = Map(param['map'], tasks, laser_range=3, screen=True)
    
    # 初始化局部地图(代表agent的视角)
    init_print("生成局部地图")
    local_map = REAL_MAP.back_to_unknown()  # 初始时为完全未知状态
    
    # 初始化代理
    init_print("生成代理")
    workers = [Agent(a_info, local_map, screen=True) for a_info in agents_info[:2]]  # 只使用前两个代理
    
    # 初始化任务规划器
    init_print("初始化任务规划器")
    task_planner = Task_Planner()
    
    # 展示初始状态
    test_print(f"地图尺寸: {dimensions}")
    test_print(f"代理数量: {len(workers)}")
    test_print(f"可用任务: {len(tasks)}")
    
    # 初始化扫描，让代理获取周围环境信息
    init_print("初始扫描")
    messages = {}
    for worker in workers:
        worker.execute(REAL_MAP, -1)  # 先执行一次，更新地图信息
        messages[worker.name] = worker.publish(-1)
    
    # 更新地图信息 - 将所有代理的感知信息融合
    for worker_name, message in messages.items():
        local_map.update(REAL_MAP, message['sense'])
    
    # 为了测试任务规划器，手动添加一些未完成任务
    for task in tasks[:5]:  # 假设前5个任务是可见的
        task_name = task['task_name']
        local_map.u_tasks.add(task_name)
    
    test_print(f"未完成任务列表: {local_map.u_tasks}")
    
    # 测试任务分配
    MAX_STEP = args.sim_steps
    for t in range(MAX_STEP):
        print('\n')
        test_print(f"** 时间步 {t} **")
        test_print(">> 1. 分配任务")
        
        # 执行任务分配
        schedules = {}
        for worker in workers:
            if len(worker.assigned_tasks) < worker.capacity and local_map.u_tasks:
                new_schedule, best_task_name = task_planner._get_best_task(worker, local_map)
                if best_task_name:
                    worker.assigned_tasks.append(best_task_name)
                    schedules[worker.name] = new_schedule
                    local_map.u_tasks.remove(best_task_name)
                    test_print(f"为代理 {worker.name} 分配任务: {best_task_name}")
                else:
                    test_print(f"无可分配任务给代理 {worker.name}")
            else:
                if not local_map.u_tasks:
                    test_print("所有任务已分配完毕")
                else:
                    test_print(f"代理 {worker.name} 已达到容量上限")
        
        # 更新代理的行程安排
        test_print(">> 2. 更新代理行程")
        for worker in workers:
            if worker.name in schedules:
                worker.update(schedules[worker.name], t)
                test_print(f"代理 {worker.name} 的行程: {worker.schedule}")
        
        # 模拟代理执行
        test_print(">> 3. 模拟代理执行")
        for worker in workers:
            # 假设代理可以移动到下一个任务点
            if worker.schedule and worker.planned_path == []:
                # 为了简化测试，直接设置planned_path为到达任务点的路径
                next_pos = worker.schedule[0]['pos']
                worker.planned_path = [worker.pos, next_pos]
                test_print(f"代理 {worker.name} 规划路径到: {next_pos}")
            
            # 执行代理行为
            worker.execute(REAL_MAP, t)
            messages[worker.name] = worker.publish(t)
            test_print(f"代理 {worker.name} 当前位置: {worker.pos}")
            test_print(f"代理 {worker.name} 已拾取任务: {worker.picked_tasks}")
        
        # 更新地图
        for worker_name, message in messages.items():
            local_map.update(REAL_MAP, message['sense'])
        
        # 检查是否所有任务都已分配
        if not local_map.u_tasks and all(len(worker.assigned_tasks) == 0 for worker in workers):
            test_print("所有任务已完成，提前结束测试")
            break
    
    # 测试结束统计
    print('\n')
    test_print("==== 测试结束 ====")
    test_print(f"完成的任务: {local_map.c_tasks}")
    test_print(f"未完成的任务: {local_map.u_tasks}")
    for worker in workers:
        test_print(f"代理 {worker.name} 已分配任务: {worker.assigned_tasks}")
        test_print(f"代理 {worker.name} 已拾取任务: {worker.picked_tasks}")