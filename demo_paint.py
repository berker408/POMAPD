import argparse
import yaml
import json
import os
from tqdm import tqdm
import RoothPath
import subprocess
import sys
import random
from copy import deepcopy

from src.agent import Agent
from src.map import Map
from src.utils import *
from src.task_planner import Task_Planner
from src.path_planner import Path_Planner
from src.simulation import Simulation
from src.painter import Painter  # 引入Painter类


if __name__ == '__main__':
    random.seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument('-sim_steps', type=int, default=100, help='simulation steps')
    parser.add_argument('--animate', action='store_true', help='使用动画可视化')
    parser.add_argument('--save', help='保存动画到指定文件路径', default=None)
    parser.add_argument('--interval', type=int, default=500, help='动画帧间隔(毫秒)')
    parser.add_argument('--render', action='store_true', default=False, help='在仿真过程中实时显示动画')

    args = parser.parse_args()

    with open(os.path.join(RoothPath.get_root(), 'config', 'config.json'), 'r') as json_file:
        config = json.load(json_file)
    args.param = os.path.join(RoothPath.get_root(), os.path.join(config['input_path'], config['input_name']))
    args.output = os.path.join(RoothPath.get_root(), 'experiments', 'output.yaml')

    # Read from input file
    with open(args.param, 'r', encoding="utf-8") as param_file:
        try:
            param = yaml.load(param_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    dimensions = param['map']['dimensions']
    obstacles = param['map']['obstacles']

    base_points = param['map']['base_points']

    agents = param['agents']
    
    tasks = param['tasks']


    with open(args.param + config['visual_postfix'], 'w') as param_file:
        yaml.safe_dump(param, param_file)
    
    # init map
    init_print("generate map-------------------")
    REAL_MAP = Map(param['map'], tasks, laser_range=3, screen=True)
    l_map = REAL_MAP.back_to_unknown() # local map
    # init agents
    init_print("generate agents--------------------")
    # agents 共用一个l_map,表示信息共享
    workers = [Agent(a_info, l_map) for a_info in agents[:10]] # 只用前两个agent
    # task_planner
    task_planner = Task_Planner()

    path_planner = Path_Planner()

    simulation = Simulation()

    ##==========================
    # Main loop for simulation
    ##==========================
    
    # 使用动画模式或传统模式
    if args.animate:
        # 创建可视化器
        painter = Painter()
        
        # 启动动画仿真
        painter.start_animation(
        sim_steps=args.sim_steps,
        local_map=l_map,
        agents=workers,
        realmap=REAL_MAP,
        simulation=simulation,
        task_planner=task_planner,
        path_planner=path_planner,
        interval=args.interval,
        save_path=args.save,  # 现在直接传递保存路径
        render=args.render  # 是否实时显示动画
        )
    else:
        # 传统仿真模式
        ##-----------------
        # Initialization
        print('\n')
        init_print('Init scanning ---------------')
        messages = {worker.name: worker.message for worker in workers}
        for worker in workers:
            worker.execute(REAL_MAP, -1) # 执行一次，更新地图信息
            messages[worker.name] = worker.publish(-1)

        MAX_STEP = args.sim_steps
        for t in range(MAX_STEP):
            print('\n')
            init_print(f'** Time {t} **')
            init_print('>> ------------------------------------')
            init_print('>> Step 1: Online update and task Planning')
            # 1. Update the map and task infos
            simulation.update_map_infos(l_map, REAL_MAP, messages)
            #simulation.update_task_infos(l_map, REAL_MAP, messages)
            task_planner.assign_tasks(workers, l_map, t)
            task_planner.assign_frontiers(workers, l_map, t)

            planned_paths = path_planner.plan_paths(workers, l_map)

            for worker in workers:
                worker.update(t)

            init_print('>> ------------------------------------')
            init_print('>> Step 2: Execution for MAS')
            for worker in workers:
                worker.execute(REAL_MAP, t)
                messages[worker.name] = worker.publish(t)
            
            # 在每步后显示当前地图状态（可选）
            l_map.show2(show_frontiers=True, show_centroids=True)