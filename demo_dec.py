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
import matplotlib.pyplot as plt

from src.agent import Agent, Base
from src.map import Map
from src.utils import *
from src.task_planner import Task_Planner
from src.path_planner import Path_Planner
from src.simulation import Simulation
from src.Token import Token
from src.dec_painter import PainterDec  # 导入可视化组件
from src.logger import Logger  # 导入日志记录器


if __name__ == '__main__':
    random.seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument('-sim_steps', type=int, default=10, help='simulation steps')
    parser.add_argument('-render', action='store_true', help='render simulation')
    parser.add_argument('-save_path', type=str, default=None, help='path to save visualization')
    parser.add_argument('-log', action='store_true', help='enable detailed logging')
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
    # init tokens
    init_print("generate tokens--------------------")
    tokens = [Token(a_info['name'], tasks=tasks) for a_info in agents] 
    # init agents
    init_print("generate agents--------------------")
    # agents 一人一个l_map,局部信息
    workers = [Agent(a_info, deepcopy(l_map), tokens[idx]) for idx, a_info in enumerate(agents)] 


    base = Base('base', base_points, Token('base', tasks=tasks), deepcopy(l_map)) # base点的token

    # task_planner
    task_planner = Task_Planner()
    path_planner = Path_Planner()
    # Simulation类
    init_print("create Simulation--------------------")
    simulation = Simulation(task_planner, path_planner, workers, REAL_MAP, base, screen=True)
    
    # 初始化可视化组件（如果需要渲染）
    if args.render:
        init_print("setup visualization--------------------")
        painter = PainterDec(max_agents_to_display=len(workers))
        simulation.setup_visualization(painter)
    
    # 初始化日志记录器（如果启用日志）
    if args.log:
        init_print("setup logger--------------------")
        logger = Logger(simulation)
        simulation.setup_logger(logger)

    ##==========================
    # Main loop for simulation
    ##==========================
    ##-----------------
    # Initialization
    print('\n')
    init_print('Init scanning ---------------')
    messages = {worker.name: worker.message for worker in workers}
    for worker in workers:
        worker.execute(REAL_MAP, -1, simulation) # 执行一次，更新地图信息
        messages[worker.name] = worker.publish(-1)
    
    # 如果有日志记录器，记录初始状态
    if args.log and simulation.logger:
        simulation.logger.record_initial_state()

    MAX_STEP = args.sim_steps
    for t in range(MAX_STEP):
        print('\n')
        init_print(f'** Time {t} **')
        should_stop = simulation.simulate_step_dec(t, render=args.render) # 执行一个仿真步骤
        
        # 如果需要保存当前帧
        if args.save_path:
            # 创建保存目录（如果不存在）
            os.makedirs(args.save_path, exist_ok=True)
            filename = os.path.join(args.save_path, f"frame_{t:03d}.png")
            simulation.save_visualization(filename)
        
        # 如果仿真应该结束
        if should_stop:
            print(f"仿真在步骤 {t} 结束")
            break
    
    # 如果有日志记录器，保存数据并生成可视化
    if args.log and simulation.logger:
        simulation.logger.save_data()
        simulation.logger.generate_summary()
        simulation.logger.generate_visualizations()
        print(f"日志和可视化已保存至 {simulation.logger.session_dir}")
    
    # 如果渲染，等待用户关闭窗口
    if args.render:
        print("仿真完成，请关闭窗口继续...")
        plt.show(block=True)