import yaml
import random
import argparse
import math
from typing import List, Tuple, Dict, Any

class WarehouseConfigGenerator:
    def __init__(self, 
                 width: int = 13, 
                 height: int = 13,
                 num_agents: int = 4,
                 agent_capacity: int = 2,
                 num_tasks: int = 10,
                 obstacle_pattern: str = "shelves",  # "shelves", "random", "custom"
                 seed: int = None):
        """
        初始化仓库配置生成器
        
        参数:
            width: 地图宽度
            height: 地图高度
            num_agents: 代理数量
            agent_capacity: 每个代理的容量
            num_tasks: 要生成的任务数量
            obstacle_pattern: 障碍物模式 ("shelves", "random", "custom")
            seed: 随机数种子(可选)
        """
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.agent_capacity = agent_capacity
        self.num_tasks = num_tasks
        self.obstacle_pattern = obstacle_pattern
        
        if seed is not None:
            random.seed(seed)
        
        # 初始化配置字典
        self.config = {
            'agents': [],
            'map': {
                'dimensions': (width, height),
                'obstacles': [],
                'base_points': []
            },
            'tasks': []
        }

    def generate_shelves_obstacles(self) -> List[Tuple[int, int]]:
        """生成货架式障碍物模式"""
        obstacles = []
        
        # 生成多排货架
        num_aisles = (self.width - 1) // 4
        for aisle in range(num_aisles):
            x_pos = 1 + aisle * 4  # 每隔4个单位放置一排货架
            
            # 货架主体
            for y in range(2, self.height - 2, 4):
                obstacles.append((x_pos, y))
                
                # 为每个主体添加左右两侧的点，形成"十"字形状
                obstacles.append((x_pos, y - 1))
                # obstacles.append((x_pos, y + 1))
                
                # 添加左右两个单元的障碍物，形成完整的货架
                if x_pos + 1 < self.width:
                    obstacles.append((x_pos + 1, y))
                
        return obstacles

    def generate_random_obstacles(self, obstacle_ratio: float = 0.07) -> List[Tuple[int, int]]:
        """生成随机障碍物"""
        obstacles = []
        num_obstacles = int(self.width * self.height * obstacle_ratio)
        
        possible_positions = [(x, y) for x in range(self.width) for y in range(self.height)
                              if x not in [self.width-1, self.width-2] ]   # 除去最右侧两列
        
        # 预留出代理起始位置和基地点区域
        right_edge = [(self.width - 1, y) for y in range(self.height)]
        for pos in right_edge:
            if pos in possible_positions:
                possible_positions.remove(pos)
        
        selected_positions = random.sample(possible_positions, min(num_obstacles, len(possible_positions)))
        obstacles = selected_positions
        
        return obstacles

    def generate_agents(self) -> None:
        """
        生成代理信息，所有代理都初始化在base_points上
        确保代理起始位置尽可能分散在不同的base_points上
        """
        agents = []
        # 确保有基地点可用
        if not self.config['map']['base_points']:
            self.generate_base_points()
        
        base_points = self.config['map']['base_points']
        used_positions = set()  # 记录已使用的位置
        
        # 为每个代理生成起始位置
        for i in range(self.num_agents):
            # 优先选择未使用的基地点
            available_base_points = [point for point in base_points if point not in used_positions]
            
            if available_base_points:
                # 如果还有未使用的基地点，随机选择一个
                start_pos = random.choice(available_base_points)
            else:
                # 如果所有基地点都已使用，随机选择一个基地点（可能会有重叠）
                # 尝试选择使用次数最少的基地点
                position_counts = {}
                for pos in base_points:
                    position_counts[pos] = sum(1 for p in used_positions if p == pos)
                
                # 找出使用次数最少的基地点
                min_count = min(position_counts.values())
                least_used = [pos for pos, count in position_counts.items() if count == min_count]
                start_pos = random.choice(least_used)
                
                print(f"警告：代理数量 ({self.num_agents}) 超过基地点数量 ({len(base_points)})，"
                    f"代理 {i} 将与其他代理共享基地点")
            
            used_positions.add(start_pos)  # 标记该位置已被使用
            
            agent = {
                'start': start_pos,
                'name': f'agent{i}',
                'capacity': self.agent_capacity
            }
            agents.append(agent)
        
        self.config['agents'] = agents

    def _find_farthest_position(self, used_positions):
        """找到距离已使用位置最远的点"""
        max_min_distance = -1
        best_position = None
        
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) not in self.config['map']['obstacles']:
                    min_distance = float('inf')
                    for pos in used_positions:
                        dist = max(abs(x - pos[0]), abs(y - pos[1]))
                        min_distance = min(min_distance, dist)
                    
                    if min_distance > max_min_distance:
                        max_min_distance = min_distance
                        best_position = (x, y)
        
        return best_position

    def generate_base_points(self) -> None:
        """生成基地点，仅在地图最右侧一列，数量为代理数量的1.5倍"""
        # 确定基地点数量
        num_base_points = int(self.num_agents * 1.5)
        
        # 限制高度不超过地图高度减去边距
        available_height = self.height - 2  # 上下各预留1个单位
        
        # 确保基地点数量不超过可用高度
        if num_base_points > available_height:
            print(f"警告：基地点数量 ({num_base_points}) 超过可用高度 ({available_height})，将被限制")
            num_base_points = available_height
        
        # 生成基地点，只在最右侧一列 (x = self.width - 1)
        # 尽量使基地点垂直居中分布
        start_y = max(1, (self.height - num_base_points) // 2)
        
        base_points = []
        for i in range(num_base_points):
            y = start_y + i
            if y < self.height - 1:  # 确保不超出地图边界
                base_points.append((self.width - 1, y))
        
        self.config['map']['base_points'] = base_points

    def generate_tasks(self, use_poisson: bool = False, poisson_rate: float = 10) -> None:
        """
        生成任务列表，确保任务起点不重复
        
        参数:
            use_poisson: 是否使用泊松分布生成任务到达时间
            poisson_rate: 泊松分布的参数λ（平均每单位时间的任务数量）
        """
        tasks = []
        obstacles = set(self.config['map']['obstacles'])
        agent_positions = set(agent['start'] for agent in self.config['agents'])
        base_points = set(self.config['map']['base_points'])
        
        # 可用的任务起点 (避开障碍物、代理起始位置和基地点所在列)
        valid_positions = [(x, y) for x in range(self.width - 3)  # 不包括最右侧三列
                        for y in range(self.height) 
                        if (x, y) not in obstacles and (x, y) not in agent_positions]
        
        # 生成任务到达时间
        if use_poisson:
            # 使用泊松分布通过逆采样生成时间间隔
            arrival_times = []
            current_time = 0
            while len(arrival_times) < self.num_tasks:
                # 指数分布是泊松过程的时间间隔分布
                # 使用逆采样法：F^(-1)(u) = -ln(1-u)/λ
                u = random.random()
                inter_arrival_time = -math.log(1 - u) / poisson_rate
                current_time += inter_arrival_time
                arrival_times.append(int(current_time))  # 向下取整
            
            arrival_times.sort()
            print(arrival_times)
        else:
            # 使用原有策略：前半部分任务从0开始，后半部分随机
            arrival_times = [0] * (self.num_tasks // 2) + [random.randint(1, 15) for _ in range(self.num_tasks - self.num_tasks // 2)]
            arrival_times.sort()
        
        # 检查有效位置数量是否足够生成不重复的起点
        if len(valid_positions) < self.num_tasks:
            print(f"警告：有效位置数量 ({len(valid_positions)}) 少于任务数量 ({self.num_tasks})，"
                f"部分任务将使用随机起点")
        
        # 用于存储已使用的起点位置
        used_start_positions = set()
        
        for i in range(self.num_tasks):
            # 尝试选择未使用的起点
            available_positions = [pos for pos in valid_positions if pos not in used_start_positions]
            
            if available_positions:
                # 如果还有未使用的位置，随机选择一个
                start = random.choice(available_positions)
                used_start_positions.add(start)
            else:
                # 如果所有有效位置都已使用，生成一个随机位置（可能与障碍物重叠）
                print(f"警告：任务 {i} 无法找到未使用的起点位置，将使用随机位置")
                # 尝试在非障碍物、非代理起始位置的地方找一个位置
                possible_random_positions = [(x, y) for x in range(self.width - 1) 
                                            for y in range(self.height)
                                            if (x, y) not in obstacles 
                                            and (x, y) not in agent_positions]
                
                if possible_random_positions:
                    start = random.choice(possible_random_positions)
                else:
                    # 最坏情况：随机生成一个位置
                    start = (random.randint(0, self.width - 2), random.randint(0, self.height - 1))
                
                # 仍然添加到已使用集合中，避免再次选择相同位置
                used_start_positions.add(start)
            
            # 终点总是在基地点
            if base_points:
                goal = random.choice(list(base_points))
            else:
                # 如果没有基地点，使用最右侧的列
                goal_y = random.randint(1, self.height - 2)
                goal = (self.width - 1, goal_y)
            
            # 使用生成的到达时间
            start_time = arrival_times[i]
            
            task = {
                'start_time': start_time,
                'start': start,
                'goal': goal,
                'task_name': f'task{i}'
            }
            tasks.append(task)
        
        # 按开始时间排序
        tasks.sort(key=lambda x: x['start_time'])
        self.config['tasks'] = tasks

    def generate_config(self, use_poisson=False, poisson_rate:float =5) -> Dict[str, Any]:
        """生成完整的配置"""
        # 生成障碍物
        if self.obstacle_pattern == "shelves":
            self.config['map']['obstacles'] = self.generate_shelves_obstacles()
        elif self.obstacle_pattern == "random":
            self.config['map']['obstacles'] = self.generate_random_obstacles()
        
        # 生成基地点 (仅在最右侧一列)
        self.generate_base_points()
        
        # 生成代理
        self.generate_agents()
        
        # 生成任务
        self.generate_tasks(use_poisson=use_poisson, poisson_rate=poisson_rate)
        
        return self.config

    def save_config(self, filepath: str) -> None:
        """保存配置到YAML文件"""
        with open(filepath, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=None)
            print(f"配置已保存到: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='生成仓库环境配置文件')
    
    parser.add_argument('--width', type=int, default=36, help='地图宽度')
    parser.add_argument('--height', type=int, default=20, help='地图高度')
    parser.add_argument('--agents', type=int, default=4, help='代理数量')
    parser.add_argument('--capacity', type=int, default=3, help='每个代理的容量')
    parser.add_argument('--tasks', type=int, default=10, help='任务数量')
    parser.add_argument('--obstacles', type=str, default='shelves', 
                        choices=['shelves', 'random'], help='障碍物模式')
    parser.add_argument('--output', type=str, default='input_r10_t100_w36_t20.yaml', 
                        help='输出文件路径')
    parser.add_argument('--seed', type=int, default=None, help='随机数种子')
    parser.add_argument('--UsePoisson', action='store_true',
                        help='是否使用泊松分布生成任务到达时间')
    parser.add_argument('--poisson_rate', type=float, default=5, help='泊松分布参数λ (平均每单位时间任务数)')
    
    args = parser.parse_args()
    
    # 创建生成器并生成配置
    generator = WarehouseConfigGenerator(
        width=args.width,
        height=args.height,
        num_agents=args.agents,
        agent_capacity=args.capacity,
        num_tasks=args.tasks,
        obstacle_pattern=args.obstacles,
        seed=args.seed,
    )
    
    generator.generate_config(use_poisson=args.UsePoisson, poisson_rate=args.poisson_rate)
    args.output = 'env' + f"r{args.agents}_t{args.tasks}_w{args.width}_h{args.height}.yaml"
    generator.save_config(args.output)


if __name__ == "__main__":
    main()