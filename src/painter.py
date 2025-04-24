"""
提供仿真过程中的可视化功能，支持动态更新地图和智能体位置
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import time
import numpy as np
from copy import deepcopy
from typing import TYPE_CHECKING, List, Dict, Tuple, Set

if TYPE_CHECKING:
    from .agent import Agent
    from .map import Map
    from .simulation import Simulation

LASER_RANGE = 2  # 假设激光范围为3，实际值应根据具体情况设置
class Painter:
    """
    用于可视化仿真过程中的地图和智能体状态
    支持实时更新和动画保存
    """
    def __init__(self, figsize=None, dpi=150):
        """
        初始化Painter类
        
        Parameters:
            figsize: 图像大小，默认为根据地图大小自动调整
            dpi: 图像分辨率
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
        plt.rcParams['axes.unicode_minus'] = False    # 显示负号
        
        self.figsize = figsize
        self.dpi = dpi
        self.fig = None
        self.ax = None
        self.task_ax = None  # 添加任务统计子图
        self.agents_scatter = {}  # 存储智能体的散点对象
        self.time_text = None     # 用于显示时间步的文本对象
        self.last_known_area = 0  # 上一次的已知区域大小，用于检测变化
        self.info_text_obj = None # 信息文本对象
        
        # 动画相关
        self.is_animation_running = False
        self.animation_obj = None
        
        # 存储元素的容器，便于移除和更新
        self.frontier_markers = []    # 存储前沿点标记对象
        self.centroid_markers = []    # 存储中心点标记对象
        self.centroid_boxes = []      # 存储中心点边界框
        self.centroid_texts = []      # 存储中心点面积文本
        self.task_markers = []        # 存储任务标记
        self.task_texts = []          # 存储任务标签
        
        # 任务统计数据
        self.time_steps = []
        self.discovered_tasks = []
        self.completed_tasks = []

    def start_animation(self, sim_steps: int, local_map: "Map", agents: List["Agent"], 
                        realmap: "Map", simulation:"Simulation", task_planner, path_planner, 
                        interval=200, save_path=None, render=False):
            """
            启动动画仿真，支持实时显示同时能完整保存
            
            Parameters:
                sim_steps: 仿真总步数
                local_map: 初始局部地图
                agents: 智能体列表
                realmap: 真实地图
                simulation: 仿真对象
                task_planner: 任务规划器
                path_planner: 路径规划器
                interval: 帧间隔(毫秒)
                save_path: 若提供，则保存动画到指定路径
                render: 是否在仿真过程中实时显示，默认为False
            """
            # 只在需要渲染时初始化图形界面
            if render:
                self.setup(local_map, agents)
            else:
                # 不渲染时只设置基本参数，不创建图形窗口
                print("运行仿真而不显示动画...")
                if self.figsize is None:
                    aspect_ratio = local_map.height / local_map.width
                    fig_width = min(20, max(12, local_map.width/4))
                    fig_height = fig_width * aspect_ratio * 1.3  # 增加高度以容纳任务统计图
                    self.figsize = (fig_width, fig_height)
                self.last_known_area = 0
            
            self.is_animation_running = True
            messages = {agent.name: agent.message for agent in agents}
            
            # 保存每一帧的状态，用于后续生成完整的动画
            self.frame_data = []
            
            # 复制初始状态用于保存
            if save_path:
                import copy
                frame = {
                    't': -1,
                    'map': copy.deepcopy(local_map), 
                    'agents': [{
                        'name': agent.name,
                        'pos': agent.pos,
                        'picked_tasks': agent.picked_tasks[:] if hasattr(agent, 'picked_tasks') else []
                    } for agent in agents]
                }
                self.frame_data.append(frame)
            
            # 执行初始扫描
            print('\n初始化扫描...')
            for agent in agents:
                agent.execute(realmap, -1)
                messages[agent.name] = agent.publish(-1)
            
            # 更新初始地图
            simulation.update_map_infos(local_map, realmap, messages)
            
            # 定义单步仿真函数，整合相同的仿真逻辑
            def simulate_step(t):
                """执行一个仿真步骤并返回是否应该结束仿真"""
                print(f'\n** 仿真步骤 {t} **')
                
                # 1. 更新地图和任务信息
                simulation.update_map_infos(local_map, realmap, messages)
                print(f"debug: u_tasks:{local_map.u_tasks}")
                task_planner.assign_tasks(agents, local_map, t)
                task_planner.assign_frontiers(agents, local_map, t)
                
                # 2. 规划路径
                path_planner.plan_paths(agents, local_map)
                
                # 3. 更新智能体
                for agent in agents:
                    agent.update(t)
                
                # 4. 执行仿真步骤
                for agent in agents:
                    agent.execute(realmap, t)
                    messages[agent.name] = agent.publish(t)
                
                # 5. 如果需要渲染，更新可视化
                if render:
                    self.update(t, local_map, agents)
                
                # 6. 保存当前帧状态用于生成完整动画
                if save_path:
                    import copy
                    frame = {
                        't': t,
                        'map': copy.deepcopy(local_map), 
                        'agents': [{
                            'name': agent.name,
                            'pos': agent.pos,
                            'picked_tasks': agent.picked_tasks[:] if hasattr(agent, 'picked_tasks') else []
                        } for agent in agents]
                    }
                    self.frame_data.append(frame)
                
                # 7. 检查是否应该结束仿真
                if len(local_map.c_tasks) == len(realmap._tasks):
                    print("所有任务已完成，仿真结束！")
                    return True
                
                return False
            
            # 根据是否渲染选择不同的仿真方式
            if render:
                # 使用实时渲染的动画函数和FuncAnimation
                def animate(t):
                    if not self.is_animation_running:
                        return
                    
                    try:
                        should_stop = simulate_step(t)
                        if should_stop:
                            self.is_animation_running = False
                    except Exception as e:
                        print(f"动画更新出错: {e}")
                        import traceback
                        traceback.print_exc()
                        self.is_animation_running = False
                
                # 创建动画
                self.animation_obj = animation.FuncAnimation(
                    self.fig, animate, frames=range(sim_steps),
                    interval=interval, repeat=False, blit=False
                )
                
                # 显示动画
                plt.show(block=True)  # 使用阻塞模式确保显示完整
            else:
                # 不渲染时直接执行仿真循环
                print("开始仿真计算...")
                try:
                    for t in range(sim_steps):
                        if not self.is_animation_running:
                            break
                        
                        should_stop = simulate_step(t)
                        if should_stop:
                            break
                    
                    print("仿真计算完成!")
                except Exception as e:
                    print(f"仿真计算出错: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 仿真结束后，如果需要保存，使用收集的帧数据创建新的动画并保存
            save_st = time.time()
            if save_path and self.frame_data:
                self._save_complete_animation(save_path, interval)
                print(f"完整动画保存完成，耗时: {time.time() - save_st:.2f}秒")


# ========================
# 辅助函数 ========================
# ========================
    def setup(self, local_map: "Map", agents: List["Agent"]):
        """
        设置初始图形
        
        Parameters:
            local_map: 要显示的地图
            agents: 智能体列表
        """
        # 如果未指定figsize，则根据地图尺寸自动调整
        if self.figsize is None:
            aspect_ratio = local_map.height / local_map.width
            fig_width = min(20, max(12, local_map.width/4))
            fig_height = fig_width * aspect_ratio * 1.3  # 增加高度以容纳任务统计图
            self.figsize = (fig_width, fig_height)
        
        # 创建图形和坐标轴
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # 创建主地图子图和任务统计子图
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1])
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.task_ax = self.fig.add_subplot(gs[1, 0])
        
        self.ax.set_title('仿真动态可视化', fontsize=16, fontweight='bold')
        
        # 设置坐标轴范围
        self.ax.set_xlim(-0.5, local_map.width - 0.5)
        self.ax.set_ylim(-0.5, local_map.height - 0.5)
        self.ax.set_aspect('equal')  # 保持网格单元为正方形
        
        # 设置刻度间隔
        tick_spacing = max(1, min(5, local_map.width // 10))
        x_ticks = range(0, local_map.width, tick_spacing)
        y_ticks = range(0, local_map.height, tick_spacing)
        self.ax.set_xticks(x_ticks)
        self.ax.set_yticks(y_ticks)
        self.ax.set_xlabel('X坐标', fontsize=12)
        self.ax.set_ylabel('Y坐标', fontsize=12)
        
        # 绘制网格线
        for x in range(local_map.width + 1):
            self.ax.axvline(x=x - 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        for y in range(local_map.height + 1):
            self.ax.axhline(y=y - 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # 在右上角添加时间步显示
        self.time_text = self.ax.text(0.98, 0.98, 'Time: -1', transform=self.ax.transAxes,
                                 fontsize=12, ha='right', va='top',
                                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
        
        # 在左上角添加信息显示
        info_text = f"地图大小: {local_map.width}×{local_map.height}\n"
        info_text += f"已知区域: {local_map.known_area}/{local_map.total_area} ({local_map.known_area/local_map.total_area*100:.1f}%)\n"
        info_text += f"前沿点数: {len(local_map.frontiers)}\n"
        info_text += f"中心点数: {len(local_map.centroids)}"
        
        self.info_text_obj = self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                                     fontsize=10, verticalalignment='top', 
                                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        



        # 初始化智能体标记和探照范围
        self.agent_laser_ranges = {}  # 存储智能体的探照范围显示对象
        
        for agent in agents:
            # 使用不同颜色区分不同智能体
            agent_color = self._get_agent_color(agent.name)
            
            # 创建智能体的探照范围
            laser_half_width = agent.laser_range
            laser_rect = patches.Rectangle(
                (agent.pos[0] - laser_half_width - 0.5, agent.pos[1] - laser_half_width - 0.5),
                2 * laser_half_width + 1, 2 * laser_half_width + 1,
                linewidth=1.5, edgecolor=agent_color, facecolor=agent_color, alpha=0.1,
                linestyle='--', zorder=10
            )
            self.ax.add_patch(laser_rect)
            self.agent_laser_ranges[agent.name] = laser_rect
            
            # 创建智能体的散点标记
            scatter = self.ax.scatter(agent.pos[0], agent.pos[1], 
                                c=agent_color, marker='o', s=120, zorder=10,
                                edgecolors='black', linewidths=1.5)
            
            # 添加智能体名称标签
            self.ax.text(agent.pos[0], agent.pos[1] - 0.3, agent.name, 
                    fontsize=9, ha='center', va='top', 
                    color='black', fontweight='bold', zorder=10)
            
            # 存储智能体的散点对象，以便后续更新
            self.agents_scatter[agent.name] = scatter
            
            # 绘制初始地图
            self._draw_map(local_map)
        
        # 绘制图例
        self._create_legend(local_map)
        
        # 设置任务统计子图
        self.task_ax.set_title('任务统计', fontsize=14)
        self.task_ax.set_xlabel('时间步', fontsize=12)
        self.task_ax.set_ylabel('任务数量', fontsize=12)
        self.task_ax.grid(True, linestyle='--', alpha=0.7)
        
        # 初始化任务统计数据
        self.time_steps = [-1]
        self.discovered_tasks = [len(local_map.tasks_seen)]
        self.completed_tasks = [len(local_map.c_tasks)]
        
        # 绘制初始任务统计图
        self._update_task_stats(local_map, -1)
        
        # 紧凑布局
        plt.tight_layout()

    def update(self, t: int, local_map: "Map", agents: List["Agent"]):
        """
        更新可视化：更新地图和智能体位置
        
        Parameters:
            t: 当前时间步
            local_map: 更新后的地图
            agents: 更新后的智能体列表
        """
        # 更新时间步显示
        self.time_text.set_text(f'Time: {t}')
        
        # 保存对智能体的引用，便于更新任务显示
        local_map.agents = agents
        
        # 用标志标记是否已更新地图
        map_updated = False
        
        # 检查地图是否有变化
        if local_map.known_area != self.last_known_area:
            # 更新地图
            self._draw_map(local_map)
            self.last_known_area = local_map.known_area
            map_updated = True
            
            # 更新信息文本
            info_text = f"地图大小: {local_map.width}×{local_map.height}\n"
            info_text += f"已知区域: {local_map.known_area}/{local_map.total_area} ({local_map.known_area/local_map.total_area*100:.1f}%)\n"
            info_text += f"前沿点数: {len(local_map.frontiers)}\n"
            info_text += f"中心点数: {len(local_map.centroids)}"
            self.info_text_obj.set_text(info_text)
        elif not map_updated:  # 仅当地图尚未更新时才更新前沿点和中心点
            # 即使已知区域没变，也需要更新前沿点和中心点的显示
            self._update_frontiers_and_centroids(local_map)
            
            # 更新前沿点和中心点的信息
            info_text = f"地图大小: {local_map.width}×{local_map.height}\n"
            info_text += f"已知区域: {local_map.known_area}/{local_map.total_area} ({local_map.known_area/local_map.total_area*100:.1f}%)\n"
            info_text += f"前沿点数: {len(local_map.frontiers)}\n"
            info_text += f"中心点数: {len(local_map.centroids)}"
            self.info_text_obj.set_text(info_text)
        
        # 更新智能体位置和任务信息
        # 更新智能体位置和任务信息
        for agent in agents:
            if agent.name in self.agents_scatter:
                # 获取智能体散点对象
                scatter = self.agents_scatter[agent.name]
                # 更新位置
                scatter.set_offsets([agent.pos[0], agent.pos[1]])
                
                # 更新探照范围 - 修复此部分
                if agent.name in self.agent_laser_ranges:
                    # 先移除旧的激光范围矩形
                    old_rect = self.agent_laser_ranges[agent.name]
                    if old_rect in self.ax.patches:
                        old_rect.remove()
                    
                    # 获取智能体颜色
                    agent_color = self._get_agent_color(agent.name)
                    
                    # 创建新的探照范围矩形
                    laser_half_width = agent.laser_range
                    laser_rect = patches.Rectangle(
                        (agent.pos[0] - laser_half_width - 0.5, agent.pos[1] - laser_half_width - 0.5),
                        2 * laser_half_width + 1, 2 * laser_half_width + 1,
                        linewidth=1.5, edgecolor=agent_color, facecolor=agent_color, alpha=0.1,
                        linestyle='--', zorder=10
                    )
                    self.ax.add_patch(laser_rect)
                    self.agent_laser_ranges[agent.name] = laser_rect

                # 查找并更新智能体名称标签和拾取的任务信息
                agent_label_found = False
                
                # 检查是否已有此智能体的标签
                for text in self.ax.texts:
                    # 检查文本是否属于该智能体（考虑可能已经附加了任务信息）
                    if text.get_text() == agent.name or text.get_text().startswith(agent.name + "\n"):
                        # 准备新的标签文本
                        if agent.picked_tasks:
                            task_text = f"{agent.name}\n[{','.join(agent.picked_tasks)}]"
                        else:
                            task_text = agent.name
                        
                        # 更新文本和位置
                        text.set_text(task_text)
                        text.set_position((agent.pos[0], agent.pos[1] - 0.3))
                        agent_label_found = True
                        break
                
                # 如果找不到标签（可能被删除），则创建新标签
                if not agent_label_found:
                    if agent.picked_tasks:
                        task_text = f"{agent.name}\n[{','.join(agent.picked_tasks)}]"
                    else:
                        task_text = agent.name
                    
                    self.ax.text(agent.pos[0], agent.pos[1] - 0.3, task_text, 
                            fontsize=9, ha='center', va='top', 
                            color='black', fontweight='bold', zorder=10)
        
        # 更新任务统计
        self._update_task_stats(local_map, t)
        
        # 刷新图形
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except (ValueError, RuntimeError) as e:
            print(f"图形更新出错（可能是窗口已关闭）: {e}")

    def _update_task_stats(self, local_map: "Map", t: int):
        """更新任务统计图"""
        # 更新数据
        if t not in self.time_steps:
            self.time_steps.append(t)
            self.discovered_tasks.append(len(local_map.tasks_seen))
            self.completed_tasks.append(len(local_map.c_tasks))
        else:
            # 更新现有数据点
            idx = self.time_steps.index(t)
            self.discovered_tasks[idx] = len(local_map.tasks_seen)
            self.completed_tasks[idx] = len(local_map.c_tasks)
        
        # 清除当前任务统计图
        self.task_ax.clear()
        
        # 重新绘制统计图
        self.task_ax.set_title('任务统计', fontsize=14)
        self.task_ax.set_xlabel('时间步', fontsize=12)
        self.task_ax.set_ylabel('任务数量', fontsize=12)
        
        # 绘制折线图
        self.task_ax.plot(self.time_steps, self.discovered_tasks, 'b-', marker='o', label='已发现任务')
        self.task_ax.plot(self.time_steps, self.completed_tasks, 'r-', marker='s', label='已完成任务')
        
        # 添加网格和图例
        self.task_ax.grid(True, linestyle='--', alpha=0.7)
        self.task_ax.legend(loc='upper left')
        
        # 设置x轴刻度，确保为整数
        self.task_ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    def _update_frontiers_and_centroids(self, local_map: "Map"):
        """
        专门更新前沿点和中心点的显示，不重绘整个地图
        
        Parameters:
            local_map: 当前地图
        """
        # 清除之前的前沿点和中心点
        self._clear_dynamic_elements()
        
        # 绘制新的前沿点
        if local_map.frontiers:
            for x, y in local_map.frontiers:
                marker = plt.Circle((x, y), 0.15, color='yellow', alpha=0.7, zorder=4)
                self.ax.add_patch(marker)
                self.frontier_markers.append(marker)
        
        # 绘制新的中心点及其边界框
        if local_map.centroids:
            for centroid, data in local_map.centroids.items():
                # 绘制中心点（品红色星形）
                star_marker = self.ax.scatter(centroid[0], centroid[1], c='magenta', marker='*', 
                                         s=200, zorder=7)
                self.centroid_markers.append(star_marker)
                
                # 绘制边界框
                try:
                    bbox = data['bound_box']
                    x_min, y_min = bbox[0]
                    x_max, y_max = bbox[2]
                    width = x_max - x_min
                    height = y_max - y_min
                    rect = patches.Rectangle((x_min - 0.5, y_min - 0.5), width + 1, height + 1, 
                                          linewidth=2, edgecolor='magenta', facecolor='none', 
                                          linestyle='--', zorder=6)
                    self.ax.add_patch(rect)
                    self.centroid_boxes.append(rect)
                    
                    # 显示未知面积
                    text = self.ax.text(centroid[0] + 0.5, centroid[1] + 0.5, f"面积: {data['area']}", 
                                  fontsize=9, color='magenta', weight='bold', zorder=7,
                                  bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
                    self.centroid_texts.append(text)
                except (IndexError, KeyError) as e:
                    print(f"无法绘制中心点 {centroid} 的边界框: {e}")
        
        # 更新任务显示
        self._update_tasks(local_map)
    
    def _update_tasks(self, local_map: "Map"):
        """更新任务显示，只显示尚未被拾取的任务"""
        # 清除之前的任务标记
        for marker in self.task_markers:
            try:
                if marker in self.ax.patches:
                    marker.remove()
            except (ValueError, RuntimeError) as e:
                pass
        self.task_markers = []
        
        for text in self.task_texts:
            try:
                if text in self.ax.texts:
                    text.remove()
            except (ValueError, RuntimeError) as e:
                pass
        self.task_texts = []
        
        # 收集所有被拾取的任务
        all_picked_tasks = set()
        for agent in local_map.agents if hasattr(local_map, 'agents') else []:
            all_picked_tasks.update(agent.picked_tasks)
        
        # 绘制新的任务（仅显示未被拾取且未完成的任务）
        for task in local_map.tasks_seen:
            task_name = task['task_name']
            # 如果任务未完成且未被任何agent拾取
            if task_name not in local_map.c_tasks and task_name not in all_picked_tasks:
                x, y = task['start']
                circle = plt.Circle((x, y), 0.4, color='green', alpha=0.8, zorder=5)
                self.ax.add_patch(circle)
                self.task_markers.append(circle)
                
                text = self.ax.text(x, y, task_name, fontsize=8, color='black', weight='bold', 
                            ha='center', va='center', zorder=6)
                self.task_texts.append(text)
    
    def _clear_dynamic_elements(self):
        """清除所有动态元素（前沿点、中心点、任务等）"""
        # 清除前沿点
        for marker in self.frontier_markers:
            try:
                if marker in self.ax.patches:  # 确保标记仍在图表中
                    marker.remove()
            except (ValueError, RuntimeError) as e:
                # 忽略"标记已被移除"的错误
                pass
        self.frontier_markers = []
        
        # 清除中心点
        for marker in self.centroid_markers:
            try:
                # 检查散点图集合是否仍在图中
                if marker in self.ax.collections:
                    marker.remove()
            except (ValueError, RuntimeError) as e:
                pass
        self.centroid_markers = []
        
        # 清除边界框
        for rect in self.centroid_boxes:
            try:
                if rect in self.ax.patches:
                    rect.remove()
            except (ValueError, RuntimeError) as e:
                pass
        self.centroid_boxes = []
        
        # 清除文本
        for text in self.centroid_texts:
            try:
                if text in self.ax.texts:
                    text.remove()
            except (ValueError, RuntimeError) as e:
                pass
        self.centroid_texts = []
    
    

    def _save_complete_animation(self, filename, interval=200):
        """使用收集的帧数据创建一个新的动画并保存"""
        if not self.frame_data:
            print("没有可用的帧数据来保存动画")
            return
            
        print(f"准备保存完整动画到: {filename}，共 {len(self.frame_data)} 帧")
        
        # 创建新的图表用于保存，包含任务进度子图
        save_fig = plt.figure(figsize=self.figsize, dpi=150)
        
        # 修复：GridSpec应该只有2行，而不是5行，且height_ratios列表长度应与行数匹配
        gs = plt.GridSpec(2, 1, height_ratios=[3, 1], figure=save_fig)
        
        # 主地图子图
        save_ax = save_fig.add_subplot(gs[0, 0])
        save_ax.set_title('探索过程回放', fontsize=16, fontweight='bold')
        
        # 任务进度子图
        save_task_ax = save_fig.add_subplot(gs[1, 0]) 
        save_task_ax.set_title('任务进度', fontsize=14)
        save_task_ax.set_xlabel('仿真步骤', fontsize=10)
        save_task_ax.set_ylabel('任务数量', fontsize=10)
        save_task_ax.grid(True, linestyle='--', alpha=0.7)
        
        # 设置坐标轴范围
        first_frame = self.frame_data[0]
        local_map = first_frame['map']
        save_ax.set_xlim(-0.5, local_map.width - 0.5)
        save_ax.set_ylim(-0.5, local_map.height - 0.5)
        save_ax.set_aspect('equal')
            
        # 设置刻度间隔
        tick_spacing = max(1, min(5, local_map.width // 10))
        x_ticks = range(0, local_map.width, tick_spacing)
        y_ticks = range(0, local_map.height, tick_spacing)
        save_ax.set_xticks(x_ticks)
        save_ax.set_yticks(y_ticks)
        save_ax.set_xlabel('X坐标', fontsize=12)
        save_ax.set_ylabel('Y坐标', fontsize=12)
        
        # 绘制网格线
        for x in range(local_map.width + 1):
            save_ax.axvline(x=x - 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        for y in range(local_map.height + 1):
            save_ax.axhline(y=y - 0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # 时间步显示
        time_text = save_ax.text(0.98, 0.98, 'Time: -1', transform=save_ax.transAxes,
                            fontsize=12, ha='right', va='top',
                            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
        
        # 信息显示
        info_text = save_ax.text(0.02, 0.98, "", transform=save_ax.transAxes,
                            fontsize=10, verticalalignment='top', 
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        # 任务统计文本
        task_stats_text = save_task_ax.text(
            0.98, 0.9, 
            f'已发现: 0 | 已完成: 0', 
            transform=save_task_ax.transAxes,
            ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        
        # 初始化任务进度数据
        time_steps = []
        discovered_tasks = []
        completed_tasks = []
        
        agents_scatter = {}
        agent_texts = {}
        agent_laser_ranges = {}
        first_agents = first_frame['agents']
        
        for agent_data in first_agents:
            agent_name = agent_data['name']
            # 使用不同颜色区分不同智能体
            agent_color = self._get_agent_color(agent_name)
            
            # 创建智能体的探照范围
            laser_half_width = LASER_RANGE  # 使用全局常量或从agent数据中获取
            laser_rect = patches.Rectangle(
                (agent_data['pos'][0] - laser_half_width - 0.5, agent_data['pos'][1] - laser_half_width - 0.5),
                2 * laser_half_width + 1, 2 * laser_half_width + 1,
                linewidth=1.5, edgecolor=agent_color, facecolor=agent_color, alpha=0.1,
                linestyle='--', zorder=3
            )
            save_ax.add_patch(laser_rect)
            agent_laser_ranges[agent_name] = laser_rect
            
            # 创建智能体的散点标记
            scatter = save_ax.scatter(agent_data['pos'][0], agent_data['pos'][1], 
                                c=agent_color, marker='o', s=120, zorder=10,
                                edgecolors='black', linewidths=1.5)
            # 添加智能体名称标签
            text = save_ax.text(agent_data['pos'][0], agent_data['pos'][1] - 0.3, agent_name, 
                        fontsize=9, ha='center', va='top', 
                        color='black', fontweight='bold', zorder=10)
            
            agents_scatter[agent_name] = scatter
            agent_texts[agent_name] = text
        
        # 创建任务进度线图
        discovered_line, = save_task_ax.plot(
            [], [], 'g-', linewidth=2, label='已发现任务')
        completed_line, = save_task_ax.plot(
            [], [], 'r-', linewidth=2, label='已完成任务')
        
        # 添加任务进度图例
        save_task_ax.legend(loc='upper left')
        
        # 动态元素容器
        frontier_markers = []
        centroid_markers = []
        centroid_boxes = []
        centroid_texts = []
        task_markers = []
        task_texts = []
        
        def update_save(i):
            frame = self.frame_data[i]
            t = frame['t']
            local_map = frame['map']
            
            # 更新时间步
            time_text.set_text(f'Time: {t}')
            
            # 更新信息文本
            info_text_str = f"地图大小: {local_map.width}×{local_map.height}\n"
            info_text_str += f"已知区域: {local_map.known_area}/{local_map.total_area} ({local_map.known_area/local_map.total_area*100:.1f}%)\n"
            info_text_str += f"前沿点数: {len(local_map.frontiers)}\n"
            info_text_str += f"中心点数: {len(local_map.centroids)}"
            info_text.set_text(info_text_str)
            
            # 清除之前的地图元素
            for patch in save_ax.patches:
                patch.remove()
            
            # 清除动态元素
            for marker in frontier_markers:
                marker.remove() if marker in save_ax.patches else None
            frontier_markers.clear()
            
            for marker in centroid_markers:
                marker.remove() if marker in save_ax.collections else None
            centroid_markers.clear()
            
            for rect in centroid_boxes:
                rect.remove() if rect in save_ax.patches else None
            centroid_boxes.clear()
            
            for text in centroid_texts:
                text.remove() if text in save_ax.texts else None
            centroid_texts.clear()
            
            for marker in task_markers:
                marker.remove() if marker in save_ax.patches else None
            task_markers.clear()
            
            for text in task_texts:
                text.remove() if text in save_ax.texts else None
            task_texts.clear()
            
            # 收集地图元素
            unknown_cells = []
            free_cells = []
            obstacle_cells = []
            base_cells = []
            
            for y in range(local_map.height):
                for x in range(local_map.width):
                    cell_value = local_map.map[y][x]
                    if cell_value == -1:        # 未知区域
                        unknown_cells.append((x, y))
                    elif cell_value == 0:       # 自由空间
                        free_cells.append((x, y))
                    elif cell_value == 1:       # 障碍物
                        obstacle_cells.append((x, y))
                    elif cell_value == 2:       # 基地点
                        base_cells.append((x, y))
            
            # 绘制自由空间
            for x, y in free_cells:
                save_ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='white', 
                                        edgecolor='lightgray', linewidth=0.5))
            
            # 绘制未知区域
            for x, y in unknown_cells:
                save_ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='gray', 
                                        alpha=0.5))
            
            # 绘制障碍物
            for x, y in obstacle_cells:
                save_ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='black'))
            
            # 绘制基地点
            for x, y in base_cells:
                save_ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='royalblue', 
                                        alpha=0.8))
            
            # 绘制任务
            # 收集所有被拾取的任务
            all_picked_tasks = set()
            for agent_data in frame['agents']:
                all_picked_tasks.update(agent_data['picked_tasks'])
            
            for task in local_map.tasks_seen:
                task_name = task['task_name']
                # 如果任务未完成且未被任何agent拾取
                if task_name not in local_map.c_tasks and task_name not in all_picked_tasks:
                    x, y = task['start']
                    circle = plt.Circle((x, y), 0.4, color='green', alpha=0.8, zorder=5)
                    save_ax.add_patch(circle)
                    task_markers.append(circle)
                    
                    text = save_ax.text(x, y, task_name, fontsize=8, color='black', weight='bold', 
                                ha='center', va='center', zorder=6)
                    task_texts.append(text)
            
            # 绘制前沿点
            if local_map.frontiers:
                for x, y in local_map.frontiers:
                    marker = plt.Circle((x, y), 0.15, color='yellow', alpha=0.7, zorder=4)
                    save_ax.add_patch(marker)
                    frontier_markers.append(marker)
            
            # 绘制中心点
            if local_map.centroids:
                for centroid, data in local_map.centroids.items():
                    # 绘制中心点
                    star_marker = save_ax.scatter(centroid[0], centroid[1], c='magenta', marker='*', 
                                            s=200, zorder=7)
                    centroid_markers.append(star_marker)
                    
                    # 绘制边界框
                    try:
                        bbox = data['bound_box']
                        x_min, y_min = bbox[0]
                        x_max, y_max = bbox[2]
                        width = x_max - x_min
                        height = y_max - y_min
                        rect = plt.Rectangle((x_min - 0.5, y_min - 0.5), width + 1, height + 1, 
                                        linewidth=2, edgecolor='magenta', facecolor='none', 
                                        linestyle='--', zorder=6)
                        save_ax.add_patch(rect)
                        centroid_boxes.append(rect)
                        
                        # 显示未知面积
                        text = save_ax.text(centroid[0] + 0.5, centroid[1] + 0.5, f"面积: {data['area']}", 
                                    fontsize=9, color='magenta', weight='bold', zorder=7,
                                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
                        centroid_texts.append(text)
                    except (IndexError, KeyError) as e:
                        print(f"无法绘制中心点 {centroid} 的边界框: {e}")
                        
            for agent_data in frame['agents']:
                agent_name = agent_data['name']
                if agent_name in agents_scatter:
                    # 更新散点位置
                    agents_scatter[agent_name].set_offsets([agent_data['pos'][0], agent_data['pos'][1]])
                    
                    # 更新探照范围
                    if agent_name in agent_laser_ranges:
                        laser_rect = agent_laser_ranges[agent_name]
                        laser_half_width =  LASER_RANGE # 使用全局常量或从agent数据中获取
                        laser_rect.set_xy((agent_data['pos'][0] - laser_half_width - 0.5, 
                                        agent_data['pos'][1] - laser_half_width - 0.5))
                    
                    # 更新文本位置和内容
                    if agent_data['picked_tasks']:
                        task_text = f"{agent_name}\n[{','.join(agent_data['picked_tasks'])}]"
                    else:
                        task_text = agent_name
                        
                    agent_texts[agent_name].set_text(task_text)
                    agent_texts[agent_name].set_position((agent_data['pos'][0], agent_data['pos'][1] - 0.3))
            
            # 更新任务进度数据
            time_steps.append(t)
            discovered_count = len(local_map.tasks_seen)
            completed_count = len(local_map.c_tasks)
            discovered_tasks.append(discovered_count)
            completed_tasks.append(completed_count)
            
            # 更新线图数据
            discovered_line.set_data(time_steps, discovered_tasks)
            completed_line.set_data(time_steps, completed_tasks)
            
            # 动态调整X轴范围，确保所有点都可见
            x_min = max(0, min(time_steps) - 2) if time_steps else 0
            x_max = max(time_steps) + 2 if time_steps else 100
            save_task_ax.set_xlim(x_min, x_max)
            
            # 动态调整Y轴范围，确保所有点都可见，并留出一些空间
            y_max = max(max(discovered_tasks) + 2, 10) if discovered_tasks else 10
            save_task_ax.set_ylim(0, y_max)
            
            # 更新任务统计文本
            task_stats_text.set_text(
                f'已发现: {discovered_count} | 已完成: {completed_count}'
            )
        
        # 创建保存用的动画对象
        save_anim = animation.FuncAnimation(
            save_fig, update_save, frames=len(self.frame_data),
            interval=interval, repeat=False, blit=False
        )
        
        # 保存动画
        print(f"开始保存完整动画...")
        try:
            # 根据文件扩展名选择Writer
            if filename.endswith('.mp4'):
                writer = animation.FFMpegWriter(fps=1000/interval, 
                                            metadata=dict(artist='POMAPD'), 
                                            bitrate=3000)  # 更高的比特率
            elif filename.endswith('.gif'):
                writer = animation.PillowWriter(fps=1000/interval)
            else:
                print(f"不支持的文件格式: {filename}，将默认保存为MP4")
                filename = filename.split('.')[0] + '.mp4'
                writer = animation.FFMpegWriter(fps=1000/interval, 
                                            metadata=dict(artist='POMAPD'), 
                                            bitrate=3000)
            
            # 保存动画
            save_anim.save(
                filename, 
                writer=writer,
                dpi=150,  # 更高的DPI
                savefig_kwargs={'facecolor': 'white', 'bbox_inches': 'tight'}
            )
            print(f"动画已成功保存到: {filename}")
        except Exception as e:
            print(f"保存动画时出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 关闭保存用的图表
        plt.close(save_fig)

    def stop_animation(self):
        """停止动画"""
        self.is_animation_running = False
        if self.animation_obj:
            self.animation_obj.event_source.stop()
    
    def _draw_map(self, local_map: "Map"):
        """绘制地图所有元素"""
        # 清除之前的地图元素（保留智能体和文本）
        for patch in self.ax.patches:
            patch.remove()
        
        # 清除所有动态元素
        self._clear_dynamic_elements()
        
        # 收集不同类型的单元格
        unknown_cells = []
        free_cells = []
        obstacle_cells = []
        base_cells = []
        
        for y in range(local_map.height):
            for x in range(local_map.width):
                cell_value = local_map.map[y][x]
                if cell_value == -1:        # 未知区域
                    unknown_cells.append((x, y))
                elif cell_value == 0:       # 自由空间
                    free_cells.append((x, y))
                elif cell_value == 1:       # 障碍物
                    obstacle_cells.append((x, y))
                elif cell_value == 2:       # 基地点
                    base_cells.append((x, y))
        
        # 绘制自由空间
        for x, y in free_cells:
            self.ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='white', 
                                        edgecolor='lightgray', linewidth=0.5))
        
        # 绘制未知区域
        for x, y in unknown_cells:
            self.ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='gray', 
                                        alpha=0.5))
        
        # 绘制障碍物
        for x, y in obstacle_cells:
            self.ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='black'))
        
        # 绘制基地点 - 改为蓝色，不再添加文字标注
        for x, y in base_cells:
            self.ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='royalblue', 
                                        alpha=0.8))
        
        # 绘制任务
        self._update_tasks(local_map)
        
        # 绘制前沿点
        if local_map.frontiers:
            for x, y in local_map.frontiers:
                marker = plt.Circle((x, y), 0.15, color='yellow', alpha=0.7, zorder=4)
                self.ax.add_patch(marker)
                self.frontier_markers.append(marker)
        
        # 绘制中心点
        if local_map.centroids:
            for centroid, data in local_map.centroids.items():
                # 绘制中心点
                star_marker = self.ax.scatter(centroid[0], centroid[1], c='magenta', marker='*', 
                                        s=200, zorder=7)
                self.centroid_markers.append(star_marker)
                
                # 绘制边界框
                try:
                    bbox = data['bound_box']
                    x_min, y_min = bbox[0]
                    x_max, y_max = bbox[2]
                    width = x_max - x_min
                    height = y_max - y_min
                    rect = patches.Rectangle((x_min - 0.5, y_min - 0.5), width + 1, height + 1, 
                                        linewidth=2, edgecolor='magenta', facecolor='none', 
                                        linestyle='--', zorder=6)
                    self.ax.add_patch(rect)
                    self.centroid_boxes.append(rect)
                    
                    # 显示未知面积
                    text = self.ax.text(centroid[0] + 0.5, centroid[1] + 0.5, f"面积: {data['area']}", 
                                fontsize=9, color='magenta', weight='bold', zorder=7,
                                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
                    self.centroid_texts.append(text)
                except (IndexError, KeyError) as e:
                    print(f"无法绘制中心点 {centroid} 的边界框: {e}")
    
    def _create_legend(self, local_map: "Map"):
        """创建图例"""
        legend_elements = []
        
        # 地图元素图例
        legend_elements.append(patches.Patch(color='gray', alpha=0.5, label='未知区域'))
        legend_elements.append(patches.Patch(color='white', edgecolor='lightgray', label='自由空间'))
        legend_elements.append(patches.Patch(color='black', label='障碍物'))
        legend_elements.append(patches.Patch(color='royalblue', alpha=0.8, label='基地点'))
        
        # 任务图例
        if local_map.tasks_seen:
            legend_elements.append(patches.Circle((0, 0), radius=0.1, color='green', alpha=0.8, label='任务'))
        
        # 前沿点图例
        if local_map.frontiers:
            legend_elements.append(patches.Circle((0, 0), radius=0.1, color='yellow', alpha=0.7, label='前沿点'))
        
        # 中心点图例
        if local_map.centroids:
            legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='magenta', 
                                        markersize=15, label='中心点'))
        
        # 添加图例
        if legend_elements:
            legend = self.ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
            legend.get_frame().set_alpha(0.7)
    
    def _get_agent_color(self, agent_name):
        """
        根据智能体名称分配唯一颜色
        使用预定义的明显不同的颜色，再根据需要动态生成
        """
        # 如果是首次调用，初始化颜色映射字典
        if not hasattr(self, 'agent_color_map'):
            self.agent_color_map = {}
            
            # 预定义一组明显不同的颜色（20种）
            self.predefined_colors = [
                '#FF0000',  # 红
                '#00FF00',  # 绿
                '#0000FF',  # 蓝
                '#FFFF00',  # 黄
                '#FF00FF',  # 洋红
                '#00FFFF',  # 青
                '#FF8000',  # 橙
                '#8000FF',  # 紫
                '#0080FF',  # 天蓝
                '#FF0080',  # 粉
                '#80FF00',  # 黄绿
                '#00FF80',  # 青绿
                '#FF8080',  # 浅红
                '#80FF80',  # 浅绿
                '#8080FF',  # 浅蓝
                '#804000',  # 棕
                '#408000',  # 橄榄
                '#004080',  # 深蓝
                '#800040',  # 深紫红
                '#008040'   # 森林绿
            ]
            
            # 已分配的颜色索引
            self.next_color_idx = 0
        
        # 如果智能体已有颜色，则返回它
        if agent_name in self.agent_color_map:
            return self.agent_color_map[agent_name]
        
        # 分配新颜色
        if self.next_color_idx < len(self.predefined_colors):
            # 使用预定义颜色
            color = self.predefined_colors[self.next_color_idx]
            self.next_color_idx += 1
        else:
            # 如果预定义颜色用完，则动态生成
            # 使用Golden Ratio方法生成更多均匀分布的颜色
            golden_ratio = 0.618033988749895
            h = (self.next_color_idx * golden_ratio) % 1
            # HSV转RGB，使用饱和度和明度为0.95，以确保颜色鲜明
            h, s, v = h, 0.95, 0.95
            # 转换HSV到RGB
            color = self._hsv_to_rgb(h, s, v)
            self.next_color_idx += 1
        
        # 保存智能体的颜色映射
        self.agent_color_map[agent_name] = color
        return color

    def _hsv_to_rgb(self, h, s, v):
        """
        将HSV值转换为RGB十六进制颜色
        
        Parameters:
            h: 色相 [0, 1]
            s: 饱和度 [0, 1]
            v: 明度 [0, 1]
        
        Returns:
            十六进制颜色字符串
        """
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        return f'#{r:02x}{g:02x}{b:02x}'