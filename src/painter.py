"""
提供仿真过程中的可视化功能，支持动态更新地图和智能体位置
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
from typing import TYPE_CHECKING, List, Dict, Tuple, Set

if TYPE_CHECKING:
    from .agent import Agent
    from .map import Map

class Painter:
    """
    用于可视化仿真过程中的地图和智能体状态
    支持实时更新和动画保存
    """
    def __init__(self, figsize=None, dpi=100):
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
            fig_width = min(15, max(8, local_map.width/5))
            fig_height = fig_width * aspect_ratio
            self.figsize = (fig_width, fig_height)
        
        # 创建图形和坐标轴
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
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
        
        # 初始化智能体标记
        for agent in agents:
            # 使用不同颜色区分不同智能体
            agent_color = self._get_agent_color(agent.name)
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
        for agent in agents:
            if agent.name in self.agents_scatter:
                # 获取智能体散点对象
                scatter = self.agents_scatter[agent.name]
                # 更新位置
                scatter.set_offsets([agent.pos[0], agent.pos[1]])
                
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
        
        # 刷新图形
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except (ValueError, RuntimeError) as e:
            print(f"图形更新出错（可能是窗口已关闭）: {e}")
    
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
    
    def start_animation(self, sim_steps: int, local_map: "Map", agents: List["Agent"], 
                   realmap: "Map", simulation, task_planner, path_planner, interval=200):
        """
        启动动画仿真
        
        Parameters:
            sim_steps: 仿真总步数
            local_map: 初始局部地图
            agents: 智能体列表
            realmap: 真实地图
            simulation: 仿真对象
            task_planner: 任务规划器
            path_planner: 路径规划器
            interval: 帧间隔(毫秒)
        """
        self.setup(local_map, agents)
        self.is_animation_running = True
        messages = {agent.name: agent.message for agent in agents}
        
        # 执行初始扫描
        print('\n初始化扫描...')
        for agent in agents:
            agent.execute(realmap, -1)
            messages[agent.name] = agent.publish(-1)
        
        # 更新初始地图
        simulation.update_map_infos(local_map, realmap, messages)
        
        def animate(t):
            if not self.is_animation_running:
                return
            
            try:
                print(f'\n** 仿真步骤 {t} **')
                
                # 1. 更新地图和任务信息
                simulation.update_map_infos(local_map, realmap, messages)
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
                
                # 5. 更新可视化
                self.update(t, local_map, agents)
                
                # # 如果全部已知，可以提前结束
                # if local_map.known_area == local_map.total_area:
                #     print("\n地图已完全探索，仿真提前结束")
                #     self.is_animation_running = False
                    
            except Exception as e:
                print(f"动画更新出错: {e}")
                import traceback
                traceback.print_exc()
                self.is_animation_running = False
        
        # 创建动画
        self.animation_obj = animation.FuncAnimation(
            self.fig, animate, frames=range(sim_steps),
            interval=interval, repeat=False
        )
        
        plt.show()
    
    def stop_animation(self):
        """停止动画"""
        self.is_animation_running = False
        if self.animation_obj:
            self.animation_obj.event_source.stop()
    
    def save_animation(self, filename="simulation.mp4", fps=5):
        """
        保存动画为视频文件
        
        Parameters:
            filename: 输出文件名
            fps: 帧率
        """
        if self.animation_obj:
            writer = animation.FFMpegWriter(fps=fps)
            self.animation_obj.save(filename, writer=writer)
            print(f"动画已保存到 {filename}")
    
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
        根据智能体名称生成一致的颜色
        """
        # 基础颜色列表
        colors = ['#FF5733', '#33FF57', '#3357FF', '#F3FF33', '#FF33F3', '#33FFF3', '#F333FF']
        
        # 根据名称生成一个索引
        name_sum = sum(ord(c) for c in agent_name)
        color_index = name_sum % len(colors)
        
        return colors[color_index]