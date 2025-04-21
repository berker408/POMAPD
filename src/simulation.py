from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .map import Map

class Simulation():
    """
    simulation 相关的逻辑
    """
    def __init__(self):
        pass

    def update_map_infos(self, local_map: "Map", real_map, messages):
        """
        更新地图信息
        """
        # Collect all senses points
        all_senses = set()
        for n, info in messages.items():
            all_senses.update(info['sense'])
        
        local_map.update(real_map, all_senses)

    def update_task_infos(self, local_map, real_map, messages):
        """
        更新任务信息
        """
        

        pass