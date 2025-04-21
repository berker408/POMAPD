#!/usr/bin/env python3

import yaml
import math
import numpy as np
from PIL import Image


class InputData:
    def __init__(self, path):
        data = self.read_from_yaml(path)
        ## extenral changes are not recommended
        self._tasks = data['Tasks']
        self._protocols = data['Protocols']
        self._agents = data['Agents']
        self._regions = data['Regions']
        self._ActionDLs = data['ActionDLs']

    @staticmethod
    def read_from_yaml(path):
        """
        Initialize the information of task and environment.
        ----------
        Parameters:
            file_path:(str), the path of the yaml file.
        """
        # Read data from file.yaml
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
            print('\n----------------------------------------')
            print(text_color('GREEN')+'[Init] Read from %s' %path + text_color('RESET'))
        # Print the information to the screen
        print('[Init] LTL Tasks:')
        for task in data['Tasks']:
            print('------ %s' %task)
        print('[Init] Reactive Protocols:')
        for task in data['Protocols']:
            print(f'------ Observe [{task[0][0]}], depend [{task[0][1]}], react {task[1]}')
        print('[Init] Regions:')
        for n, region in enumerate(data['Regions']):
            region['pos'] = tuple(region['pos'])
            print('------ Id: %s, Pos: %s, Label: %s' %(n, region['pos'], region['semtic']))                
        # Reconstruct the information structure of agents
        print('[Init] Agents:')
        agents = list()
        for agent in data['Agents']:
            agent['pos'] = tuple(agent['pos'])
            actions, actionDLs = dict(), list()
            # Assign the action features
            for action, metric in agent['actions'].items():
                for adl in data['ActionDLs']:
                    if action == adl['type']:
                        actions[action] = {
                            'metric': metric,
                            'duration': adl['duration'],
                            }
                        actionDLs.append(adl)
            agent['actions'] = actions
            agent['actionDLs'] = actionDLs
            agents.append(agent)
            print('------ Id: %s, Type: %s, Init pos: %s'
                %(agent['id'], agent['type'], agent['pos']))
        data['Agents'] = agents
        return data

    @property
    def tasks(self):
        return self._tasks

    @property
    def protocols(self):
        return self._protocols

    @property
    def agents(self):
        return self._agents

    @property
    def regions(self):
        return self._regions

    @property
    def ActionDLs(self):
        return self._ActionDLs

def rgba2rgb(rgba):
    r, g, b, a = rgba
    r_int = int(r * 255)
    g_int = int(g * 255)
    b_int = int(b * 255)
    return np.array((r_int, g_int, b_int))


def text_color(color):
    ESC = '\033['
    if color == 'RED':
        return ESC + '31m'
    elif color == 'GREEN':
        return ESC + '32m'
    elif color == 'YELLOW':
        return ESC + '33m'
    elif color == 'BLUE':
        return ESC + '34m'
    elif color == 'MAGENTA':
        return ESC + '35m'
    elif color == 'CYAN':
        return ESC + '36m'
    elif color == 'WHITE':
        return ESC + '37m'
    elif color == 'RESET':
        return ESC + '0m'

def error_print(string):
    print(text_color('RED') + '[ERROR]' + string + text_color('RESET'))

def note_print(string):
    print(text_color('BLUE') + string + text_color('RESET'))

def init_print(string):
    print(text_color('GREEN') + string + text_color('RESET'))

def warn_print(string):
    print(text_color('YELLOW') + string + text_color('RESET'))

def q_distance(q1, q2):
    # 切比雪夫距离
    return max(abs(q1[0]-q2[0]), abs(q1[1]-q2[1]))

def distance(ps, pt):
    return round(math.sqrt((pt[0]-ps[0])**2 + (pt[1]-ps[1])**2), 3)

def p2i(pos, size=100):
    return pos[0]*size +pos[1]

def i2p(index, size=100):
    return (int(index)//size, int(index)%size)

def png_to_gridmap(png_path, grid_size):
    # Open the PNG image
    img = Image.open(png_path)
    # Convert the image to grayscale
    img = img.convert('L')
    # Invert the pixel values
    img_array = np.array(img)
    inverted_img_array = (255 - img_array) / 255
    flipped_grid_map = np.flipud(inverted_img_array)
    # Convert the inverted array back to an image
    img = Image.fromarray(flipped_grid_map)
    # Get the image dimensions
    width, height = img.size
    grid_height = height // grid_size
    grid_width = width // grid_size
    # Initialize an empty grid map
    grid_map = np.zeros((grid_height, grid_width), dtype=np.int32)
    # Iterate through the image and discretize it into the grid map
    for grid_y in range(0, grid_height):
        for grid_x in range(0, grid_width):
            x = grid_x * grid_size
            y = grid_y * grid_size
            # Calculate the average pixel value in the grid cell
            cell_sum = np.sum(np.array(img.crop((x, y, x+grid_size, y+grid_size))))
            # Assign the pixel value to the corresponding cell in the grid map
            grid_map[grid_y, grid_x] = cell_sum // (grid_size ** 2)
    return grid_map, grid_height, grid_width

def replace_labels(template: str, replacement: str) -> str:
    import re
    ## This regular expression finds patterns anclosed in {}
    pattern = re.compile(r'{.*?}')
    ## Substitute the found pattern with the replacement string
    result = re.sub(pattern, replacement, template)
    return result