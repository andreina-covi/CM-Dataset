import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import pygame

from ai2thor.controller import Controller

ROOT_PATH = 'Dataset'
NAVIGATION_FILE_PATH = os.path.join(ROOT_PATH, 'navigation.csv')
OBJECTS_FILE_PATH = os.path.join(ROOT_PATH, 'objects.csv')
IMAGE_PATH = os.path.join(ROOT_PATH, 'Images')
# Define key-to-action mapping
def get_action(key):
    dict_action = None
    if key == pygame.K_UP:
        dict_action =  {"action": "MoveAhead"}
    elif key == pygame.K_DOWN:
        dict_action =  {"action": "MoveBack"}
    elif key == pygame.K_LEFT:
        dict_action =  {"action": "RotateLeft"}
    elif key == pygame.K_RIGHT:
        dict_action =  {"action": "RotateRight"}
    elif key == pygame.K_SPACE:
        dict_action =  {"action": "LookDown"}
    elif key == pygame.K_a:
        dict_action =  {"action": "MoveLeft"}
    elif key == pygame.K_d:
        dict_action =  {"action": "MoveRight"}
    return dict_action

def stop_running(key):
    stop = False
    if key == pygame.K_ESCAPE or key == pygame.K_q:
        stop = True
    return stop

def get_object_data(arr_objects):
    objects = set()
    cond_objs = []

    for obj_dict in arr_objects:
        if not obj_dict['visible']:
            continue
        cond_objs.append([
            # class label
            obj_dict['objectType'],
            # name
            obj_dict['name'],
            # global position 
            tuple(round_number(obj_dict['position'], 2)),
            # local rotation
            tuple(round_number(obj_dict['rotation'], 2)),
            # euclidean distance from the center-point to the agent
            np.round(obj_dict['distance'], 4)
        ])
        objects.add((
            obj_dict['objectType'],
            obj_dict['name'],
            tuple(round_number(obj_dict['position'], 2)),
        ))
    return cond_objs, objects

def round_number(arr_numbers, n_round):
    rounded_arr = []
    if type(arr_numbers) is dict:
        rounded_arr = [arr_numbers['x'], arr_numbers['y'], arr_numbers['z']]
    else:
        rounded_arr = arr_numbers
    return tuple([np.round(number, n_round) for number in rounded_arr])

def collect_data(event, dict_agent, dict_objects, action, index, image_path):
    position = round_number(event.metadata['agent']['position'], 2)
    rotation = round_number(event.metadata['agent']['rotation'], 2)
    key = (action, position, rotation)
    if key not in dict_agent:
        dict_agent[key] = {'objects': [], 'image': ''}
        cond_objs, objects = get_object_data(event.metadata['objects'])
        dict_agent[key]['objects'] = cond_objs
        if dict_objects['objects']:
            dict_objects['objects'].update(objects)
        else:
            dict_objects['objects'] = objects
        # save image
        image_name = os.path.join(image_path, 'img_' + str(index) + '.png')
        dict_agent[key]['image'] = image_name
        cv2.imwrite(image_name, event.cv2img)

def navigate(controller, screen, dict_agent, dict_objects, image_path=IMAGE_PATH):
    # Game loop
    running = True
    index_img = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if stop_running(event.key):
                    running = False
                    break
                action = get_action(event.key)
                if action:
                    ai2_event = controller.step(action=action)
                    collect_data(ai2_event, dict_agent, dict_objects, action['action'], index_img, image_path)
                    index_img += 1

        # Render frame
        frame = controller.last_event.frame
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = np.rot90(frame)
        frame = np.flipud(frame)
        frame = pygame.surfarray.make_surface(frame)
        screen.blit(frame, (0, 0))
        pygame.display.flip()

    controller.stop()
    pygame.quit()
    
def draw_allocentric_map(obj_list):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # plt.figure(figsize=(10, 10))
    # plt.scatter(0, 0, c='red', label='Agent')  # Agent at center
    
    for t in obj_list:
        ax.scatter(t[2][0], t[2][1], t[2][2], label=t[0])
        ax.text(t[2][0], t[2][1], t[2][2], t[0], fontsize=8)

    # plt.xlim(-2, 2)
    # plt.ylim(-2, 2)
    # plt.gca().invert_zaxis()  # Optional for aligning with camera view
    # plt.legend()
    plt.title('Egocentric Object Map')
    # plt.grid(True)
    plt.show()

def save_data_by_axis(dict_data, base_name, array):
    axis_names = ['-x', '-y', '-z']
    for (axis, item) in zip(axis_names, array):
        dict_data[base_name + axis].append(item)

def save_data_navigation(dict_navigation, key, objects_data, path_image):
    for object_data in objects_data:
        dict_navigation['ag-action'].append(key[0])
        save_data_by_axis(dict_navigation, 'ag-pos', key[1])
        save_data_by_axis(dict_navigation, 'ag-rot', key[2])
        dict_navigation['obj-type'].append(object_data[0])
        dict_navigation['obj-name'].append(object_data[1])
        save_data_by_axis(dict_navigation, 'obj-pos', object_data[2])
        save_data_by_axis(dict_navigation, 'obj-rot', object_data[3])
        dict_navigation['obj-distance'].append(object_data[4])
        dict_navigation['path'].append(path_image)    

def get_dict_navigation(dict_agent):
    dict_navigation = {'ag-action': [], 'ag-pos-x': [], 'ag-pos-y': [], 'ag-pos-z': [], 
        'ag-rot-x': [], 'ag-rot-y': [], 'ag-rot-z': [], 'obj-type': [], 
        'obj-name': [], 'obj-pos-x': [], 'obj-pos-y': [], 'obj-pos-z': [], 
        'obj-rot-x': [], 'obj-rot-y': [], 'obj-rot-z': [], 'obj-distance': [], 'path': []}
    for key in dict_agent:
        object_data = dict_agent[key]['objects']
        image_path = dict_agent[key]['image']
        save_data_navigation(dict_navigation, key, object_data, image_path)
    return dict_navigation

def get_dict_objects(set_objects):
    dict_objects = {'obj-type': [], 'obj-name': [], 'obj-pos-x': [], \
        'obj-pos-y': [], 'obj-pos-z': []}
    for t in set_objects:
        dict_objects['obj-type'].append(t[0])
        dict_objects['obj-name'].append(t[1])
        save_data_by_axis(dict_objects, 'obj-pos', t[2])
    return dict_objects

WIDTH = 500
HEIGHT = 500
GRID_SIZE = 0.25
VISIBILITY_DISTANCE = 1.5
ROTATE_STEP_DEGREES = 45

if __name__ == '__main__':
    # Initialize AI2-THOR
    controller = Controller(
        scene="FloorPlan1",
        visibility_distance=VISIBILITY_DISTANCE,
        rotateStepDegrees=ROTATE_STEP_DEGREES, 
        snapToGrid=False,
        gridSize=GRID_SIZE, 
        width=WIDTH, 
        height=HEIGHT)

    os.makedirs(ROOT_PATH, exist_ok=True)
    os.makedirs(IMAGE_PATH, exist_ok=True)
    # Pygame setup
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AI2-THOR Explorer")
    dict_agent = {}
    data_objects = {'objects': set()}
    navigate(controller, screen, dict_agent, data_objects, IMAGE_PATH)
    dict_navigation = get_dict_navigation(dict_agent)
    df_navigation = pd.DataFrame(dict_navigation)
    dict_objects = get_dict_objects(data_objects['objects'])
    df_objects = pd.DataFrame(dict_objects)
    df_navigation.to_csv(NAVIGATION_FILE_PATH)
    df_objects.to_csv(OBJECTS_FILE_PATH)