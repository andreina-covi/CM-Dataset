import pygame
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ai2thor.controller import Controller

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

def preprocess_objectId(text):
    output = []
    arr_strs = text.split("|")
    output.append(arr_strs[0])
    output.extend([float(num) for num in arr_strs[1:]])
    return tuple(output)

# def print_movement(event):
#     agent_pos = event.metadata['agent']['position']
#     agent_rot = event.metadata['agent']['rotation']
#     print("Position: ", agent_pos, "Rotation: ", agent_rot)

def get_object_map(arr_objects, position, rotation):
    object_map = set()
    map = set()

    for obj in arr_objects:
        if not obj['visible']:
            continue

        obj_pos = obj['position']
        dx = obj_pos['x'] - position['x']
        dz = obj_pos['z'] - position['z']

        # Convert to agent-centric coordinates
        theta = -np.deg2rad(rotation)
        x_rel = dx * np.cos(theta) - dz * np.sin(theta)
        z_rel = dx * np.sin(theta) + dz * np.cos(theta)
        tuple_object = preprocess_objectId(obj['objectId'])
        object_map.add(tuple_object)

        map.add((
            #'objectId': 
            tuple_object[0],
            #'x': 
            np.round(x_rel, 4),
            #'z': 
            np.round(z_rel, 4)
        ))
    return map, object_map

def set_egocentric_data(event, map, action):
    # print(event.metadata['agent'], event.metadata['cameraPosition'], event.metadata['lastAction'])
    agent_pos = event.metadata['agent']['position']
    agent_rot = event.metadata['agent']['rotation']['y']  # yaw
    # if not map['positions']:
    map['action'].append(action['action'])
    map['positions'].append([np.round(agent_pos['x'], 4), np.round(agent_pos['z'], 4)])
    map['rotations'].append(agent_rot)

    cond_objs, objects_map = get_object_map(event.metadata['objects'], agent_pos, agent_rot)
    print(cond_objs, objects_map)
    map['cond_objects'].update(cond_objs)
    map['objects'].update(objects_map)

def navigate(controller, screen, map):
    # Game loop
    running = True

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
                    # print(ai2_event.metadata['agent'].keys())
                    # running = False
                    # break
                    set_egocentric_data(ai2_event, map, action)
                    # print_movement(ai2_event)

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

def draw_egocentric_map(obj_list):
    plt.figure(figsize=(6,6))
    plt.scatter(0, 0, c='red', label='Agent')  # Agent at center

    for obj in obj_list:
        plt.scatter(obj['x'], obj['z'], label=obj['objectId'].split('|')[0])
        plt.text(obj['x'], obj['z'], obj['objectId'].split('|')[0][:6], fontsize=8)

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    # plt.gca().invert_zaxis()  # Optional for aligning with camera view
    plt.legend()
    plt.title('Egocentric Object Map')
    plt.grid(True)
    plt.show()
    

WIDTH = 500
HEIGHT = 500
GRID_SIZE = 0.1

if __name__ == '__main__':
    # Initialize AI2-THOR
    controller = Controller(scene="FloorPlan1", gridSize=GRID_SIZE, width=WIDTH, height=HEIGHT)

    # Pygame setup
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AI2-THOR Explorer")
    map = {'action': [], 'positions': [], 'rotations': [], 'objects': {}, \
        'cond_objects': []}
    navigate(controller, screen, map)
    print(map)
    # for ego_pos in ego_position_list:
    #     draw_egocentric_map(ego_pos)