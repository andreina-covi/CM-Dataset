import pygame
import cv2
import numpy as np
from ai2thor.controller import Controller

WIDTH = 500
HEIGHT = 500

# Initialize AI2-THOR
controller = Controller(scene="FloorPlan1", gridSize=0.05, width=WIDTH, height=HEIGHT)

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI2-THOR Explorer")

# Define key-to-action mapping
def get_action(key):
    if key == pygame.K_UP:
        return {"action": "MoveAhead"}
    elif key == pygame.K_DOWN:
        return {"action": "MoveBack"}
    elif key == pygame.K_LEFT:
        return {"action": "RotateLeft"}
    elif key == pygame.K_RIGHT:
        return {"action": "RotateRight"}
    elif key == pygame.K_SPACE:
        return {"action": "LookDown"}
    elif key == pygame.K_a:
        return {"action": "MoveLeft"}
    elif key == pygame.K_d:
        return {"action": "MoveRight"}
    return None

def stop_running(key):
    stop = False
    if key == pygame.K_ESCAPE or key == pygame.K_q:
        stop = True
    return stop

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
                    # print(ai2_event)

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
