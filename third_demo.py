import cv2
import numpy as np
from ai2thor.controller import Controller

# Initialize controller with a scene
controller = Controller(scene='FloorPlan1', gridSize=0.25, width=640, height=480)

def show_frame(event):
    frame = event.frame  # RGB array
    cv2.imshow("AI2-THOR Scene", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)  # Required for window to update

def print_metadata(event):
    print("\nVisible Objects:")
    for obj in event.metadata['objects']:
        if obj['visible']:
            print(f"- {obj['objectId']} (interactable={obj['pickupable'] or obj['openable']})")

# Action loop
try:
    while True:
        event = controller.last_event  # Get last view
        show_frame(event)
        print_metadata(event)

        action = input("Enter action (MoveAhead, RotateRight, PickupObject, etc. or 'quit'): ")

        if action == 'quit':
            break

        # Optional: objectId prompt
        if action in ['PickupObject', 'OpenObject', 'CloseObject', 'ToggleObjectOn', 'ToggleObjectOff']:
            object_id = input("Enter objectId: ")
            event = controller.step(action=action, objectId=object_id)
        else:
            event = controller.step(action=action)

except KeyboardInterrupt:
    pass

controller.stop()
cv2.destroyAllWindows()
