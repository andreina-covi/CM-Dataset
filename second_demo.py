from ai2thor.controller import Controller

controller = Controller(
    agentMode="default",
    visibilityDistance=1.5,
    scene="FloorPlan212",

    # step sizes
    gridSize=0.25,
    snapToGrid=True,
    rotateStepDegrees=90,

    # image modalities
    renderDepthImage=False,
    renderInstanceSegmentation=False,

    # camera properties
    width=300,
    height=300,
    fieldOfView=90
)

controller.step("PausePhysicsAutoSim")

controller.step(
    action="AdvancePhysicsStep",
    timeStep=0.01
)