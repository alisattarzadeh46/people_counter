from enum import Enum

class Flow(Enum):
    UP_TO_DOWN = "UP_TO_DOWN"
    DOWN_TO_UP = "DOWN_TO_UP"
    LEFT_TO_RIGHT = "LEFT_TO_RIGHT"
    RIGHT_TO_LEFT = "RIGHT_TO_LEFT"

def is_vertical(flow: "Flow") -> bool:
    return flow in (Flow.UP_TO_DOWN, Flow.DOWN_TO_UP)

def human_flow(flow: "Flow") -> str:
    mapping = {
        Flow.UP_TO_DOWN: "Up → Down",
        Flow.DOWN_TO_UP: "Down → Up",
        Flow.LEFT_TO_RIGHT: "Left → Right",
        Flow.RIGHT_TO_LEFT: "Right → Left",
    }
    return mapping.get(flow, "")
