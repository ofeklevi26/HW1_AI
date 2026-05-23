import numpy as np
from HaifaEnv import HaifaEnv
from typing import List, Tuple
import heapdict

class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0, depth=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = depth


class Agent:
    def solution(self, node):
        actions = []
        while node.parent is not None:
            actions.append(node.action)
            node = node.parent
        actions.reverse()
        return actions

    def expand(self, env, node):
        children = []
        for action, (new_state, cost, terminated) in env.succ(node.state).items():
            if new_state is None:
                continue

            child = Node(
                state=new_state,
                parent=node,
                action=action,
                path_cost=node.path_cost + cost,
                depth=node.depth + 1
            )
            children.append(child)

        return children

class BFSGAgent(Agent):
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError
        


class GreedyAgent(Agent):
  
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError



class AStarEpsilonAgent(Agent):
    def __init__(self):
        raise NotImplementedError
        
    def h_focal(self, state: int) -> float: # heuristic for focal list (you don't have to use it)
        pass

    def search(self, env: HaifaEnv, epsilon: float = None): 
        raise NotImplementedError  



class AStarAgent(Agent):
    
    def __init__(self):
        raise NotImplementedError

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError 

