import numpy as np
from HaifaEnv import HaifaEnv
from typing import List, Tuple
import heapdict

class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0, depth=0, discovery_order= 0)-> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = depth
        self.discovery_order = discovery_order


class BaseAgent:

    def __init__(self)-> None:
        self.env = None
        self.discovery_order_counter = 0
        self.expanded_nodes = 0

    def next_in_order(self) -> int:
        self.discovery_order_counter += 1
        return self.discovery_order_counter

    def expand_node(self, node: Node) -> None:
        # Counts nodes that actually enter the OPEN.
        self.expanded_nodes += 1

    def reset_search_counters(self) -> None:
        self.discovery_order_counter = 0
        self.expanded_nodes = 0

    def make_node(self, state, parent=None, action=None, path_cost = 0, depth: int = 0) -> Node:
        return Node(
            state=state,
            parent=parent,
            action=action,
            path_cost=path_cost,
            depth=depth,
            discovery_order=self.next_in_order(),
        )

    def solution(self, node) -> List[int]:
        actions = []
        current = node
        while current is not None and current.parent is not None:
            actions.append(current.action)
            current = current.parent
        actions.reverse()
        return actions

    def is_goal(self, node: Node, env: HaifaEnv) -> bool:
        return node is not None and env.is_final_state(node.state)

    def is_valid_successor(self, new_state, cost, terminated, env: HaifaEnv) -> bool:

        # Holes are not valid succesors.

        if new_state is None or cost is None:
            return False
        if not np.isfinite(cost):
            return False
        if terminated and not env.is_final_state(new_state):
            return False
        return True

    def expand(self, env, node)-> List[Node]:
        children = []
        successors = env.succ(node.state)

        for action in sorted(successors.keys()):
            new_state, step_cost, terminated = successors[action]
            if not self.is_valid_successor(new_state, step_cost, terminated, env):
                continue

            child = self.make_node(
                state=new_state,
                parent=node,
                action=action,
                path_cost=node.path_cost + step_cost,
                depth=node.depth + 1,
            )
            children.append(child)

        return children

class BFSGAgent(BaseAgent):
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError
        


class GreedyAgent(BaseAgent):
  
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError



class AStarEpsilonAgent(BaseAgent):
    def __init__(self):
        raise NotImplementedError
        
    def h_focal(self, state: int) -> float: # heuristic for focal list (you don't have to use it)
        pass

    def search(self, env: HaifaEnv, epsilon: float = None): 
        raise NotImplementedError  



class AStarAgent(BaseAgent):
    
    def __init__(self):
        raise NotImplementedError

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError 

