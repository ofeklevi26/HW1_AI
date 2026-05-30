import numpy as np
from HaifaEnv import HaifaEnv
from typing import List, Tuple
from collections import deque
import heapdict

class Node:
    def __init__(self, state, parent=None, action=None, path_cost : float =0.0, depth : int =0, discovery_order : int= 0)-> None:
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
        # Counts nodes that actually are expanded - sons are checked.
        self.expanded_nodes += 1

    def reset_search_counters(self) -> None:
        self.discovery_order_counter = 0
        self.expanded_nodes = 0

    def make_node(self, state: int, parent : Node = None , action : int =None, path_cost : float = 0.0, depth: int = 0) -> Node:
        return Node(
            state=state,
            parent=parent,
            action=action,
            path_cost=path_cost,
            depth=depth,
            discovery_order=self.next_in_order(),
        )

    def solution(self, node: Node) -> List[int]:
        actions = []
        current = node
        while current is not None and current.parent is not None:
            actions.append(current.action)
            current = current.parent
        actions.reverse()
        return actions

    def is_goal(self, node: Node, env: HaifaEnv) -> bool:
        return node is not None and env.is_final_state(node.state)

    def is_valid_successor(self, new_state: int, cost: float, terminated: bool, env: HaifaEnv) -> bool:

        # Holes are not valid succesors.

        if new_state is None or cost is None:
            return False
        if not np.isfinite(cost):
            return False
        if terminated and not env.is_final_state(new_state):
            return False
        return True

    def expand(self, env : HaifaEnv, node : Node)-> List[Node]:
        children = []
        successors = env.succ(node.state)
        self.expand_node(node)
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

    def h_Haifa(self, state: int, env: HaifaEnv) -> float:
        row, col = env.to_row_col(state)

        min_goal_distance = float("inf")

        for goal_state in env.get_goal_states():
            goal_row, goal_col = env.to_row_col(goal_state)
            manhattan_distance = abs(row - goal_row) + abs(col - goal_col)
            min_goal_distance = min(min_goal_distance, manhattan_distance)

        c_passway = 100.0

        return min(min_goal_distance, c_passway)

class BFSGAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__()

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        self.reset_search_counters()
        initial_state = env.get_initial_state()
        root = self.make_node(state=initial_state, parent= None, action= None, path_cost=0.0)

        OPEN = deque([root])
        OPEN_states = {initial_state}
        CLOSE = set()

        while len(OPEN) > 0:
            current = OPEN.popleft()
            OPEN_states.remove(current.state)

            if self.is_goal(current, env):
                return self.solution(current), current.path_cost, self.expanded_nodes

            CLOSE.add(current.state)

            children = self.expand(env, current)

            for child in children:
                if child.state not in OPEN_states and child.state not in CLOSE:
                    OPEN.append(child)
                    OPEN_states.add(child.state)
        return [], float('inf'), self.expanded_nodes


        


class GreedyAgent(BaseAgent):
  
    def __init__(self) -> None:
        super().__init__()

    def greedy_priority(self, node: Node, env: HaifaEnv) -> Tuple[float, int]:
        #it takes into account the heuristic and also the order of entrance to the open state list
        return self.h_Haifa(node.state, env), node.discovery_order

    def should_add_to_open(self, node: Node, OPEN_states: set, CLOSE: set) -> bool:
        #checks if the state is already in the open list or closed list
        return node.state not in OPEN_states and node.state not in CLOSE


    def add_to_open(self, OPEN : heapdict.heapdict, OPEN_states: set, node: Node, env: HaifaEnv) -> None:
        OPEN[node] = self.greedy_priority(node, env)
        OPEN_states.add(node.state)

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        self.reset_search_counters()

        initial_state = env.get_initial_state()
        root = self.make_node(
            state=initial_state,
            parent=None,
            action=None,
            path_cost=0.0,
            depth=0
        )
        OPEN = heapdict.heapdict()
        OPEN_states = set()
        CLOSE = set()
        self.add_to_open(OPEN, OPEN_states, root, env)

        while len(OPEN) > 0:
            current, _ = OPEN.popitem()
            OPEN_states.remove(current.state)

            if self.is_goal(current, env):
                return self.solution(current), current.path_cost, self.expanded_nodes

            CLOSE.add(current.state)

            children = self.expand(env, current)

            for child in children:
                if self.should_add_to_open(child, OPEN_states, CLOSE):
                    self.add_to_open(OPEN, OPEN_states, child, env)

        return [], float("inf"), self.expanded_nodes



class AStarEpsilonAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def f_value(self, node: Node, env: HaifaEnv) -> float:
        return node.path_cost + self.h_Haifa(node.state, env)
        
    def h_focal(self, state: int) -> float: # heuristic for focal list (you don't have to use it)
        return self.h_Haifa(state, self.env)

    def astar_priority(self, node: Node, env: HaifaEnv) -> Tuple[float, int]:
        #it takes into account the heuristic and also the order of entrance to the open state list
        return self.f_value(node, env), node.discovery_order

    def get_f_min(self, OPEN: heapdict.heapdict) -> float:
        return min(priority[0] for priority in OPEN.values())

    def is_in_focal_condition(self, f_value: float, f_min: float, epsilon: float) -> bool:
        return f_value <= (1.0 + epsilon) * f_min

    def build_focal_list(self, OPEN: heapdict.heapdict, epsilon: float) -> List[int]:
        focal = []
        f_min = self.get_f_min(OPEN)

        for state, priority in OPEN.items():
            node_f_value = priority[0]

            if self.is_in_focal_condition(node_f_value, f_min, epsilon):
                focal.append(state)

        return focal

    def focal_priority(self, state: int, OPEN_nodes: dict) -> Tuple[float, int]:
        return self.h_focal(state), OPEN_nodes[state].discovery_order

    def choose_from_focal(self, focal: List[int], OPEN_nodes: dict) -> int:
        best_state = focal[0]
        best_priority = self.focal_priority(best_state, OPEN_nodes)

        for state in focal[1:]:
            current_priority = self.focal_priority(state, OPEN_nodes)

            if current_priority < best_priority:
                best_state = state
                best_priority = current_priority

        return best_state

    def search(self, env: HaifaEnv, epsilon: float = None) -> Tuple[List[int], float, int]:
        self.reset_search_counters()
        self.env = env

        if epsilon is None:
            epsilon = 0.0

        initial_state = env.get_initial_state()

        root = self.make_node(
            state=initial_state,
            parent=None,
            action=None,
            path_cost=0.0,
            depth=0
        )

        OPEN = heapdict.heapdict()
        OPEN_nodes = {}
        CLOSED = set()
        best_g = {}

        OPEN[root.state] = self.astar_priority(root, env)
        OPEN_nodes[root.state] = root
        best_g[root.state] = root.path_cost

        while len(OPEN) > 0:
            focal = self.build_focal_list(OPEN, epsilon)
            chosen_state = self.choose_from_focal(focal, OPEN_nodes)

            current = OPEN_nodes.pop(chosen_state)
            del OPEN[chosen_state]

            if self.is_goal(current, env):
                return self.solution(current), current.path_cost, self.expanded_nodes

            CLOSED.add(current.state)

            children = self.expand(env, current)

            for child in children:
                old_best_g = best_g.get(child.state, float("inf"))

                if child.path_cost < old_best_g:
                    best_g[child.state] = child.path_cost

                    if child.state in CLOSED:
                        CLOSED.remove(child.state)

                    OPEN_nodes[child.state] = child
                    OPEN[child.state] = self.astar_priority(child, env)

        return [], float("inf"), self.expanded_nodes



class AStarAgent(BaseAgent):
    
    def __init__(self):
        raise NotImplementedError

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError 

