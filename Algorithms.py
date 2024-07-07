import time
import numpy as np
from typing import List, Tuple, Dict
import heapdict
from CampusEnv import CampusEnv

class Node:
    """
    Initializes a new node.

    Args:
        state (int): The state represented by this node.
        parent (Node, optional): The parent node. Defaults to None.
        cost (float): The cost to reach this node. Defaults to 0.0.
        heuristic (int): The heuristic estimate from this node to the goal. Defaults to 0.
        h_weight (float): The weight of the heuristic in cost calculation. Defaults to 0.0.
    """
    def __init__(self, state: int = 0, parent = None, cost: float = 0.0 , heuristic: int = 0, h_weight: float = 0, is_hole:bool = False, action = None ) -> None:
        self.state = state
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic
        self.h_weight = h_weight
        self.is_hole = is_hole
        self.action = action

    @staticmethod
    def make_node(state: int, parent = None, cost: float = 0, action = None, is_hole:bool = False, heuristic: int = 0,  h_weight: float = 0 ):
        """
        Static method to create a new node.

        Args:
            state (int): The state represented by this node.
            parent (Node, optional): The parent node. Defaults to None.
            cost (float): The cost to reach this node. Defaults to 0.0.
            heuristic (int): The heuristic estimate from this node to the goal. Defaults to 0.
            h_weight (float): The weight of the heuristic in cost calculation. Defaults to 0.0.

        Returns:
            Node: A new instance of Node.
        """
        return Node(state, parent, cost, heuristic, h_weight, is_hole, action)
    
    def __repr__(self) -> str:
        return f"Node(state={self.state}, parent={self.parent}, cost={self.cost})"
    
    def __str__(self) -> str:
        return f"Node({self.state}, {(1 - self.h_weight) * self.cost + self.h_weight * self.heuristic:.2f})"
    
    def __lt__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        current_f = (1 - self.h_weight) * self.cost + self.h_weight * self.heuristic
        other_f = (1 - self.h_weight) * other.cost + self.h_weight * other.heuristic
        return (current_f, self.state) < (other_f, other.state)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.state == other.state and self.cost == other.cost
    


class Utility:
    @staticmethod
    def solution(node: Node, expended: int) -> Tuple[List[int], float, int]:
        """
        Reconstruct the solution path from a node to the root node and calculate the total cost.

        Args:
            node (Node): The end node of the solution path.
            expended (int): The number of nodes expanded during the search.

        Returns:
            Tuple[List[int], float, int]: A tuple containing:
                - A list of actions to reach the solution.
                - The total cost of the solution path.
                - The number of nodes expanded.
        """
        actions = []
        cost = 0.0
        while node.parent is not None:
            actions.append(node.action)
            cost += node.cost
            node = node.parent
        actions.reverse()  # Reverse actions to get them in the correct order
        return actions, cost, expended

class CampusHeuristic:
    """
     A heuristic for a campus environment.

    The heuristic value is calculated as the minimum of the Manhattan distance to any goal state and the portal cost.

    Args:
        env (CampusEnv): The environment representing the campus.
        portal_cost (float): The cost to use a portal.
    """
    def __init__(self, env: CampusEnv, portal_cost: float):
        self.env = env
        self.portal_cost = portal_cost

    def get_heuristic_value(self, state: int) -> float:
        """
        Calculate the heuristic value for a given state.

        Args:
            state (int): The current state for which the heuristic value is calculated.

        Returns:
            float: The heuristic value
        """
        possible_heuristic_values = [self.portal_cost]
        state_row, state_col = self.env.to_row_col(state)
        for goal_state in self.env.get_goal_states():
            goal_row, goal_col = self.env.to_row_col(goal_state)
            manhattan_distance = abs(goal_row - state_row) + abs(goal_col - state_col)
            possible_heuristic_values.append(manhattan_distance)
        return min(possible_heuristic_values)

class Agent:
    def __init__(self):
        self.actions = []
        self.cost = 0.0
        self.expended = 0
        self.failure = ([], 0.0, 0)
    
class DFSGAgent(Agent):
    def __init__(self):
        super().__init__()

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        self.actions = []
        self.cost = 0.0
        self.expended = 0
        self.failure = ([], 0.0, 0)
        init_node = Node.make_node(env.get_initial_state())
        open_list = [init_node]
        close_list = []
        return self.recursiveDFSG(env, open_list, close_list)
    
    def recursiveDFSG(self, env: CampusEnv, open_list: List[Node], close_list: List[int]) -> Tuple[List[int], float, int]: 
        if not open_list:
            return self.failure

        node = open_list.pop()
        close_list.append(node.state)

        if env.is_final_state(node.state):
            return Utility.solution(node, self.expended)
        
        self.expended += 1
        for action, (new_state, cost, terminated) in env.succ(node.state).items():
            child_node = Node.make_node(new_state, node, cost, action)
            if child_node.state not in close_list and all(c.state != child_node.state for c in open_list):
                open_list.append(child_node)
                if not terminated or env.is_final_state(new_state):
                    result = self.recursiveDFSG(env, open_list, close_list)
                    if result != self.failure:
                        return result
                else:
                    self.expended += 1  # add the holes to the expended states counter
        
        return self.failure

class UCSAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.open_list = heapdict.heapdict() 
        self.close_list = []

    def reset(self):
        self.open_list = heapdict.heapdict() 
        self.close_list = []
        self.actions = []
        self.cost = 0.0
        self.expended = 0
        self.failure = ([], 0.0, 0)
    
    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        self.reset()
        init_node = Node.make_node(env.get_initial_state())
        self.open_list[init_node.state] = init_node
        self.close_list = []
        while self.open_list:
            state, node = self.open_list.popitem()
            self.close_list.append(node.state)  

            if env.is_final_state(state):
                actions, final_cost, expended =  Utility.solution(node, self.expended)
                final_cost = node.cost
                return actions, final_cost, expended

            self.expended += 1
            if node.is_hole:
                continue

            if state == 92:
                print()

            for action, (new_state, cost, terminated) in env.succ(state).items():
                if cost == None:
                    print(f"UCS({state})")
                new_cost = node.cost + cost
                child_node = Node.make_node(new_state, node, new_cost, action, terminated)
                if child_node.state not in self.close_list and child_node.state not in self.open_list.keys():
                    self.open_list[new_state] = child_node
                elif child_node.state in self.open_list.keys():
                    if new_cost < self.open_list[new_state].cost:
                        self.open_list[new_state] = child_node
        return self.failure


class WeightedAStarAgent(Agent):
    def __init__(self):
        super().__init__()
        self.open_list = heapdict.heapdict() 
        self.close_list = heapdict.heapdict()

    def reset(self):
        self.open_list = heapdict.heapdict() 
        self.close_list = heapdict.heapdict()
        self.actions = []
        self.cost = 0.0
        self.expended = 0
        self.failure = ([], 0.0, 0)

    def search(self, env: CampusEnv, h_weight: float) -> Tuple[List[int], float, int]:
        self.reset()
        heuristic = CampusHeuristic(env, 100)
        init_state = env.get_initial_state()
        init_node = Node.make_node(init_state, None, 0,None, False, heuristic.get_heuristic_value(init_state), h_weight)
        self.open_list[init_state] = init_node
        while self.open_list:
            state, node = self.open_list.popitem()
            self.close_list[state] = node  

            if env.is_final_state(state):
                actions, final_cost, expended =  Utility.solution(node, self.expended)
                final_cost = node.cost
                return actions, final_cost, expended

            self.expended += 1
            if node.is_hole:
                continue
            for action, (child_state, cost, terminated) in env.succ(state).items():
                total_cost = node.cost + cost
                child_heuristic = heuristic.get_heuristic_value(child_state)
                child_node = Node.make_node(child_state, node, total_cost, action, terminated, child_heuristic, h_weight)
                if child_state not in self.close_list.keys() and child_state not in self.open_list.keys():
                    self.open_list[child_state] = child_node
                elif child_state in self.open_list.keys():
                    if total_cost < self.open_list[child_state].cost:
                        self.open_list[child_state] = child_node # check why never get in
                else:
                    if total_cost < self.close_list[child_state].cost:
                        self.open_list[child_state] = child_node  # check why never get in
                        self.close_list.pop(child_state)
        return self.failure


class AStarAgent(WeightedAStarAgent):
    def __init__(self):
        super().__init__()

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        actions, final_cost, expended = super().search(env, 0.5)
        return actions, final_cost, expended

if __name__ == "__main__":
    MAPS = {
    "4x4": ["SFFF",
            "FHFH",
            "FFFH",
            "HFFG"],
    "4x4_2": ["SFFF",
            "FFFF",
            "FFFF",
            "FFFG"], # 0->4->8->12->13->14
    "8x8": ["SFFFFFFF",
            "FFFFFTAL",
            "TFFHFFTF",
            "FPFFFHTF",
            "FAFHFPFF",
            "FHHFFFHF",
            "FHTFHFTL",
            "FLFHFFFG"],
    }
    env = CampusEnv(MAPS["8x8"])
    state = env.reset()
    DFSG_agent = DFSGAgent()
    actions, total_cost, expanded = DFSG_agent.search(env)
    print("DFS")
    print(f"Total_cost: {total_cost}")
    print(f"Expanded: {expanded}")
    print(f"Actions: {actions}")
    print()

    # UCS_agent = UCSAgent()
    # print("UCS")
    # actions, total_cost, expanded = UCS_agent.search(env)
    # print(f"Total_cost: {total_cost}")
    # print(f"Expanded: {expanded}")
    # print(f"Actions: {actions}")
    # print()

    # WAstar_agent = WeightedAStarAgent()
    # print("WeightedAStarAgent")
    # actions, total_cost, expanded = WAstar_agent.search(env, h_weight=0.7)
    # print(f"Total_cost: {total_cost}")
    # print(f"Expanded: {expanded}")
    # print(f"Actions: {actions}")
    # print()

    # WAstar_agent = AStarAgent()
    # print("AStarAgent")
    # actions, total_cost, expanded = WAstar_agent.search(env)
    # print(f"Total_cost: {total_cost}")
    # print(f"Expanded: {expanded}")
    # print(f"Actions: {actions}")
    # print()
