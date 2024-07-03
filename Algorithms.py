import time
import numpy as np
from typing import List, Tuple, Dict
import heapdict
from CampusEnv import CampusEnv

class Node:
    def __init__(self, state=0, parent=None, cost=0, heuristic = 0, h_weight = 0 ):
        self.state = state
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic
        self.h_weight = h_weight

    @staticmethod
    def make_node(state: int, parent=None, cost: float = 0, heuristic = 0,  h_weight = 0 ):
        return Node(state, parent, cost, heuristic, h_weight)
    
    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state
    
    def __repr__(self) -> str:
        return f"Node({self.state},  {(1 - self.h_weight) * self.cost + self.h_weight * self.heuristic:.2f})"
        # return f"Node(state={self.state}, parent={self.parent}, cost={self.cost})"
    
    def __str__(self) -> str:
        return f"Node({self.state}, {(1 - self.h_weight) * self.cost + self.h_weight * self.heuristic:.2f})"
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.state == other.state and self.cost == other.cost

    def __lt__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        if (1 - self.h_weight) * self.cost + self.h_weight * self.heuristic == (1 - self.h_weight) * other.cost + self.h_weight * other.heuristic:
            return self.state < other.state
        return (1 - self.h_weight) * self.cost + self.h_weight * self.heuristic < (1 - self.h_weight) * other.cost + self.h_weight * other.heuristic

class Utility:
    @staticmethod
    def action_between_nodes(child: Node, parent: Node) -> int:
        if parent.state - child.state == -1:
            return 1  # Move right
        elif parent.state - child.state == 1:
            return 3  # Move left
        elif parent.state - child.state < -1:
            return 0  # Move down
        elif parent.state - child.state > 1:
            return 2  # Move up

    @staticmethod
    def solution(node: Node, expended: int) -> Tuple[List[int], float, int]:
        actions = []
        cost = 0.0
        while node.parent is not None:
            actions.append(Utility.action_between_nodes(node, node.parent))
            cost += node.cost
            node = node.parent
        actions.reverse()  # Reverse actions to get them in the correct order
        return actions, cost, expended

class CampusHeuristic:
    def __init__(self, env: CampusEnv, portal_cost: float):
        self.env = env
        self.portal_cost = portal_cost

    def get_heuristic_value(self, state: int) -> float:
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

class DFSGAgent(Agent):
    def __init__(self):
        super().__init__()

    def search(self, env) -> Tuple[List[int], float, int]:
        init_node = Node.make_node(env.get_initial_state())
        open_list = [init_node]
        close_list = []
        return self.recursiveDFSG(env, open_list, close_list)
    
    def recursiveDFSG(self, env, open_list, close_list) -> Tuple[List[int], float, int]: 
        node = open_list.pop()
        close_list.append(node.state)

        if env.is_final_state(node.state):
            return Utility.solution(node, self.expended)
        
        self.expended += 1
        for action, (new_state, cost, terminated) in env.succ(node.state).items():
            child_node = Node.make_node(new_state, node, cost)
            if child_node.state not in close_list and child_node not in open_list:
                open_list.append(child_node)
                if not terminated or env.is_final_state(new_state):
                    return self.recursiveDFSG(env, open_list, close_list)
                else:
                    self.expended += 1 # add the holes to the expended states counter
        
        return ([], 0.0, 0)

class UCSAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.open_list = heapdict.heapdict() 
        self.close_list = []
    
    def search(self, env) -> Tuple[List[int], float, int]:
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
            for action, (new_state, cost, terminated) in env.succ(state).items():
                new_cost = node.cost + cost
                child_node = Node.make_node(new_state, node, new_cost)
                if child_node.state not in self.close_list and child_node.state not in self.open_list.keys():
                    self.open_list[new_state] = child_node
                elif child_node.state in self.open_list.keys():
                    if new_cost < self.open_list[new_state].cost:
                        self.open_list[new_state] = child_node
        return ([], 0.0, 0)


class WeightedAStarAgent(Agent):
    def __init__(self):
        super().__init__()
        self.open_list = heapdict.heapdict() 
        self.close_list = heapdict.heapdict()

    def search(self, env: CampusEnv, h_weight: float) -> Tuple[List[int], float, int]:
        heuristic = CampusHeuristic(env, 100)
        init_state = env.get_initial_state()
        init_node = Node.make_node(init_state, None, 0, heuristic.get_heuristic_value(init_state), h_weight)
        self.open_list[init_state] = init_node
        while self.open_list:
            # print('list of values in h:\n', list(self.open_list.values())) 
            state, node = self.open_list.popitem()
            # print(node)
            # print("________________________________________________________-")
            self.close_list[state] = node  

            if env.is_final_state(state):
                actions, final_cost, expended =  Utility.solution(node, self.expended)
                final_cost = node.cost
                return actions, final_cost, expended

            self.expended += 1
            if node.cost != np.inf: # avoid holes
                for action, (child_state, cost, terminated) in env.succ(state).items():
                    total_cost = node.cost + cost
                    if child_state not in self.close_list.keys() and child_state not in self.open_list.keys():
                        child_heuristic = heuristic.get_heuristic_value(child_state)
                        child_node = Node.make_node(child_state, node, total_cost, child_heuristic, h_weight)
                        self.open_list[child_state] = child_node
                    elif child_state in self.open_list.keys():
                        if total_cost < self.open_list[child_state].cost:
                            self.open_list[child_state] = child_node # check why never get in
                    else:
                        if total_cost < self.close_list[child_state].cost:
                            self.open_list[child_state] = child_node  # check why never get in
                            self.close_list.pop(child_state)
        return ([], 0.0, 0)


# class AStarAgent(Agent):
#     def __init__(self):
#         super().__init__()

#     def search(self, env: CampusEnv, open_list: List[Node], close_list: List[Node]) -> Tuple[List[int], float, int]:
#         # Implement A* search logic here
#         pass

if __name__ == "__main__":
    MAPS = {
    "4x4": ["SFFF",
            "FHFH",
            "FFFH",
            "HFFG"],
    "4x4_2": ["SFFF",
            "FFFF",
            "FFFH",
            "FFHG"], # 0->4->8->12->13->14
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

    UCS_agent = UCSAgent()
    print("UCS")
    actions, total_cost, expanded = UCS_agent.search(env)
    print(f"Total_cost: {total_cost}")
    print(f"Expanded: {expanded}")
    print(f"Actions: {actions}")
    print()

    WAstar_agent = WeightedAStarAgent()
    print("WeightedAStarAgent")
    actions, total_cost, expanded = WAstar_agent.search(env, h_weight=1)
    print(f"Total_cost: {total_cost}")
    print(f"Expanded: {expanded}")
    print(f"Actions: {actions}")
    print()

