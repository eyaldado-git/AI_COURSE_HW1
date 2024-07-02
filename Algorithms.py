import time
import numpy as np
from typing import List, Tuple, Dict
import heapdict
from CampusEnv import CampusEnv

class Node:
    def __init__(self, state=0, parent=None, cost=0):
        self.state = state
        self.parent = parent
        self.cost = cost

    @staticmethod
    def make_node(state: int, parent=None, cost: float = 0):
        return Node(state, parent, cost)
    
    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state
    
    def __repr__(self) -> str:
         return f"Node({self.state}, cost={self.cost})"
        # return f"Node(state={self.state}, parent={self.parent}, cost={self.cost})"
    
    def __str__(self) -> str:
        return f"Node({self.state}, cost={self.cost})"
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.state == other.state and self.cost == other.cost

    def __lt__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        if self.cost == other.cost:
            return self.state < other.state
        return self.cost < other.cost

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

class Agent:
    def __init__(self):
        self.actions = []
        self.cost = 0.0
        self.expended = 0
        self.open_list = heapdict.heapdict() 
        self.close_list = []

    def pop_from_open_list(self):
        return self.open_list.peekitem()
    
    def initialize_search(self, env: CampusEnv) -> None:
        init_node = Node.make_node(env.get_initial_state())
        self.open_list[init_node.state] = init_node
        self.close_list = []

    def search(self, env) -> Tuple[List[int], float, int]:
        self.initialize_search(env)
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
        
        return 

class UCSAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

# class WeightedAStarAgent(Agent):
#     def __init__(self):
#         super().__init__()

#     def search(self, env: CampusEnv, open_list: List[Node], close_list: List[Node], h_weight: float) -> Tuple[List[int], float, int]:
#         # Implement Weighted A* search logic here
#         pass


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
    print("UCS\n")
    actions, total_cost, expanded = UCS_agent.search(env)
    print(f"Total_cost: {total_cost}")
    print(f"Expanded: {expanded}")
    print(f"Actions: {actions}")  # i have error in the optimal path, should be 98 and i got 101

