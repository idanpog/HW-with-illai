import search
import random
import math

ids = ["212778229", "325968565"]


class TaxiProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        search.Problem.__init__(self, initial)

    def actions(self, state, all_mem=[], mem=()):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""

        if len(mem) == len(state['taxis']):  # checks if the current action is 'complete'
            all_mem.append(mem)  # and adds it to the list of possible actions
            return

        active_taxis = [action[1] for action in mem]  # lists all the active taxis
        possible_sub_actions = []
        for taxi, taxi_info in state['taxis'].items():
            if taxi not in active_taxis:
                # move
                for diff in [[1, 0], [-1, 0], [0, -1], [0, 1]]:
                    desired_loc = (taxi_info['location'][0] + diff[0], taxi_info['location'][1] + diff[1])
                    free_spot = True
                    for sub_action in mem:
                        if sub_action[1] == 'move' and sub_action[2] == desired_loc:
                            free_spot = False
                    try:
                        self.map[desired_loc[0]][desired_loc[1]]
                    except:
                        free_spot = False
                    if free_spot:
                        possible_sub_actions.append((('move', taxi, desired_loc),))

            must_move = False
            for sub_action in mem:
                if sub_action[1] == 'move' and sub_action[2] == taxi_info['location']:
                    must_move = True

            if not must_move:
                # pickup
                for passenger, passenger_info in state['passengers'].items():
                    if passenger_info['location'] == taxi_info['location']:
                        possible_sub_actions.append((('pick up', taxi, passenger),))
                # drop-off
                for passenger, passenger_info in state['passengers'].items():
                    if passenger_info['in_taxi'] == taxi and taxi_info['location'] == passenger_info['goal']:
                        possible_sub_actions.append((('drop off', taxi, passenger),))
                # refuel
                if self.map[taxi['location'][0]][taxi['location'][1]] == 'G':
                    possible_sub_actions.append((('refuel', taxi),))
                # wait
                possible_sub_actions.append((('wait', taxi),))

            # recursively expend the branches for each sub action
            for sub_action in possible_sub_actions:
                self.actions(state, all_mem, mem + sub_action)
        return tuple(all_mem)

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""

    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        passengers = state['passengers']
        for passenger in passengers:
            if passenger['location'] != passenger['destination']:
                return False
        return True

    def h(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate"""
        return 0

    def h_1(self, node):
        """
        This is a simple heuristic
        """

    def h_2(self, node):
        """
        This is a slightly more sophisticated Manhattan heuristic
        """

    """Feel free to add your own functions
    (-2, -2, None) means there was a timeout"""


class Node:
    def __init__(self, parent, state, action):
        self.parent = parent
        self.state = state
        self.action = action


def create_taxi_problem(game):
    return TaxiProblem(game)
