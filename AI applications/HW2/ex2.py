ids = ["212778229", "325069565"]

import networkx as nx
import logging

RESET_PENALTY = 50
REFUEL_PENALTY = 10
DROP_IN_DESTINATION_REWARD = 100
INIT_TIME_LIMIT = 300
TURN_TIME_LIMIT = 0.1



class OptimalTaxiAgent:
    def __init__(self, initial):
        self.initial = initial
        self.graph = self.build_graph()
        self.state = self.initial

    def act(self, state):
        raise NotImplemented

    def is_action_legal(self, action):
        """
        check if the action is legal
        """

        def _is_move_action_legal(move_action):
            taxi_name = move_action[1]
            if taxi_name not in self.state['taxis'].keys():
                return False
            if self.state['taxis'][taxi_name]['fuel'] == 0:
                return False
            l1 = self.state['taxis'][taxi_name]['location']
            l2 = move_action[2]
            return l2 in list(self.graph.neighbors(l1))

        def _is_pick_up_action_legal(pick_up_action):
            taxi_name = pick_up_action[1]
            passenger_name = pick_up_action[2]
            # check same position
            if self.state['taxis'][taxi_name]['location'] != self.state['passengers'][passenger_name]['location']:
                return False
            # check taxi capacity
            if self.state['taxis'][taxi_name]['capacity'] <= 0:
                return False
            # check passenger is not in his destination
            if self.state['passengers'][passenger_name]['destination'] == self.state['passengers'][passenger_name][
                'location']:
                return False
            return True

        def _is_drop_action_legal(drop_action):
            taxi_name = drop_action[1]
            passenger_name = drop_action[2]
            # check same position
            if self.state['taxis'][taxi_name]['location'] != self.state['passengers'][passenger_name]['destination']:
                return False
            return True

        def _is_refuel_action_legal(refuel_action):
            """
            check if taxi in gas location
            """
            taxi_name = refuel_action[1]
            i, j = self.state['taxis'][taxi_name]['location']
            if self.state['map'][i][j] == 'G':
                return True
            else:
                return False

        def _is_action_mutex(global_action):
            assert type(global_action) == tuple, "global action must be a tuple"
            # one action per taxi
            if len(set([a[1] for a in global_action])) != len(global_action):
                return True
            # pick up the same person
            pick_actions = [a for a in global_action if a[0] == 'pick up']
            if len(pick_actions) > 1:
                passengers_to_pick = set([a[2] for a in pick_actions])
                if len(passengers_to_pick) != len(pick_actions):
                    return True
            return False

        if action == "reset":
            return True
        if action == "terminate":
            return True
        if len(action) != len(self.state["taxis"].keys()):
            logging.error(f"You had given {len(action)} atomic commands, while there are {len(self.state['taxis'])}"
                          f" taxis in the problem!")
            return False
        for atomic_action in action:
            # illegal move action
            if atomic_action[0] == 'move':
                if not _is_move_action_legal(atomic_action):
                    logging.error(f"Move action {atomic_action} is illegal!")
                    return False
            # illegal pick action
            elif atomic_action[0] == 'pick up':
                if not _is_pick_up_action_legal(atomic_action):
                    logging.error(f"Pick action {atomic_action} is illegal!")
                    return False
            # illegal drop action
            elif atomic_action[0] == 'drop off':
                if not _is_drop_action_legal(atomic_action):
                    logging.error(f"Drop action {atomic_action} is illegal!")
                    return False
            # illegal refuel action
            elif atomic_action[0] == 'refuel':
                if not _is_refuel_action_legal(atomic_action):
                    logging.error(f"Refuel action {atomic_action} is illegal!")
                    return False
            elif atomic_action[0] != 'wait':
                return False
        # check mutex action
        if _is_action_mutex(action):
            logging.error(f"Actions {action} are mutex!")
            return False
        # check taxis collision
        if len(self.state['taxis']) > 1:
            taxis_location_dict = dict([(t, self.state['taxis'][t]['location']) for t in self.state['taxis'].keys()])
            move_actions = [a for a in action if a[0] == 'move']
            for move_action in move_actions:
                taxis_location_dict[move_action[1]] = move_action[2]
            if len(set(taxis_location_dict.values())) != len(taxis_location_dict):
                logging.error(f"Actions {action} cause collision!")
                return False
        return True

    def all_actions(self):
        """
        return all possible actions
        """
        all_actions = []
        for taxi_name in self.state['taxis'].keys():
            # move actions
            for neighbor in self.graph.neighbors(self.state['taxis'][taxi_name]['location']):
                all_actions.append(('move', taxi_name, neighbor))
            # pick up actions
            for passenger_name in self.state['passengers'].keys():
                if self.state['passengers'][passenger_name]['location'] == self.state['taxis'][taxi_name]['location']:
                    all_actions.append(('pick up', taxi_name, passenger_name))
            # drop off actions
            for passenger_name in self.state['passengers'].keys():
                if self.state['passengers'][passenger_name]['destination'] == self.state['taxis'][taxi_name]['location']:
                    all_actions.append(('drop off', taxi_name, passenger_name))
            # refuel actions
            i, j = self.state['taxis'][taxi_name]['location']
            if self.state['map'][i][j] == 'G':
                all_actions.append(('refuel', taxi_name))
            # wait actions
            all_actions.append(('wait', taxi_name))
        # reset action
        all_actions.append('reset')
        # terminate action
        all_actions.append('terminate')
        return all_actions

    def build_graph(self):
        """
        build the graph of the problem
        """
        n, m = len(self.initial_state['map']), len(self.initial_state['map'][0])
        g = nx.grid_graph((m, n))
        nodes_to_remove = []
        for node in g:
            if self.initial_state['map'][node[0]][node[1]] == 'I':
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            g.remove_node(node)
        return g


class TaxiAgent:
    def __init__(self, initial):
        self.initial = initial

    def act(self, state):
        raise NotImplemented
