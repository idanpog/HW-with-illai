ids = ["212778229", "325069565"]

import networkx as nx
import logging
from copy import deepcopy
from collections import defaultdict
from utils import FIFOQueue
import itertools
import random
import numpy as np
from tqdm import tqdm
from time import time

RESET_PENALTY = 50
REFUEL_PENALTY = 10
DROP_IN_DESTINATION_REWARD = 100
INIT_TIME_LIMIT = 300
TURN_TIME_LIMIT = 0.1

# random.seed(10)


TAXIS = 0
# Taxi params
NAME = 0
LOC = 1
FUEL = 2
CAP = 3

PASSENGERS = 1
# Passenger params
DESTINATION = 2
POSSIBLE_DESTINATIONS = 3
PROBABILITY = 4

TURNS_TO_GO = 2


def init_to_tup(initial):
    """converts the initial state to a tuple"""
    taxis = initial["taxis"]
    passengers = initial["passengers"]
    passengers_tups = tuple([(passenger, passengers[passenger]["location"],
                              passengers[passenger]["destination"], passengers[passenger]['possible_goals'],
                              ((len(passengers[passenger]['possible_goals']) - 1) / (
                                  len(passengers[passenger]['possible_goals'])) *
                               passengers[passenger]['prob_change_goal'])) for passenger in passengers])
    taxis_tups = tuple([(taxi, taxis[taxi]["location"], taxis[taxi]["fuel"], taxis[taxi]["capacity"])
                        for taxi in taxis])
    initial = (taxis_tups, passengers_tups, initial["turns to go"])
    return initial


def passenger_name_to_id(initial):
    d = {}
    for idx, passenger in enumerate(initial[PASSENGERS]):
        d[passenger[NAME]] = idx
    return d


def taxi_name_to_id(initial):
    d = {}
    for idx, taxi in enumerate(initial[TAXIS]):
        d[taxi[NAME]] = idx
    return d


def build_graph(map):
    """
    build the graph of the problem
    """
    n, m = len(map), len(map[0])
    g = nx.grid_graph((m, n))
    nodes_to_remove = []
    for node in g:
        type = map[node[0]][node[1]]
        if type == 'I':
            nodes_to_remove.append(node)
        g.nodes[node]['type'] = type

    for node in nodes_to_remove:
        g.remove_node(node)
    return g


class OptimalTaxiAgent:
    def __init__(self, initial):
        self.initial = init_to_tup(initial)
        self.tName2id = taxi_name_to_id(self.initial)
        self.pName2id = passenger_name_to_id(self.initial)
        # self.add_change_prob(self.initial)
        self.graph = initial['graph'] if 'graph' in initial else build_graph(initial["map"])
        self.state = self.initial
        self.all_actions_dict = dict()
        self.next_dict = dict()
        self.inner_prob_dict = dict()
        # self.policy = self.policy_iterations(max_iterations=self.initial[TURNS_TO_GO])

        self.policy = self.value_iterations(max_iterations=self.initial[TURNS_TO_GO])
        # print(f"the expected value is {self.policy[self.initial]}")

    def taxi_name_to_id(self):
        d = {}
        for idx, taxi in enumerate(self.initial[TAXIS]):
            d[taxi[NAME]] = idx
        return d

    def passenger_name_to_id(self):
        d = {}
        for idx, passenger in enumerate(self.initial[PASSENGERS]):
            d[passenger[NAME]] = idx
        return d

    def next(self, state, action):
        """runs the given action form the given state and returns the new state"""
        return self.apply(state, action)

    def apply(self, state, action):
        """
        apply the action to the state
        """
        if action[0] == "reset":
            return self.initial[0], self.initial[1], state[TURNS_TO_GO] - 1
        next = (state[TAXIS], state[PASSENGERS], state[TURNS_TO_GO] - 1)
        if action[0] == "terminate":
            return None
        for atomic_action in action:
            next = self.apply_atomic_action(next, atomic_action)
        return next

    def state_to_tup(self, state):
        """converts a state to a tuple"""
        return tuple([tuple(tup) for tup in state[TAXIS]]), tuple([tuple(tup) for tup in state[PASSENGERS]]), state[
            TURNS_TO_GO]

    def apply_atomic_action(self, state, atomic_action):
        """
        apply an atomic action to the state
        """
        if atomic_action[0] == 'wait':
            return state
        old_state = state
        state = [[list(tup) for tup in old_state[TAXIS]], [list(tup) for tup in old_state[PASSENGERS]],
                 old_state[TURNS_TO_GO]]
        taxi_name = atomic_action[1]
        if atomic_action[0] == 'move':
            state[TAXIS][self.tName2id[taxi_name]][LOC] = atomic_action[2]
            state[TAXIS][self.tName2id[taxi_name]][FUEL] -= 1
            return self.state_to_tup(state)
        elif atomic_action[0] == 'pick up':
            passenger_name = atomic_action[2]
            state[TAXIS][self.tName2id[taxi_name]][CAP] -= 1
            state[PASSENGERS][self.pName2id[passenger_name]][LOC] = taxi_name
            return self.state_to_tup(state)
        elif atomic_action[0] == 'drop off':
            passenger_name = atomic_action[2]
            state[PASSENGERS][self.pName2id[passenger_name]][LOC] = state[TAXIS][self.tName2id[taxi_name]][LOC]
            state[TAXIS][self.tName2id[taxi_name]][CAP] += 1
            return self.state_to_tup(state)
        elif atomic_action[0] == 'refuel':
            state[TAXIS][self.tName2id[taxi_name]][FUEL] = self.initial[TAXIS][self.tName2id[taxi_name]][FUEL]
            return self.state_to_tup(state)
        # else:
        #     raise NotImplemented

    def reward(self, state, action):
        """
        return the reward of performing the action in the state
        """
        if action == ('reset',):
            return -RESET_PENALTY
        if action == 'terminate':
            return 0

        reward = 0
        for atomic_action in action:
            if atomic_action == 'refuel':
                reward -= REFUEL_PENALTY
            if atomic_action[0] == 'drop off':
                # passenger_name = atomic_action[2]
                # if state[PASSENGERS][self.pName2id[passenger_name]][LOC] == \
                #         state[PASSENGERS][self.pName2id[passenger_name]][DESTINATION]:
                reward += DROP_IN_DESTINATION_REWARD
        return reward

    def act(self, state):
        """
        return the action to perform in the state
        """
        state = init_to_tup(state)
        action = self.policy[state]
        return action if action != ("reset",) else "reset"

    def all_actions(self, state):
        if state not in self.all_actions_dict:
            self.all_actions_dict[state] = self.all_actions_aux(state)
        return self.all_actions_dict[state]

    def extract_locations(self, action, state):
        """extract the taxi locations from an action"""
        locations = []
        for atomic_action in action:
            if atomic_action[0] == 'move':
                locations.append(atomic_action[2])
            else:
                locations.append(state[TAXIS][self.tName2id[atomic_action[1]]][LOC])
        return locations

    def clean_collisions(self, actions, state):
        """
        remove collisions from the actions
        """
        new_actions = [action for action in actions if
                       len(self.extract_locations(action, state)) == len(set(self.extract_locations(action, state)))]

        return new_actions

    def all_actions_aux(self, state):
        """
        return all possible actions
        """
        all_actions = []
        taxi_actions = {}
        for taxi in state[TAXIS]:
            taxi_actions[taxi[NAME]] = []
            taxi_name = taxi[NAME]
            # move actions
            if taxi[FUEL] > 0:
                for neighbor in self.graph.neighbors(taxi[LOC]):
                    taxi_actions[taxi_name].append(('move', taxi_name, neighbor))
            # pick up actions
            for passenger in state[PASSENGERS]:
                # passenger = state[PASSENGERS][self.pName2id[passenger_name]]
                # passenger_name = passenger_name[NAME]
                if passenger[LOC] == taxi[LOC] and taxi[CAP] > 0 and passenger[DESTINATION] != passenger[LOC]:
                    taxi_actions[taxi_name].append(('pick up', taxi_name, passenger[NAME]))
                # drop off actions
                if passenger[LOC] == taxi_name and passenger[DESTINATION] == taxi[LOC]:
                    taxi_actions[taxi_name].append(('drop off', taxi_name, passenger[NAME]))
            # refuel actions
            i, j = state[TAXIS][self.tName2id[taxi_name]][LOC]
            if self.graph.nodes[(i, j)]['type'] == 'G':
                taxi_actions[taxi_name].append(('refuel', taxi_name))
            # wait actions
            taxi_actions[taxi_name].append(('wait', taxi_name))
        # reset action
        all_actions = list(itertools.product(*taxi_actions.values()))
        if len(state[TAXIS]) > 1:
            all_actions = self.clean_collisions(all_actions, state)
        if (('move', 'taxi 1', (2, 2)), ('wait', 'taxi 2')) in all_actions and state[TAXIS][1][LOC] == (2, 2):
            print("something's weird")
            # if (('move', 'taxi 1', (2, 2)), ('wait', 'taxi 2')) in all_actions and state[TAXIS][1][LOC] == (2, 2):
            #     x = 1
            # for action in all_actions:
            #     flag = False
            #     for taxi_action in action:
            #         if flag:
            #             break
            #         for taxi2_action in action:
            #             if taxi_action[1] != taxi2_action[1]:
            #                 if (taxi_action[0] == 'move' and taxi2_action[0] == 'move' and taxi_action[2] == taxi2_action[
            #                     2]) or \
            #                         (taxi_action[0] == 'move' and taxi2_action[0] != 'move' and taxi_action[2] ==
            #                          state[TAXIS][self.tName2id[taxi2_action[1]]][LOC]):
            #                     all_actions.remove(action)
            #                     flag = True
            #                     break

        all_actions.append(('reset',))
        return all_actions

    def generate_all_states(self):
        """uses all_actions to generate all possible states, kinda runs BFS"""
        turns_to_go = self.state[TURNS_TO_GO] + 1
        all_states = defaultdict(lambda: set())
        all_states[0] = set((self.state,))
        for i in range(1, turns_to_go + 1):
            # diff = new_states.difference(old_states)
            # old_states = new_states.copy()
            count = 0
            for state in all_states[i - 1]:
                self.all_actions_dict[state] = {}
                self.next_dict[state] = {}
                count += 1
                for action in self.all_actions_aux(state):
                    new_state = self.next(state, action)
                    self.next_dict[state][action] = new_state
                    all_new_states = self.split_across_MDP(new_state)
                    all_states[i].update(all_new_states)
                    if action in self.all_actions_dict[state]:
                        self.all_actions_dict[state][action].append(list(all_new_states))
                    else:
                        self.all_actions_dict[state][action] = list(all_new_states)
        return all_states

    # action = (('move', 'taxi 1', (1,0)), ('move', 'taxi 2', (0,0)))

    # ((('taxi 1', (1, 0), 2, 1), ('taxi 2', (0, 0), 2, 1)), (('Dana', (0, 2), (2, 2), ((2, 2),), 0.0), ('Dan', (2, 0), (2, 2), ((2, 2),), 0.0)), 99)

    def split_across_MDP(self, state):
        """
        split the state across MDPs
        """
        new_states = set()
        if len(state[PASSENGERS]) > 1:
            possible_goals = [state[PASSENGERS][self.pName2id[passenger_name]][POSSIBLE_DESTINATIONS] for passenger_name
                              in
                              [passenger[NAME] for passenger in state[PASSENGERS]]]

            for goals in itertools.product(*possible_goals):
                pass_list = list(list(passenger) for passenger in state[PASSENGERS])
                for i, goal in enumerate(goals):
                    pass_list[i][DESTINATION] = goal
                new_state = (state[TAXIS], tuple(tuple(passenger) for passenger in pass_list), state[TURNS_TO_GO])
                # _state = deepcopy(state)
                # for goal, passenger in zip(goals, _state[PASSENGERS]):
                #     passenger[DESTINATION] = goal
                new_states.add(new_state)
        else:
            passenger = state[PASSENGERS][0]
            for goal in passenger[POSSIBLE_DESTINATIONS]:
                new_state = (state[TAXIS], (
                    (passenger[NAME], passenger[LOC], goal, passenger[POSSIBLE_DESTINATIONS], passenger[PROBABILITY]),),
                             state[TURNS_TO_GO])
                new_states.add(new_state)
        return new_states

    def build_graph(self):
        """
        build the graph of the problem
        """
        n, m = len(self.map), len(self.map[0])
        g = nx.grid_graph((m, n))
        nodes_to_remove = []
        for node in g:
            if self.map[node[0]][node[1]] == 'I':
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            g.remove_node(node)
        return g

    def inner(self, state, action, new_state, old_values):
        """returns the probability of the new state given the state and action"""
        if (state, action, new_state) in self.inner_prob_dict:
            return self.inner_prob_dict[(state, action, new_state)] * old_values[new_state]
        else:
            prob = 1
            # next_state = self.next(state, action)
            next_state = self.next_dict[state][action]
            for curr_passenger, new_passenger in zip(next_state[PASSENGERS], new_state[PASSENGERS]):
                if curr_passenger[DESTINATION] != new_passenger[DESTINATION]:
                    prob *= curr_passenger[PROBABILITY]
                else:
                    prob *= (1 - curr_passenger[PROBABILITY])
            self.inner_prob_dict[(state, action, new_state)] = prob
            return self.inner_prob_dict[(state, action, new_state)] * old_values[new_state]

    def expected_value(self, state, action, old_values):
        """
        return the expected value of the action in the state
        """
        if state[TURNS_TO_GO] == 0:
            return 0
        next_states = self.all_actions_dict[state][action]
        reward = self.reward(state, action)
        ex = sum([self.inner(state, action, next_state, old_values) for next_state in next_states])
        return reward + ex

    def policy_iterations(self, max_iterations=200, gamma=0.9, epsilon=1e-21):
        """
        policy iteration algorithm
        """
        # initialize value function
        all_state_list = self.generate_all_states()
        best_action = {}
        num = sum([len(list(all_state_list[i])) for i in range(max_iterations + 1)])
        # values = dict([(s, 0) for s in all_state_list])
        # values = defaultdict(lambda: 0)
        values = dict([(s, 0) for i in range(max_iterations + 1) for s in all_state_list[i]])
        # initialize policy
        policy = dict(
            [(s, random.choice(self.all_actions(s))) for i in range(max_iterations + 1) for s in all_state_list[i]])
        # initialize iteration counter
        counter = 0
        # initialize delta
        delta = float('inf')
        for iter in (tq := tqdm(range(max_iterations), leave=True, position=0)):
            if delta <= epsilon:
                print("-------------- broke --------------")
                break
            delta = 0
            old_values = values.copy()
            for i in range(max_iterations + 1):
                for state in all_state_list[i]:
                    old_value = old_values[state]
                    counter += 1
                    for action in self.all_actions_dict[state].keys():
                        expected = self.expected_value(state, action, gamma, old_values)
                        if expected > values[state]:
                            values[state] = expected
                            policy[state] = action

                    values[state] = self.expected_value(state, action, gamma, old_values)
                    delta = max(delta, abs(old_value - values[state]))
            tq.set_description(f"Value Iterations")

        return policy

    def value_iterations(self, max_iterations=100):
        """
        value iteration algorithm, using dynamic programming
        """
        all_state_list = self.generate_all_states()
        # print(sum([len(list(all_state_list[i])) for i in range(max_iterations + 1)]))
        values = dict([(s, 0) for i in range(max_iterations + 1) for s in all_state_list[i]])
        policy = dict(
            [(s, random.choice(list(self.all_actions_dict[s].keys()))) for i in range(max_iterations + 1) for s in
             all_state_list[i]])
        # for iter in (tq := tqdm(range(max_iterations), leave=True, position=0)):
        start = time()
        for i in range(max_iterations - 1, -1, -1):
            for state in all_state_list[i]:
                actions = list(self.all_actions_dict[state].keys())
                action_values = [self.expected_value(state, a, values) for a in actions]
                values[state] = max(action_values)
                policy[state] = actions[np.argmax(action_values)]
        end = time()
        #print("Time taken: ", end - start)
        #print(f"{values[list(all_state_list[0])[0]]=}")
        return policy


class TaxiAgent:
    def __init__(self, initial):
        self.map = initial["map"]
        self.initial_dict = initial
        self.initial = init_to_tup(initial)
        self.tName2id = taxi_name_to_id(self.initial)
        self.pName2id = passenger_name_to_id(self.initial)
        # self.add_change_prob(self.initial)
        self.graph = build_graph(self.map)
        self.state = self.initial
        self.shortest_paths_len = dict(nx.all_pairs_shortest_path_length(self.graph))
        self.shortest_paths = dict(nx.all_pairs_shortest_path(self.graph))
        self.assignments = self.associate_passengers()
        self.sub_graphs = self.get_sub_graphs(self.map, self.graph, self.assignments)
        self.sub_agents = self.initiate_sub_agents(self.sub_graphs, self.assignments)
        self.active = [assignment[0] for assignment in self.assignments]
        self.passive = [taxi for taxi in self.initial_dict['taxis'].keys() if taxi not in self.active]

    def initiate_sub_agents(self, sub_graphs, assignments):
        """returns a dict that holds the sub agents as values and the tuples (taxi_name, passenger_name)
         as keys"""
        sub_agents = {}
        for taxi, passenger in assignments:
            thingy = {}
            thingy["passengers"] = {passenger: self.initial_dict["passengers"][passenger]}
            thingy["taxis"] = {taxi: self.initial_dict["taxis"][taxi]}
            thingy["turns to go"] = self.initial_dict['turns to go']
            thingy['graph'] = sub_graphs[(taxi, passenger)]
            sub_agents[(taxi, passenger)] = OptimalTaxiAgent(thingy)
        return sub_agents

    def get_sub_graphs(self, map, graph, assignments):
        """creates a sub map for each pair of taxi and passenger which includes
        the shortest paths between the taxi, the nearest gas station, the passenger and a chosen destination"""

        def _path_len(taxi, passenger, pair):
            """spits out the shortest possible path length that allows the taxi to pickup and deliver the passenger
            attempts:
            1. taxi -> passenger -> gas station -> destination
            2. taxi -> gas station -> passenger -> destination
            3. taxi -> gas station -> passenger -> gas station -> destination"""
            dest, gas = pair
            ploc, tloc = passenger[LOC], taxi[LOC]
            fuel = taxi[FUEL]
            cap = self.initial[TAXIS][self.tName2id[taxi[NAME]]][FUEL]
            options = [float("inf")]
            # if pick up and then refuel
            if self.d(tloc, ploc) + self.d(ploc, gas) <= fuel:
                options.append(self.d(tloc, ploc) + self.d(passenger, gas) + self.d(gas, dest))
            # else refuel then pick up
            # if dest right after the pickup
            elif self.d(tloc, gas) <= FUEL:
                if self.d(gas, ploc) + self.d(ploc, dest) <= cap:
                    options.append(self.d(tloc, gas) + self.d(gas, ploc) + self.d(ploc, dest))
                # in this case we need to refuel before and after the pickup
                elif self.d(gas, ploc) + self.d(ploc, gas) <= cap and self.d(gas, dest) <= cap:
                    options.append(self.d(tloc, gas) + self.d(gas, ploc) + self.d(ploc, gas) + self.d(gas, dest))
            return min(options)

        def _choose_gas_and_dest(map, graph, taxi, passenger):
            """chooses the gas station and destination for the sub map
            first checks if there's a possibility to get the passenger to a destination without refueling
            otherwise, finds the gas stations that allows the taxi to get to the passenger and the destination
            with minimal steps"""
            # attempt to avoid gas stations
            d_without_fuel = lambda dest: self.d(taxi[LOC], passenger[LOC]) + self.d(passenger[LOC], dest)
            destination = min(passenger[POSSIBLE_DESTINATIONS], key=d_without_fuel)
            if d_without_fuel(destination) <= taxi[FUEL]:
                return destination, destination

            # choose best pair of gas station and destination
            gas_stations = [(i, j) for i in range(len(map)) for j in range(len(map[0])) if map[i][j] == "G"]
            dest_gas_pairs = [(dest, gas) for dest in passenger[POSSIBLE_DESTINATIONS] for gas in gas_stations]
            pair = min(dest_gas_pairs, key=lambda pair: _path_len(taxi, passenger, pair))
            return pair

        def _nodes_to_keep(map, graph, taxi, passenger):
            """returns the nodes to keep while building a subgraph that contains the shortest paths"""
            nodes_to_keep = set()
            shortest_path_pairs = []
            dest_loc, gas_loc = _choose_gas_and_dest(map, graph, taxi, passenger)
            shortest_path_pairs.append((taxi[LOC], passenger[LOC]))
            shortest_path_pairs.append((passenger[LOC], gas_loc))
            shortest_path_pairs.append((taxi[LOC], gas_loc))
            shortest_path_pairs.append((gas_loc, dest_loc))
            shortest_path_pairs.append((passenger[LOC], dest_loc))
            for pair in shortest_path_pairs:
                if pair[0] in self.shortest_paths and pair[1] in self.shortest_paths[pair[0]]:
                    nodes_to_keep.update(self.shortest_paths[pair[0]][pair[1]])
            return nodes_to_keep

        sub_maps = {}
        for taxi_name, passenger_name in assignments:
            taxi = self.state[TAXIS][self.tName2id[taxi_name]]
            passenger = self.state[PASSENGERS][self.pName2id[passenger_name]]
            nodes_to_keep = _nodes_to_keep(map, graph, taxi, passenger)
            sub_map = graph.subgraph(nodes_to_keep)
            sub_maps[(taxi_name, passenger_name)] = sub_map

        return sub_maps

    def d(self, loc1, loc2):
        if loc1 in self.shortest_paths_len and loc2 in self.shortest_paths_len[loc1]:
            return self.shortest_paths_len[loc1][loc2]
        else:
            return np.inf

    def associate_passengers(self):
        """
        associate the passengers with the taxis
        returns tuple of tuples (taxi_name, passenger_name)
        """

        def _generate_possible_assignments():
            """
            generate all possible assignments
            pairs of taxi and passenger names
            returns a tuple of tuples of tuples
            (((tname),(pname)..), ((),(),..), ..)
            """
            taxi_names = list(self.tName2id.keys())
            passenger_names = list(self.pName2id.keys())
            valid_assignments = set()
            permutations = list(itertools.permutations(passenger_names))
            # unique_combinations = []
            # permute = itertools.permutations(taxi_names, min(len(passenger_names), len(taxi_names)))
            # for comb in permute:
            #     zipped = zip(comb, passenger_names)
            #     unique_combinations.append(list(zipped))
            # num_premutations = (len(self.initial[TAXIS]) * len(self.initial[PASSENGERS])) ** 3
            # permute = itertools.permutations(taxi_names, min(len(passenger_names), len(taxi_names)))
            for permutation in permutations:
                # for comb in permute:
                #     zipped = zip(comb, passenger_names)
                #     unique_combinations.append(list(zipped))
                # for comb in unique_combinations:
                #     valid_assignments.add(tuple(comb))
                valid_assignments.add(tuple(zip(taxi_names, permutation)))
            # unique_combinations = [(tname, pname) for pname in passenger_names for tname in taxi_names]
            return tuple(valid_assignments)

        def _evaluate_assignment(assignment):
            """
            ignores fuel
            gives a score to the assignment based on the max distance between a pair of a taxi and a passenger
            smaller is better
            """
            score = 0
            for taxi_name, passenger_name in assignment:
                taxi = self.state[TAXIS][self.tName2id[taxi_name]]
                passenger = self.state[PASSENGERS][self.pName2id[passenger_name]]
                score = max(score, self.d(taxi[LOC], passenger[LOC]))
            return score

        possible_assignments = _generate_possible_assignments()
        best_assignment = min(possible_assignments, key=_evaluate_assignment)
        return best_assignment

    def act_per_sub_graph(self, passenger, taxi, graph):
        """
        act for a single taxi
        if taxi carries a passenger and reached the passenger location, drop off and wait till he changes loc
        if taxi doesn't have a dedicated passenger, wait
        if taxi has a dedicated passenger, go towards the passenger
        """

    def extract_locations(self, action, state):
        """extract the taxi locations from an action"""
        locations = []
        for atomic_action in action:
            if atomic_action[0] == 'move':
                locations.append(atomic_action[2])
            else:
                locations.append(state[TAXIS][self.tName2id[atomic_action[1]]][LOC])
        return locations
    def _move_inactive(self, resolved_actions, state, avoid):
        """moves all the taxis that aren't assigned to a passenger"""
        for i in range(len(resolved_actions)):
            if resolved_actions[i][1] in self.passive:
                tname = resolved_actions[i][1]
                new_loc = self._random_step(state['taxis'][tname], avoid)
                if new_loc is not None:
                    resolved_actions[i] = ('move', resolved_actions[i][1], new_loc)
        return resolved_actions
    def _random_step(self, taxi, avoid):
        """returns a random step in the map"""
        loc = taxi["location"]
        neighbors = self.graph.neighbors(loc)
        cool_neighbors = [n for n in neighbors if n not in avoid]
        if cool_neighbors == []:
            return None
        decision = random.choice(cool_neighbors)
        return decision

    def act(self, state):
        try:
            actions = []
            for taxi, passenger in self.assignments:
                sub_state = {}
                reset_count = 0
                sub_state["passengers"] = {passenger: state["passengers"][passenger]}
                sub_state["taxis"] = {taxi: state["taxis"][taxi]}
                sub_state["turns to go"] = state["turns to go"]
                sub_state['graph'] = self.sub_graphs[(taxi, passenger)]
                action = self.sub_agents[(taxi, passenger)].act(sub_state)
                if action == 'reset':
                    reset_count +=1
                action = action[0] if action != 'reset' else ('wait', taxi)
                actions.append(action)
            acted = [action[1] for action in actions]
            for taxi in state["taxis"]:
                if taxi not in acted:
                    actions.append(('wait', taxi))

            resolved_actions = []
            locations = set()

            locations.update([state["taxis"][action[1]]["location"] for action in actions if action[0]=='wait'])
            for atomic_action in [action for action in actions if action[0] != 'wait']:
                location = atomic_action[2] if atomic_action[0] == "move" else state["taxis"][atomic_action[1]]["location"]
                if location not in locations:
                    locations.add(location)
                    resolved_actions.append(atomic_action)
                else:
                    resolved_actions.append(('wait', atomic_action[1]))
            resolved_actions += [action for action in actions if action[0] == 'wait']
            if [action[0] for action in resolved_actions].count('wait')==len(resolved_actions):
                taxi_locations = [taxi["location"] for taxi in state["taxis"].values()]
                resolved_actions = self._move_inactive(resolved_actions, state, taxi_locations)
            resolved_actions = tuple(resolved_actions)

            if reset_count >= len(actions) * 0.7:
                resolved_actions = "reset"
            return resolved_actions
        except KeyError:
            return "terminate"