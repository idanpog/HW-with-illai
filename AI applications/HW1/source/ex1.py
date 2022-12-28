import search
import random
import math
import json
from itertools import product
import collections
from collections import defaultdict
import networkx as nx
import numpy as np

ids = ["212778229", "325968565"]

TAXIS = 0
# Taxi params
NAME = 0
LOC = 1
FUEL = 2
CAP = 3
NUM_PASSENGERS = 4

PASSENGERS = 1
# Passenger params
DESTINATION = 2
IN_TAXI = 3


class TaxiProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        self.dist_mat = defaultdict(int)
        self.max_fuel = {}
        self.map = initial["map"]
        taxis = initial["taxis"]
        self.pnums = {}
        for i, passenger in enumerate(initial['passengers']):
            self.pnums[passenger] = i
        passengers = initial["passengers"]
        passengers_tups = tuple([(passenger, passengers[passenger]["location"],
                                  passengers[passenger]["destination"], 'null') for passenger in passengers])
        taxis_tups = tuple([(taxi, taxis[taxi]["location"], taxis[taxi]["fuel"], taxis[taxi]["capacity"], 0)
                            for taxi in taxis])
        initial = (taxis_tups, passengers_tups)
        for taxi in initial[TAXIS]:
            self.max_fuel[taxi[NAME]] = taxi[FUEL]
        self.map_size = (len(self.map) - 1, len(self.map[0]) - 1)
        self.shortest_paths = self.create_shortest_paths_dict()
        self.nearest_gas_station = self.create_nearest_gas_dict()
        search.Problem.__init__(self, initial)

    def add_edge(self, mat, loc1, loc2):
        mat[self.translate_to_index(loc1)][self.translate_to_index(loc2)] = 1
        mat[self.translate_to_index(loc2)][self.translate_to_index(loc1)] = 1

    def translate_to_index(self, loc):
        return loc[0] * (self.map_size[1] + 1) + loc[1]

    def create_shortest_paths_dict(self):
        n = (self.map_size[0] + 1) * (self.map_size[1] + 1)
        admat = np.zeros((n, n))
        for loc in product(range(self.map_size[0] + 1), range(self.map_size[1] + 1)):
            for diff in [[1, 0], [-1, 0], [0, -1], [0, 1]]:
                desired_loc = (loc[0] + diff[0], loc[1] + diff[1])
                if self.check_in_map(desired_loc):
                    if self.map[desired_loc[0]][desired_loc[1]] != 'I' and self.map[loc[0]][loc[1]] != "I":
                        self.add_edge(admat, desired_loc, loc)

        graph = nx.from_numpy_matrix(admat)

        return dict(nx.all_pairs_shortest_path_length(graph))

    def create_nearest_gas_dict(self):
        output = {}
        for loc in product(range(self.map_size[0] + 1), range(self.map_size[1] + 1)):
            output[loc] = np.inf
            for g_loc in product(range(self.map_size[0] + 1), range(self.map_size[1] + 1)):
                if self.map[g_loc[0]][g_loc[1]] == "G":
                    output[loc] = min(output[loc], self.d(loc, g_loc) - 1)
        return output

    def silver_shortest_gas(self):
        G1 = nx.Graph()
        G1.add_node((-1, -1))

        for y in range(len(self.map)):
            for x in range(len(self.map[y])):
                if (self.map[y][x] != 'I'):
                    G1.add_node((y, x))
                    if (self.map[y][x] == 'G'):
                        G1.add_edge((y, x), (-1, -1))

        for y in range(len(self.map)):
            for x in range(len(self.map[y])):
                if G1.has_node((y, x)):
                    if G1.has_node((y + 1, x)):
                        G1.add_edge((y, x), (y + 1, x))

                    if G1.has_node((y - 1, x)):
                        G1.add_edge((y, x), (y - 1, x))

                    if G1.has_node((y, x + 1)):
                        G1.add_edge((y, x), (y, x + 1))

                    if G1.has_node((y, x - 1)):
                        G1.add_edge((y, x), (y, x - 1))
        output = dict(nx.all_pairs_shortest_path_length(G1))
        return output

    def actions(self, state):
        taxis = state[TAXIS]
        passengers = state[PASSENGERS]
        per_taxi_sub_actions = []

        for taxi in taxis:
            per_taxi_sub_actions.append(self.generate_possible_sub_actions_for_taxi(taxi, passengers, state))

        unfiltered_actions = list(product(*per_taxi_sub_actions))
        filtered_actions = list(set(filter(lambda x: self.check_legit(state, x), unfiltered_actions)))

        return filtered_actions

    def generate_possible_sub_actions_for_taxi(self, taxi, passengers, state):
        """
        takes a taxi and returns all its possible moves (and a little more)
        its a relaxation because it ignores the locations of the other taxis
        """
        possible_sub_actions = []
        # move
        for diff in [[1, 0], [-1, 0], [0, -1], [0, 1]]:
            desired_loc = (taxi[LOC][0] + diff[0], taxi[LOC][1] + diff[1])
            if self.check_in_map(desired_loc):
                free_spot = (self.map[desired_loc[0]][desired_loc[1]] != 'I')
            else:
                free_spot = False

            if free_spot and taxi[FUEL] > 0:
                possible_sub_actions.append(('move', taxi[NAME], desired_loc))

        # pickup
        for passenger in passengers:
            if passenger[LOC] == taxi[LOC] and taxi[CAP] > taxi[NUM_PASSENGERS]:
                possible_sub_actions.append(('pick up', taxi[NAME], passenger[NAME]))
        # drop-off
        for passenger in passengers:
            if passenger[IN_TAXI] == taxi[NAME] and taxi[LOC] == passenger[DESTINATION]:
                possible_sub_actions.append(('drop off', taxi[NAME], passenger[NAME]))
        # refuel
        if self.map[taxi[LOC][0]][taxi[LOC][1]] == 'G':
            possible_sub_actions.append(('refuel', taxi[NAME]))
        # wait
        possible_sub_actions.append(('wait', taxi[NAME]))
        return possible_sub_actions

    def check_legit(self, state, action):
        # makes sure there are no taxis that collide
        taxi_locations = []
        for sub_action in action:
            taxi_num = int(sub_action[1][-1])
            if sub_action[0] in ("wait", "refuel", 'pick up', 'drop off'):
                taxi_locations.append(tuple(state[TAXIS][taxi_num - 1][LOC]))
            if sub_action[0] == 'move':
                taxi_locations.append(sub_action[2])

        return len(taxi_locations) == len(set(taxi_locations))

    def check_in_map(self, desired_loc):
        if self.map_size[0] < desired_loc[0] or 0 > desired_loc[0]:
            return False
        if self.map_size[1] < desired_loc[1] or 0 > desired_loc[1]:
            return False
        return True

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        old_state = state
        state = [[list(tup) for tup in old_state[TAXIS]], [list(tup) for tup in old_state[PASSENGERS]]]
        if action not in self.actions(state):
            return -1
        for taxi_action in action:
            act, taxi = taxi_action[0], taxi_action[1]
            taxi_num = int(taxi[-1]) - 1
            if act == "move":
                state[TAXIS][taxi_num][LOC] = taxi_action[2]
                state[TAXIS][taxi_num][FUEL] -= 1
            elif act == "pick up":
                state[TAXIS][taxi_num][NUM_PASSENGERS] += 1
                state[PASSENGERS][self.pnums[taxi_action[2]]][IN_TAXI] = taxi
            elif act == "drop off":
                state[TAXIS][taxi_num][NUM_PASSENGERS] -= 1
                state[PASSENGERS][self.pnums[taxi_action[2]]][IN_TAXI] = 'null'
            elif act == "refuel":
                state[TAXIS][taxi_num][FUEL] = self.max_fuel[taxi]
        for passenger in state[PASSENGERS]:
            if passenger[IN_TAXI] != 'null':
                passenger[LOC] = state[TAXIS][int(passenger[IN_TAXI][-1]) - 1][LOC]
        taxis = tuple([tuple(taxi) for taxi in state[TAXIS]])
        passengers = tuple([tuple(passenger) for passenger in state[PASSENGERS]])
        return tuple([taxis, passengers])

    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        passengers = state[PASSENGERS]
        for passenger in passengers:
            if passenger[LOC] != passenger[DESTINATION] or passenger[IN_TAXI] != 'null':
                return False
        return True

    def d(self, loc1, loc2):
        # return self.MD(loc1,loc2)
        if self.translate_to_index(loc1) in self.shortest_paths and \
                self.translate_to_index(loc2) in self.shortest_paths[self.translate_to_index(loc1)]:
            return self.shortest_paths[self.translate_to_index(loc1)][self.translate_to_index(loc2)]
        else:
            return np.inf

    def h(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate"""
        # return max(self.max_manhatten(node), self.h_1(node), self.h_2(node))
        return self.h_silver(node)

    def max_passenger(self, node):
        total_distances = []
        for passenger in node.state[PASSENGERS]:
            taxi_distances = []
            for taxi in node.state[TAXIS]:
                if taxi[CAP] > taxi[NUM_PASSENGERS] or taxi[NAME] == passenger[IN_TAXI]:  # default case when
                    taxi_distances.append(self.d(passenger[LOC], taxi[LOC]))
                else:
                    # calculate the distance = taxi_to_nearest_passenger_destination + nearest_passenger_dest_to_current_one
                    passengers_distances = []
                    for taxi_passenger in node.state[PASSENGERS]:
                        if taxi_passenger[IN_TAXI] == taxi[NAME]:
                            passengers_distances.append(
                                self.d(taxi[LOC], taxi_passenger[DESTINATION]) +
                                self.d(passenger[LOC], taxi_passenger[DESTINATION]))
                    taxi_distances.append(min(passengers_distances))
            total_distances.append(min(taxi_distances) + self.d(passenger[LOC], passenger[DESTINATION]))

            if passenger[IN_TAXI] == 'null' and passenger[LOC] != passenger[DESTINATION]:
                total_distances[-1] += 2  # not in taxi but needs a taxi
            if passenger[IN_TAXI] != 'null':
                total_distances[-1] += 1  # in taxi but not in location

        return max(total_distances)

    def max_passenger_fuel(self, node):
        total_distances = []
        for passenger in node.state[PASSENGERS]:
            taxi_distances = []
            for taxi in node.state[TAXIS]:
                if taxi[CAP] > taxi[NUM_PASSENGERS] or taxi[NAME] == passenger[IN_TAXI]:  # default case when
                    taxi_distances.append((taxi[NAME], self.d(passenger[LOC], taxi[LOC])))
                    # fuel checks
                    if taxi[FUEL] < self.nearest_gas_station[taxi[LOC]] and taxi[FUEL] < taxi_distances[-1][1]:
                        taxi_distances[-1] = (taxi[NAME],
                                              np.inf)  # if taxi doesn't have enough fuel to refuel AND can't deliever the passenger before the fuel runs out, taxi_dist = inf
                    if taxi[FUEL] < taxi_distances[-1][1]:
                        taxi_distances[-1] = (taxi_distances[-1][0], taxi_distances[-1][1] + 1)
                else:
                    # calculate the distance = taxi_to_nearest_passenger_destination + nearest_passenger_dest_to_current_one
                    passengers_distances = []
                    for taxi_passenger in node.state[PASSENGERS]:
                        if taxi_passenger[IN_TAXI] == taxi[NAME]:
                            passengers_distances.append(
                                self.d(taxi[LOC], taxi_passenger[DESTINATION]) +
                                self.d(passenger[LOC], taxi_passenger[DESTINATION]) + 1)
                    taxi_distances.append((taxi[NAME], min(passengers_distances)))
                    # # fuel checks
                    # if taxi[FUEL] < self.nearest_gas_station[taxi[LOC]] and taxi[FUEL] < taxi_distances[-1][1]:
                    #     taxi_distances[-1] = (taxi[NAME],
                    #                           np.inf)  # if taxi doesn't have enough fuel to refuel AND can't deliever the passenger before the fuel runs out, taxi_dist = inf
                    if taxi[FUEL] < taxi_distances[-1][1]:
                        taxi_distances[-1] = (taxi_distances[-1][0], taxi_distances[-1][1] + 1)

            total_distances.append(
                min(taxi_distances, key=lambda x: x[1])[1] + self.d(passenger[LOC], passenger[DESTINATION]))

            # if passenger[IN_TAXI] == 'null' and passenger[LOC] != passenger[DESTINATION]:
            #     total_distances[-1] += 2  # not in taxi but needs a taxi
            # if passenger[IN_TAXI] != 'null':
            #     total_distances[-1] += 1  # in taxi but not in location

        return max(total_distances)

    def h_silver(self, node):
        """
        This is a *OUR* heuristic
        # """
        if len(node.state[PASSENGERS]) == 0:
            return 0

        dist_lst = np.zeros(len(node.state[PASSENGERS]))
        poss = 0

        for p in node.state[PASSENGERS]:
            if p[IN_TAXI] != 'null':
                carrying_taxi_name = p[IN_TAXI]
                carrying_taxi = None
                for taxi in node.state[TAXIS]:
                    if taxi[NAME] == carrying_taxi_name:
                        carrying_taxi = taxi
                assert carrying_taxi != None
                dist = self.d(carrying_taxi[LOC], p[DESTINATION])
                if dist < carrying_taxi[FUEL]:
                    min_gas_station_dist = self.nearest_gas_station[carrying_taxi[LOC]]
                    if carrying_taxi[FUEL] < min_gas_station_dist:
                        return np.inf
                    dist_lst[poss] = dist + 1
                    poss += 1
                else:
                    dist_lst[poss] = dist + 1 + 1
                    poss += 1
            else:
                dist_to_dest = self.d(p[LOC], p[DESTINATION])
                if dist_to_dest == np.inf:
                    return np.inf

                min_oa_dist = np.inf

                for t in node.state[TAXIS]:
                    if t[CAP] > t[NUM_PASSENGERS]:
                        p_t_dist = self.d(t[LOC], p[LOC])
                        min_gas_station_dist = self.nearest_gas_station[t[LOC]]
                        if min_gas_station_dist > t[FUEL]:
                            min_gas_station_dist = np.inf

                        if p_t_dist + dist_to_dest < t[FUEL] and p_t_dist + dist_to_dest < min_oa_dist:
                            min_oa_dist = p_t_dist + dist_to_dest
                        elif max(p_t_dist + dist_to_dest + 1, min_gas_station_dist + 1) < min_oa_dist:
                            min_oa_dist = max(p_t_dist + dist_to_dest + 1, min_gas_station_dist + 1)

                if min_oa_dist == np.inf:
                    for p2 in node.state[PASSENGERS]:

                        if p2[IN_TAXI] != 'null':
                            carrying_taxi_name = p2[IN_TAXI]
                            carrying_taxi = None
                            for taxi in node.state[TAXIS]:
                                if taxi[NAME] == carrying_taxi_name:
                                    carrying_taxi = taxi
                            assert carrying_taxi != None
                            if carrying_taxi[FUEL] != 0:
                                p2pd = self.d(p[LOC], p2[DESTINATION])
                                t2pd = self.d(p2[DESTINATION], carrying_taxi[LOC]) + 1
                                if carrying_taxi[FUEL] < p2pd + dist_to_dest + t2pd:
                                    p2pd += 1
                                if t2pd + p2pd + dist_to_dest < min_oa_dist:
                                    min_oa_dist = t2pd + p2pd + dist_to_dest

                dist_lst[poss] = min_oa_dist + 2
                poss += 1

        heuristic = max(dist_lst)  # (sum(dist_lst)/len(dist_lst) + max(dist_lst))/2 + self.h_1(node)

        return heuristic

    def h_1(self, node):
        """
        This is a simple heuristic
        """
        unpicked_passengers = 0
        undelivered_passengers = 0
        num_of_taxis = len(node.state[TAXIS])
        for passenger in node.state[PASSENGERS]:
            if passenger[LOC] != passenger[DESTINATION]:
                if passenger[IN_TAXI] != 'null':
                    undelivered_passengers += 1
                else:
                    unpicked_passengers += 1
        return (unpicked_passengers * 2 + undelivered_passengers) / num_of_taxis

    def MD(self, loc1, loc2):
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

    def h_2(self, node):
        """
        This is a slightly more sophisticated Manhattan heuristic
        """
        total_distance = 0
        for passenger in node.state[PASSENGERS]:
            total_distance += self.MD(passenger[DESTINATION], passenger[LOC])
        return total_distance

    """Feel free to add your own functions
    (-2, -2, None) means there was a timeout"""

    def bfs(self, grid, start, destination):
        queue = collections.deque([[start]])
        seen = set([start])
        width, height = self.map_size
        while queue:
            path = queue.popleft()
            x, y = path[-1]
            if grid[y][x] == destination:
                return path
            for x2, y2 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= x2 < width and 0 <= y2 < height and grid[y2][x2] != 'I' and (x2, y2) not in seen:
                    queue.append(path + [(x2, y2)])
                    seen.add((x2, y2))


def create_taxi_problem(game):
    return TaxiProblem(game)
