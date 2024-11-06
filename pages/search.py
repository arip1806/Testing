import matplotlib.pyplot as plt
from itertools import permutations, combinations
from random import shuffle
import numpy as np
import streamlit as st

# Get user input for city names and coordinates
num_cities = int(input("Enter the number of cities: "))
city_names = []
city_coords = []

for i in range(num_cities):
    city_name = input(f"Enter the name of city {i+1}: ")
    x_coord = float(input(f"Enter the x-coordinate for {city_name}: "))
    y_coord = float(input(f"Enter the y-coordinate for {city_name}: "))
    city_names.append(city_name)
    city_coords.append((x_coord, y_coord))

# Create a dictionary to store city names and coordinates
city_coords_dict = dict(zip(city_names, city_coords))


def initial_population(cities_list, n_population=250):
    """
    Generating initial population of cities randomly selected from the all possible permutations
    of the given cities.
    Input:
    1- Cities list
    2- Number of population
    Output:
    Generated lists of cities
    """

    population_perms = []
    possible_perms = list(permutations(cities_list))
    random_ids = random.sample(range(0, len(possible_perms)), n_population)

    for i in random_ids:
        population_perms.append(list(possible_perms[i]))

    return population_perms


def dist_two_cities(city_1, city_2):
    """
    Calculates the distance between two cities based on their coordinates.
    """

    city_1_coords = city_coords_dict[city_1]
    city_2_coords = city_coords_dict[city_2]
    return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords))**2))


def total_dist_individual(individual):
    """
    Calculates the total distance of a travel route represented by an individual (city permutation).
    """

    total_dist = 0
    for i in range(0, len(individual)):
        if (i == len(individual) - 1):
            total_dist += dist_two_cities(individual[i], individual[0])
        else:
            total_dist += dist_two_cities(individual[i], individual[i + 1])
    return total_dist


def fitness_prob(population):
    """
    Calculating the fitness probability
    Input:
    1- Population
    Output:
    Population fitness probability
    """

    total_dist_all_individuals = []
    for i in range(0, len(population)):
        total_dist_all_individuals.append(total_dist_individual(population[i]))

    max_population_cost = max(total_dist_all_individuals)
    population_fitness = max_population_cost - total_dist_all_individuals
    population_fitness_sum = sum(population_fitness)
    population_fitness_probs = population_fitness / population_fitness_sum
    return population_fitness_probs


def roulette_wheel(population, fitness_probs):
    """
    Implement selection strategy based on roulette wheel proportionate selection.
    Input:
    1- population
    2- fitness probabilities
    Output:
    selected individual
    """

    population_fitness_probs_cumsum = fitness_probs.cumsum()
    bool_prob_array = population_fitness_probs_cumsum < np.random.uniform(0, 1, 1)
    selected_individual_index = len(bool_prob_array[bool_prob_array == True]) - 1
    return population[selected_individual_index]


def crossover(parent_1, parent_2):
    """
    Implement mating strategy using simple crossover between 2 parents
    Input:
    1- parent 1
    2- parent 2
    Output:
    1- offspring 1
    2- offspring 2
    """

    n_cities_cut = len(city_names) - 1
    cut = round(random.uniform(1, n_cities_cut))
    offspring_1 = []
    offspring_2
