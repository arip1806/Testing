import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
from itertools import permutations
import streamlit as st

def generate_tsp_solution(city_data, n_population=250, crossover_per=0.8, mutation_per=0.2, n_generations=200):
    """
    Generates the TSP solution using a genetic algorithm for given cities.
    
    Parameters:
    - city_data: Dictionary with city names as keys and (x, y) coordinates as values.
    - n_population: Population size for GA.
    - crossover_per: Crossover percentage.
    - mutation_per: Mutation percentage.
    - n_generations: Number of generations for GA.

    """
    cities_names = list(city_data.keys())
    city_coords = city_data
    colors = sns.color_palette("pastel", len(cities_names))

    # Visualize Cities
    fig, ax = plt.subplots()
    for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
        color = colors[i]
        ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
        ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30),
                    textcoords='offset points')
        for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
            if i != j:
                ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)

    fig.set_size_inches(16, 12)
    st.pyplot(fig)

    def initial_population(cities_list, n_population=250):
        population_perms = []
        possible_perms = list(permutations(cities_list))
        random_ids = random.sample(range(len(possible_perms)), n_population)
        for i in random_ids:
            population_perms.append(list(possible_perms[i]))
        return population_perms

    def dist_two_cities(city_1, city_2):
        city_1_coords = city_coords[city_1]
        city_2_coords = city_coords[city_2]
        return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords))**2))

    def total_dist_individual(individual):
        total_dist = 0
        for i in range(len(individual)):
            if i == len(individual) - 1:
                total_dist += dist_two_cities(individual[i], individual[0])
            else:
                total_dist += dist_two_cities(individual[i], individual[i+1])
        return total_dist

    def fitness_prob(population):
        total_dist_all_individuals = [total_dist_individual(ind) for ind in population]
        max_population_cost = max(total_dist_all_individuals)
        population_fitness = max_population_cost - np.array(total_dist_all_individuals)
        population_fitness_sum = sum(population_fitness)
        return population_fitness / population_fitness_sum

    def roulette_wheel(population, fitness_probs):
        population_fitness_probs_cumsum = fitness_probs.cumsum()
        selected_individual_index = len(population_fitness_probs_cumsum[population_fitness_probs_cumsum < np.random.uniform(0, 1, 1)])
        return population[selected_individual_index]

    def crossover(parent_1, parent_2):
        cut = round(random.uniform(1, len(cities_names) - 1))
        offspring_1 = parent_1[:cut] + [city for city in parent_2 if city not in parent_1[:cut]]
        offspring_2 = parent_2[:cut] + [city for city in parent_1 if city not in parent_2[:cut]]
        return offspring_1, offspring_2

    def mutation(offspring):
        idx1, idx2 = random.sample(range(len(offspring)), 2)
        offspring[idx1], offspring[idx2] = offspring[idx2], offspring[idx1]
        return offspring

    def run_ga():
        population = initial_population(cities_names, n_population)
        best_mixed_offspring = population
        for _ in range(n_generations):
            fitness_probs = fitness_prob(best_mixed_offspring)
            parents_list = [roulette_wheel(best_mixed_offspring, fitness_probs) for _ in range(int(crossover_per * n_population))]
            offspring_list = []
            for i in range(0, len(parents_list), 2):
                offspring_1, offspring_2 = crossover(parents_list[i], parents_list[i + 1])
                if random.random() < mutation_per:
                    offspring_1 = mutation(offspring_1)
                if random.random() < mutation_per:
                    offspring_2 = mutation(offspring_2)
                offspring_list.extend([offspring_1, offspring_2])
            best_mixed_offspring = sorted(offspring_list + parents_list, key=total_dist_individual)[:n_population]
        return min(best_mixed_offspring, key=total_dist_individual)

    # Run GA and display result
    shortest_path = run_ga()
    min_distance = total_dist_individual(shortest_path)

    x_shortest, y_shortest = zip(*[city_coords[city] for city in shortest_path + [shortest_path[0]]])

    fig, ax = plt.subplots()
    ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
    plt.legend()
    for i, (city_x, city_y) in enumerate(zip(x_shortest, y_shortest)):
        ax.annotate(f"{i+1}- {shortest_path[i] if i < len(shortest_path) else shortest_path[0]}", (city_x, city_y), fontsize=12)
    plt.title("TSP Best Route Using GA")
    plt.suptitle(f"Total Distance: {round(min_distance, 3)}\n{n_generations} Generations, {n_population} Population, {crossover_per} Crossover, {mutation_per} Mutation")
    fig.set_size_inches(16, 12)
    st.pyplot(fig)

# Sample usage
city_data = {
    "Gliwice": (0, 3), "Cairo": (3, 6), "Rome": (6, 7), "Krakow": (7, 15), "Paris": (15, 10),
    "Alexandria": (10, 16), "Berlin": (16, 5), "Tokyo": (5, 8), "Rio": (8, 1.5), "Budapest": (1.5, 12)
}
generate_tsp_solution(city_data)
