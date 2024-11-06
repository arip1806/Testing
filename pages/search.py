import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random
from itertools import permutations
import seaborn as sns

st.title("Traveling Salesman Problem (TSP) Using Genetic Algorithm")

# User Inputs for Cities and Coordinates
st.subheader("City Coordinates Input")
st.write("Enter up to 11 cities with their coordinates (x, y) in range 1-10.")

city_names = []
x_coords = []
y_coords = []

# Input fields for each city
for i in range(1, 12):  # Updated to allow 11 cities
    st.write(f"City {i}")
    city_name = st.text_input(f"City {i} Name", key=f"city_name_{i}")
    x_coord = st.number_input(f"x-coordinate (City {i})", min_value=1.0, max_value=10.0, step=0.1, key=f"x_coord_{i}")
    y_coord = st.number_input(f"y-coordinate (City {i})", min_value=1.0, max_value=10.0, step=0.1, key=f"y_coord_{i}")

    if city_name:
        city_names.append(city_name)
        x_coords.append(x_coord)
        y_coords.append(y_coord)

# Submit button
if st.button("Run Genetic Algorithm"):
    if len(city_names) < 2:
        st.error("Please enter at least two cities to run the genetic algorithm.")
    else:
        # Define the city coordinates dictionary
        city_coords = dict(zip(city_names, zip(x_coords, y_coords)))
        
        # Parameters for the Genetic Algorithm
        n_population = 250
        crossover_per = 0.8
        mutation_per = 0.2
        n_generations = 200

        # Pastel Palette for plotting
        colors = sns.color_palette("pastel", len(city_names))

        # Genetic Algorithm Code
        def initial_population(cities_list, n_population=250):
            population_perms = []
            possible_perms = list(permutations(cities_list))
            random_ids = random.sample(range(0, len(possible_perms)), n_population)
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
            population_fitness_probs = population_fitness / sum(population_fitness)
            return population_fitness_probs

        def roulette_wheel(population, fitness_probs):
            population_fitness_probs_cumsum = fitness_probs.cumsum()
            selected_individual_index = np.searchsorted(population_fitness_probs_cumsum, np.random.rand())
            return population[selected_individual_index]

        def crossover(parent_1, parent_2):
            cut = random.randint(1, len(parent_1) - 1)
            offspring_1 = parent_1[:cut] + [city for city in parent_2 if city not in parent_1[:cut]]
            offspring_2 = parent_2[:cut] + [city for city in parent_1 if city not in parent_2[:cut]]
            return offspring_1, offspring_2

        def mutation(offspring):
            index_1, index_2 = random.sample(range(len(offspring)), 2)
            offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
            return offspring

        def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
            population = initial_population(cities_names, n_population)
            best_mixed_offspring = population

            for _ in range(n_generations):
                fitness_probs = fitness_prob(best_mixed_offspring)
                parents_list = [roulette_wheel(best_mixed_offspring, fitness_probs) for _ in range(int(crossover_per * n_population))]
                offspring_list = []

                for i in range(0, len(parents_list), 2):
                    offspring_1, offspring_2 = crossover(parents_list[i], parents_list[i+1])
                    if random.random() < mutation_per:
                        offspring_1 = mutation(offspring_1)
                    if random.random() < mutation_per:
                        offspring_2 = mutation(offspring_2)
                    offspring_list.extend([offspring_1, offspring_2])

                mixed_offspring = parents_list + offspring_list
                best_mixed_offspring = sorted(mixed_offspring, key=total_dist_individual)[:n_population]

            return best_mixed_offspring

        best_mixed_offspring = run_ga(city_names, n_population, n_generations, crossover_per, mutation_per)

        # Calculate best route and distance
        shortest_path = min(best_mixed_offspring, key=total_dist_individual)
        minimum_distance = total_dist_individual(shortest_path)
        
        # Plotting the shortest path
        x_shortest, y_shortest = zip(*[city_coords[city] for city in shortest_path + [shortest_path[0]]])
        
        fig, ax = plt.subplots()
        ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
        plt.legend()
        for i, (city_x, city_y) in enumerate(zip(x_shortest, y_shortest)):
            ax.annotate(f"{i+1}- {shortest_path[i % len(shortest_path)]}", (city_x, city_y), fontsize=12)
        plt.title("TSP Best Route Using GA\nTotal Distance: {:.3f}".format(minimum_distance))
        fig.set_size_inches(16, 12)
        st.pyplot(fig)

        # Display results
        st.write("Shortest Path:", shortest_path)
        st.write("Minimum Distance:", round(minimum_distance, 3))
