import streamlit as st
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from random import shuffle
import numpy as np

def get_user_input():
    num_cities = st.number_input("Enter the number of cities:", min_value=2, value=10)
    cities = []
    coords = []
    for i in range(num_cities):
        city_name = st.text_input(f"Enter the name of city {i+1}:")
        x_coord = st.number_input(f"Enter the x-coordinate for {city_name}:")
        y_coord = st.number_input(f"Enter the y-coordinate for {city_name}:")
        cities.append(city_name)
        coords.append((x_coord, y_coord))
    return cities, coords

def update_city_data(city_coords_dict):
    for city, coords in city_coords_dict.items():
        st.write(f"**City: {city}**")
        new_name = st.text_input(f"New name for {city}:", value=city)
        new_x = st.number_input(f"New x-coordinate for {city}:", value=coords[0])
        new_y = st.number_input(f"New y-coordinate for {city}:", value=coords[1])
        city_coords_dict[new_name] = (new_x, new_y)
        if new_name != city:
            del city_coords_dict[city]

def distance_between_cities(city1, city2, city_coords_dict):
    x1, y1 = city_coords_dict[city1]
    x2, y2 = city_coords_dict[city2]
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def total_distance(route, city_coords_dict):
    total_distance = 0
    num_cities = len(route)
    for i in range(num_cities):
        j = (i + 1) % num_cities
        city1, city2 = route[i], route[j]
        total_distance += distance_between_cities(city1, city2, city_coords_dict)
    return total_distance

def initial_population(cities, population_size):
    population = []
    for _ in range(population_size):
        random.shuffle(cities)
        population.append(list(cities))
    return population

def selection(population, fitness_scores):
    selected_indices = np.random.choice(len(population), size=len(population), p=fitness_scores)
    return [population[i] for i in selected_indices]

def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutation(individual, mutation_rate):
    if np.random.rand() < mutation_rate:
        index1, index2 = np.random.choice(len(individual), 2, replace=False)
        individual[index1], individual[index2] = individual[index2], individual[index1]
    return individual

def genetic_algorithm(cities, city_coords_dict, population_size=100, generations=100, crossover_rate=0.8, mutation_rate=0.05):
    population = initial_population(cities, population_size)
    best_distance = float('inf')
    best_route = None

    for generation in range(generations):
        fitness_scores = [1 / total_distance(individual, city_coords_dict) for individual in population]
        fitness_scores = fitness_scores / np.sum(fitness_scores)

        new_population = []
        for _ in range(population_size):
            parent1, parent2 = selection(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = new_population

        for individual in population:
            distance = total_distance(individual, city_coords_dict)
            if distance < best_distance:
                best_distance = distance
                best_route = individual

    return best_route, best_distance

def visualize_route(route, city_coords_dict):
    x_coords = [city_coords_dict[city][0] for city in route]
    y_coords = [city_coords_dict[city][1] for city in route]
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])

    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='blue')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Optimal Route')
    st.pyplot(plt)

def main():
    st.title("Traveling Salesman Problem Solver")
    cities, coords = get_user_input()
    city_coords_dict = dict(zip(cities, coords))

    if st.button("Update City Coordinates"):
        update_city_data(city_coords_dict)

    if st.button("Run Genetic Algorithm"):
        best_route, best_distance = run_ga(cities, city_coords_dict, n_population=250, n_generations=100)

        st.write("Best Route:", best_route)
        st.write("Best Distance:", best_distance)
        visualize_route(best_route, city_coords_dict)

if __name__ == "__main__":
    main()
