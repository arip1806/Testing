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

# ... (rest of the code, including function definitions and GA implementation)

# Visualize the initial city map
fig, ax = plt.subplots()

for i, (city, (city_x, city_y)) in enumerate(city_coords_dict.items()):
    ax.scatter(city_x, city_y, color='blue', marker='o')
    ax.annotate(city, (city_x, city_y), fontsize=10)

plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.title("Initial City Map")
plt.show()

# Run the genetic algorithm
best_route = run_ga(city_names, n_population, n_generations, crossover_per, mutation_per)

# Visualize the best route
fig, ax = plt.subplots()

for i in range(len(best_route)):
    city1, city2 = best_route[i], best_route[(i + 1) % len(best_route)]
    x1, y1 = city_coords_dict[city1]
    x2, y2 = city_coords_dict[city2]
    plt.plot([x1, x2], [y1, y2], 'b-')

for city, coords in city_coords_dict.items():
    x, y = coords
    plt.scatter(x, y, color='red', marker='o')
    plt.annotate(city, (x, y))

plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.title("Best Route Found by Genetic Algorithm")
plt.show()
