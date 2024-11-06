import streamlit as st
import matplotlib.pyplot as plt
from itertools import permutations, combinations
from random import shuffle
import numpy as np

def get_user_input():
    num_cities = st.number_input("Enter the number of cities:", min_value=2, value=5)
    cities = []
    coords = []
    for i in range(num_cities):
        city_name = st.text_input(f"Enter the name of city {i+1}:")
        x_coord = st.number_input(f"Enter the x-coordinate for {city_name}:")
        y_coord = st.number_input(f"Enter the y-coordinate for {city_name}:")
        cities.append(city_name)
        coords.append((x_coord, y_coord))
    return cities, coords

def run_ga(cities_names, n_population=250, n_generations=200, crossover_per=0.8, mutation_per=0.2):
    # ... (rest of the GA implementation)

def visualize_cities(cities, coords):
    fig, ax = plt.subplots()
    for i, (city, (x, y)) in enumerate(zip(cities, coords)):
        ax.scatter(x, y, color='blue', marker='o')
        ax.annotate(city, (x, y), fontsize=10)
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.title("Initial City Map")
    st.pyplot(fig)

def main():
    st.title("TSP Solver with User Input")
    cities, coords = get_user_input()

    if st.button("Run Genetic Algorithm"):
        best_route = run_ga(cities, n_population=250, n_generations=200, crossover_per=0.8, mutation_per=0.2)

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
        st.pyplot(fig)

if __name__ == "__main__":
    main()
