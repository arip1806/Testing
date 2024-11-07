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

# ... rest of the code (initial_population, dist_two_cities, etc.)

def main():
    st.title("TSP Solver with User Input")
    cities, coords = get_user_input()
    city_coords_dict = dict(zip(cities, coords))

    # ... rest of the code (GA implementation, visualization)

    if st.button("Run Genetic Algorithm"):
        best_route, best_distance = run_ga(cities, n_population=250, n_generations=200)

        # ... (visualization code)
        
if __name__ == "__main__":
    main()
