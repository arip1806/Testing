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
        new_x = st.number_input(f"New x-coordinate for {city}:", value=coords[0])
        new_y = st.number_input(f"New y-coordinate for {city}:", value=coords[1])
        city_coords_dict[city] = (new_x, new_y)

# ... rest of the functions (distance_between_cities, total_distance, etc.)

def main():
    st.title("Traveling Salesman Problem Solver")
    cities, coords = get_user_input()
    city_coords_dict = dict(zip(cities, coords))

    if st.button("Update City Coordinates"):
        update_city_data(city_coords_dict)

    if st.button("Run Genetic Algorithm"):
        best_route, best_distance = run_ga(cities, city_coords_dict, n_population=250, n_generations=100)

        # ... (visualization code)

if __name__ == "__main__":
    main()
