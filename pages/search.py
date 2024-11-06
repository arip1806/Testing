import matplotlib.pyplot as plt
from itertools import permutations
import random
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# User inputs for coordinates and city names
x_input = st.text_input("Enter x coordinates (comma-separated):", "0,3,6,7,15,10,16,5,8,1.5")
y_input = st.text_input("Enter y coordinates (comma-separated):", "1,2,1,4.5,-1,2.5,11,6,9,12")
cities_input = st.text_area("Enter city names (comma-separated):", "Gliwice, Cairo, Rome, Krakow, Paris, Alexandria, Berlin, Tokyo, Rio, Budapest")

# Convert user input to lists
x = list(map(float, x_input.split(',')))
y = list(map(float, y_input.split(',')))
cities_names = cities_input.split(',')

# Validate that input lengths are consistent
if len(x) != len(y) or len(x) != len(cities_names):
    st.error("The number of x coordinates, y coordinates, and city names must be the same.")
else:
    # User inputs for GA parameters
    n_population = st.number_input("Enter population size:", min_value=1, value=250, step=1)
    crossover_per = st.slider("Crossover percentage:", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
    mutation_per = st.slider("Mutation percentage:", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    n_generations = st.number_input("Number of generations:", min_value=1, value=200, step=1)

    # Pastel color palette
    colors = sns.color_palette("pastel", len(cities_names))

    # City icons
    city_icons = {
        "Gliwice": "♕",
        "Cairo": "♖",
        "Rome": "♗",
        "Krakow": "♘",
        "Paris": "♙",
        "Alexandria": "♔",
        "Berlin": "♚",
        "Tokyo": "♛",
        "Rio": "♜",
        "Budapest": "♝"
    }

    # Create dictionary for city coordinates
    city_coords = dict(zip(cities_names, zip(x, y)))

    # Plot city points and names
    fig, ax = plt.subplots()
    ax.grid(False)

    for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
        color = colors[i]
        icon = city_icons.get(city, "⬤")  # Default icon if city not in predefined icons
        ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
        ax.annotate(icon, (city_x, city_y), fontsize=40, ha='center', va='center', zorder=3)
        ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30),
                    textcoords='offset points')

        # Connect cities with opaque lines
        for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
            if i != j:
                ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)

    fig.set_size_inches(16, 12)
    st.pyplot(fig)

    # Code for running the TSP genetic algorithm and plotting results
    # You can include your existing GA and plotting logic here as before
    # ...
