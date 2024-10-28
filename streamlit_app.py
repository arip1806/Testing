import streamlit as st
import random

def initialize_pop(TARGET):
    # ... (rest of the function)

# ... (rest of the functions)

def main(POP_SIZE, MUT_RATE, TARGET, GENES):
    # ... (rest of the function)

# Streamlit app
st.set_page_config(
    page_title="Genetic Algorithm"
)
st.header("Genetic Algorithm", divider="gray")

# User input fields
target_input = st.text_input("Enter the target string:")
mutation_rate_input = st.number_input("Enter the mutation rate (0.0 - 1.0):", min_value=0.0, max_value=1.0, step=0.1)

# Button to trigger the calculation
calculate_button = st.button("Calculate")

if calculate_button:
    TARGET = target_input.upper()
    MUT_RATE = mutation_rate_input

    # Rest of the code, including the main function call
    result = main(POP_SIZE, MUT_RATE, TARGET, GENES)
