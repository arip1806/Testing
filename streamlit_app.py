import streamlit as st
import random

POP_SIZE = 500
GENES = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ '

def initialize_population(target):
    population = []
    target_len = len(target)
    for _ in range(POP_SIZE):
        chromosome = [random.choice(GENES) for _ in range(target_len)]
        population.append(chromosome)
    return population

def calculate_fitness(target, chromosome):
    fitness = 0
    for i in range(len(target)):
        if target[i] != chromosome[i]:
            fitness += 1
    return fitness

def selection(population, fitness_scores):
    selected_parents = []
    for _ in range(POP_SIZE // 2):
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        if fitness_scores[population.index(parent1)] < fitness_scores[population.index(parent2)]:
            selected_parents.append(parent1)
        else:
            selected_parents.append(parent2)
    return selected_parents

def crossover(parents, crossover_rate):
    offspring = []
    for i in range(0, POP_SIZE, 2):
        parent1, parent2 = parents[i], parents[i+1]
        child1, child2 = parent1.copy(), parent2.copy()
        if random.random() < crossover_rate:
            crossover_point = random.randint(1, len(parent1) - 1)
            child1[crossover_point:] = parent2[crossover_point:]
            child2[crossover_point:] = parent1[crossover_point:]
        offspring.append(child1)
        offspring.append(child2)
    return offspring

def mutation(offspring, mutation_rate):
    mutated_offspring = []
    for child in offspring:
        mutated_child = child.copy()
        for i in range(len(child)):
            if random.random() < mutation_rate:
                mutated_child[i] = random.choice(GENES)
        mutated_offspring.append(mutated_child)
    return mutated_offspring

def genetic_algorithm(target, population_size, mutation_rate, crossover_rate):
    population = initialize_population(target)
    generation = 1

    while True:
        fitness_scores = [calculate_fitness(target, chromosome) for chromosome in population]
        best_fitness = min(fitness_scores)
        best_chromosome = population[fitness_scores.index(best_fitness)]

        if best_fitness == 0:
            print(f"Target found in generation {generation}: {''.join(best_chromosome)}")
            break

        selected_parents = selection(population, fitness_scores)
        offspring = crossover(selected_parents, crossover_rate)
        mutated_offspring = mutation(offspring, mutation_rate)
        population = mutated_offspring

        generation += 1

# Streamlit app
st.set_page_config(
    page_title="Genetic Algorithm"
)
st.header("Genetic Algorithm", divider="gray")

target_input = st.text_input("Enter the target string:")
mutation_rate_input = st.number_input("Enter the mutation rate (0.0 - 1.0):", min_value=0.0, max_value=1.0, step=0.1)


calculate_button = st.button("Calculate")

if calculate_button:
    target = target_input.upper()
    mutation_rate = mutation_rate_input

    genetic_algorithm(target, POP_SIZE, mutation_rate, crossover_rate)
