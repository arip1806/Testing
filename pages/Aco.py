import random
import numpy as np
import streamlit as st
import pandas as pd

class JobShopSchedulingACO:

    def __init__(self, jobs, machines, processing_times, num_ants=10, max_iter=100, alpha=1.0, beta=2.0, rho=0.1):
        """
        Initializes the ACO algorithm for job shop scheduling.

        Args:
            jobs: Number of jobs.
            machines: Number of machines.
            processing_times: A 3D array representing processing times for each job on each machine.
                Shape: (jobs, machines, operations)
            num_ants: Number of ants in the colony.
            max_iter: Maximum number of iterations.
            alpha: Importance of pheromone trails.
            beta: Importance of heuristic information.
            rho: Evaporation rate of pheromone trails.
        """
        self.jobs = jobs
        self.machines = machines
        self.processing_times = processing_times
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.pheromone_matrix = np.ones((jobs, machines))  # Initialize pheromone matrix

    def calculate_makespan(self, solution):
        """
        Calculates the makespan of a given schedule.

        Args:
            solution: A 2D array representing the schedule, where 
                     solution[job_index][machine_index] = operation_index

        Returns:
            The makespan of the schedule.
        """
        machine_loads = [0] * self.machines
        for job in range(self.jobs):
            for machine in range(self.machines):
                operation_index = solution[job][machine]
                machine_loads[machine] += self.processing_times[job][machine][operation_index]
        return max(machine_loads)

    def construct_solution(self):
        """
        Constructs a solution (schedule) for one ant.
        """
        solution = []
        for job in range(self.jobs):
            available_machines = list(range(self.machines))
            job_sequence = []
            for operation in range(self.machines):
                probabilities = []
                for machine in available_machines:
                    pheromone = self.pheromone_matrix[job][machine] ** self.alpha
                    heuristic = 1 / (self.processing_times[job][machine][operation] + 1e-6)  # Avoid division by zero
                    probabilities.append(pheromone * heuristic ** self.beta)
                probabilities = np.array(probabilities) / np.sum(probabilities)
                selected_machine = np.random.choice(available_machines, p=probabilities)
                job_sequence.append(selected_machine)
                available_machines.remove(selected_machine)
            solution.append(job_sequence)
        return solution

    def update_pheromone_trails(self, best_solution):
        """
        Updates the pheromone trails based on the best solution.
        """
        for job in range(self.jobs):
            for machine in range(self.machines):
                self.pheromone_matrix[job][machine] *= (1 - self.rho)  # Evaporate pheromone
                if machine == best_solution[job][machine]:
                    self.pheromone_matrix[job][machine] += 1.0  # Deposit pheromone on the best path

    def run(self):
        """
        Runs the ACO algorithm for job shop scheduling.
        """
        best_makespan = float('inf')
        best_solution = None

        for iteration in range(self.max_iter):
            solutions = [self.construct_solution() for _ in range(self.num_ants)]
            makespans = [self.calculate_makespan(solution) for solution in solutions]
            best_index = np.argmin(makespans)
            best_solution_current = solutions[best_index]
            best_makespan_current = makespans[best_index]

            if best_makespan_current < best_makespan:
                best_makespan = best_makespan_current
                best_solution = best_solution_current

            self.update_pheromone_trails(best_solution)

            print(f"Iteration: {iteration+1}, Best Makespan: {best_makespan}")

        return best_solution, best_makespan

# Example usage (replace with your actual job shop data)
jobs = 3
machines = 3
processing_times = np.random.randint(1, 10, size=(jobs, machines, machines))  # Example processing times

aco = JobShopSchedulingACO(jobs, machines, processing_times)
best_solution, best_makespan = aco.run()

st.write("Best Solution:")
st.write(best_solution)
st.write("Best Makespan:", best_makespan)
