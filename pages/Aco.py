import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Title and Description
st.title("Job Shop Scheduling with ACO")
st.markdown("""
This application demonstrates the Ant Colony Optimization (ACO) algorithm applied to the job shop scheduling problem (JSSP).
""")

# Data Upload Section
uploaded_file = st.file_uploader("Upload JSSP Data (CSV)", type="csv")

# Input Parameters (Enabled only if data is uploaded)
num_jobs = None
num_machines = None
iterations = None
alpha = None
beta = None
processing_times = None

if uploaded_file is not None:
    # Read data from uploaded CSV file
    df = pd.read_csv(uploaded_file)
    num_jobs = df.shape[0]
    num_machines = df.shape[1]
    processing_times = df.to_numpy()

    # Input parameters are enabled only if data is uploaded successfully
    iterations = st.slider("Number of Iterations", min_value=10, max_value=500, value=100)
    alpha = st.slider("Pheromone Importance (Alpha)", min_value=0.1, max_value=5.0, value=1.0)
    beta = st.slider("Heuristic Importance (Beta)", min_value=0.1, max_value=5.0, value=2.0)

# Solve JSSP with ACO (using the imported data)
def solve_jssp_aco(num_jobs, num_machines, processing_times, iterations, alpha, beta):
    """
    Solves the JSSP using ACO with the provided data.

    Args:
        num_jobs: Number of jobs.
        num_machines: Number of machines.
        processing_times: A 3D array representing processing times for each job on each machine.
            Shape: (jobs, machines, operations)
        iterations: Number of iterations for the ACO algorithm.
        alpha: Importance of pheromone trails.
        beta: Importance of heuristic information.

    Returns:
        A list containing the best makespan found in each iteration.
    """
    # Initialize pheromone matrix
    pheromone_matrix = np.ones((num_jobs, num_machines))

    # ACO parameters
    num_ants = 10  # Adjust as needed
    rho = 0.1

    best_makespan_history = []

    for it in range(iterations):
        # Generate ant solutions
        solutions = []
        for _ in range(num_ants):
            solution = []
            for job in range(num_jobs):
                available_machines = list(range(num_machines))
                job_sequence = []
                for operation in range(num_machines):
                    probabilities = []
                    for machine in available_machines:
                        pheromone = pheromone_matrix[job][machine] ** alpha
                        heuristic = 1 / (processing_times[job][machine][operation] + 1e-6)
                        probabilities.append(pheromone * heuristic ** beta)
                    probabilities = np.array(probabilities) / np.sum(probabilities)
                    selected_machine = np.random.choice(available_machines, p=probabilities)
                    job_sequence.append(selected_machine)
                    available_machines.remove(selected_machine)
                solution.append(job_sequence)
            solutions.append(solution)

        # Calculate makespans
        makespans = [calculate_makespan(solution, processing_times) for solution in solutions]
        best_index = np.argmin(makespans)
        best_makespan = makespans[best_index]
        best_solution = solutions[best_index]

        # Update pheromone trails
        pheromone_matrix *= (1 - rho)
        for job in range(num_jobs):
            for machine in range(num_machines):
                if machine == best_solution[job][machine]:
                    pheromone_matrix[job][machine] += 1.0

        best_makespan_history.append(best_makespan)

    return best_makespan_history

def calculate_makespan(solution, processing_times):
    """
    Calculates the makespan of a given schedule.
    """
    machine_loads = [0] * processing_times.shape[1]
    for job in range(processing_times.shape[0]):
        for machine in range(processing_times.shape[1]):
            operation_index = solution[job][machine]
            machine_loads[machine] += processing_times[job][machine][operation_index]
    return max(machine_loads)

# Optimization (run ACO only if data is uploaded)
if uploaded_file is not None:
  if st.button("Run ACO"):
    best_makespan_history = solve_jssp_aco(num_jobs, num_machines, processing_times, iterations, alpha, beta)

    # Plot Convergence
    st.subheader("Convergence Plot")
    fig, ax = plt.subplots()
    ax.plot(best_makespan_history, label="Best Makespan")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Makespan")
    ax.legend()
    st.pyplot(fig)

    # Display Results Table
    st.subheader("Results Table")
    results_df = pd.DataFrame({'Iteration': range(1, iterations + 1), 'Best Makespan': best_makespan_history})
    st.table(results_df)

