import numpy as np
import pandas as pd
import data_generator as dag
import discretizer as dis
import utility as utl
import discretization_generator as dig
import estimator as st

"""
This script provides a complete example of estimating a dynamic discrete choice model
for an infinite-horizon bus engine replacement setting (Rust-type problem).

Steps:
1. Generate a synthetic discretization and transition model for latent variables.
2. Generate training, validation, and test data using EngineReplacementDataGenerator.
3. Estimate the discretization using the data-driven method.
4. Estimate structural parameters via a nested fixed point algorithm (Rust-style).
"""

# ------------------------------
# Step 1: Set simulation parameters
# ------------------------------
dim_q = 5
dim_active_q = 2
max_q = 10
dim_pi = 4
replacement_cost = -np.arange(4, 8)
mileage_coefficient = -0.2
discounting_factor = 0.9
max_mileage = 30
buses = 500
periods = 100

# ------------------------------
# Step 2: Generate discretization and transition matrix
# ------------------------------
def get_discretization():
    par_gen = dig.RandomDiscretizationGenerator(dim_q, dim_active_q, dim_pi, max_q)
    discretization = par_gen.generate_random_discretization(balance=2)
    discretization['f_dc'] = replacement_cost
    discretization['f_tr'] = np.arange(dim_pi)
    q_transition = utl.generate_pi_transition(dim_pi, 3, 2)
    return discretization, q_transition

# ------------------------------
# Step 3: Generate training and test datasets
# ------------------------------
def get_data(discretization, q_transition):
    data_gen = dag.EngineReplacementDataGenerator(
        max_mileage=max_mileage,
        mileage_coefficient=mileage_coefficient,
        discretization=discretization,
        q_transition=q_transition,
        max_q=max_q,
        dim_q=dim_q,
        discounting_factor=discounting_factor
    )
    train_df = data_gen.generate(buses=buses, periods=periods)
    data = utl.get_partitioning_variables(train_df)

    valid_df = data_gen.generate(buses=buses, periods=periods)
    val_data = utl.get_partitioning_variables(valid_df)

    test_df = data_gen.generate(buses=buses, periods=periods)
    test_data = utl.get_partitioning_variables(test_df)

    return train_df, data, valid_df, val_data, test_df, test_data

# ------------------------------
# Step 4: Estimate discretization
# ------------------------------
def estimate_discretization(data, val_data, test_data):
    discretizer = dis.DataDriveDiscretizer(delta=0.01, max_pi=10, finite_horizon=False)
    _, report = discretizer.discretize(data, val_data)
    optimal_parts = report.iloc[report.test_score.argmax()].part.astype(int)
    print(f'The estimated optimal number of partitions is {optimal_parts}')

    discretizer = dis.DataDriveDiscretizer(max_pi=optimal_parts)
    discretization_est, _ = discretizer.discretize(test_data, None)
    return discretization_est, report

# ------------------------------
# Step 5: Estimate model
# ------------------------------
def estimate_model(discretization_est, test_data):
    estimator = st.BusEngineNFXP()
    pi = utl.q_to_pi_states(discretization_est, test_data['Q'], dim_q)
    bestll, f, alpha = estimator.estimate_theta(
        ids=test_data['ids'],
        periods=test_data['periods'],
        X=test_data['X'].flatten().tolist(),
        pi=pi,
        y=test_data['Y'],
        discounting_factor=discounting_factor
    )
    return bestll, f, alpha

# ------------------------------
# Step 6: Main simulation
# ------------------------------
def simulation():
    print("Generating discretization...")
    discretization, q_transition = get_discretization()

    print("Generating data...")
    train_df, data, valid_df, val_data, test_df, test_data = get_data(discretization, q_transition)

    print("Estimating discretization...")
    discretization_est, report = estimate_discretization(data, val_data, test_data)

    print("Estimating model parameters...")
    bestll, f, alpha = estimate_model(discretization_est, test_data)

    estimated_replacement = np.abs(f)
    estimated_replacement.sort()

    print(f"Estimated maintenance cost coefficient: {alpha[0]} (True: {mileage_coefficient})")
    print(f"Estimated replacement costs: {np.round(estimated_replacement, 3)}")
    print(f"True replacement costs:      {np.round(np.abs(replacement_cost), 3)}")
    return discretization, discretization_est, report, bestll, f, alpha

# Run the simulation
np.random.seed(0)
simulation()
