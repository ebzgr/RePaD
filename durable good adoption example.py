import numpy as np
import data_generator as dag
import discretizer as dis
import utility as utl
import discretization_generator as dig
import estimator as st
import pandas as pd

"""
This script provides a complete example for estimating a dynamic discrete choice model 
for a finite-horizon durable good adoption setting.

Steps:
1. Generate a synthetic discretization and transition model for high-dimensional state variables.
2. Generate training, validation, and test data using the DurablesAdoptionDataGenerator.
3. Use the data-driven discretizer to recover the latent state partitions.
4. Estimate the model parameters (utility and price sensitivity) using the NFXP algorithm.
"""

# ------------------------------
# Step 1: Set simulation parameters
# ------------------------------
dim_q = 10
dim_active_q = 2
max_q = 10
dim_pi = 4
buy_util = np.linspace(start=19.2, stop=20, num=dim_pi)  # True purchase utility
x_transit = np.arange(dim_pi-1,-1,-1)                    # Price transitions per state

discounting_factor = 0.90
price_coeff = -0.5                                       # True price sensitivity
price_start = 400
price_end = 1
periods = 30
consumers = 5000

# ------------------------------
# Step 2: Define a discretization 
# ------------------------------
partitions = pd.DataFrame(columns = ['state','q_0_min','q_0_max','q_1_min','q_1_max'])
partitions.loc[len(partitions)] = [0,0,max_q/2,0,max_q/2]
partitions.loc[len(partitions)] = [1,0,max_q/2,max_q/2,max_q]
partitions.loc[len(partitions)] = [2,max_q/2,max_q,max_q/2,max_q]
partitions.loc[len(partitions)] = [3,max_q/2,max_q,0,max_q/2]
for i in range(2,dim_q):
    partitions['q_{}_min'.format(i)] = 0
    partitions['q_{}_max'.format(i)] = max_q
partitions['f_dc'] = buy_util
partitions['f_tr'] = x_transit

# ------------------------------
# Step 3: Helper functions
# ------------------------------

def get_discretization():
    """Randomly generate a discretization and transition model"""
    par_gen = dig.RandomDiscretizationGenerator(dim_q, dim_active_q, dim_pi, max_q)
    discretization = par_gen.generate_random_discretization(balance = 2)
    discretization['f_dc'] = buy_util
    discretization['f_tr'] = x_transit
    q_transition = utl.generate_pi_transition(len(discretization), 3, 2)
    return discretization, q_transition

def get_data(discretization, q_transition):
    """Generate train, validation, and test datasets"""
    data_gen = dag.DurablesAdoptionDataGenerator(price_start, price_end, price_coeff, discretization, q_transition,
                                                 periods, max_q, dim_q, discounting_factor)
    train_df = data_gen.generate(consumers, periods)
    data = utl.get_partitioning_variables(train_df)

    valid_df = data_gen.generate(consumers, periods)
    val_data = utl.get_partitioning_variables(valid_df)

    test_df = data_gen.generate(consumers, periods)
    test_data = utl.get_partitioning_variables(test_df)

    return train_df, data, valid_df, val_data, test_df, test_data

def estimate_discretization(data, val_data, test_data):
    """Estimate discretization via unsupervised learning"""
    discretizer = dis.DataDriveDiscretizer(delta=0.01, max_pi=4, finite_horizon=True)
    _, report = discretizer.discretize(data, val_data)
    optimal_parts = report.iloc[report.test_score.argmax()].part.astype(int)
    print(f'The estimated optimal number of partitions is {optimal_parts}')

    discretizer = dis.DataDriveDiscretizer(max_pi=optimal_parts, finite_horizon=True)
    discretization_est, _ = discretizer.discretize(test_data, None)
    return discretization_est, report

def estimate_model(discretization_est, test_data):
    """Estimate the dynamic utility model using nested fixed point"""
    init_params = [22] * len(discretization_est) + [-1]  # Initial guess
    bnds = [(1e-12, None)] * len(discretization_est) + [(None, -1e-12)]  # bounds

    estimator = st.DurablesAdoptionEstimate()
    pi = utl.q_to_pi_states(discretization_est, test_data['Q'], dim_q)
    results = estimator.estimate_theta(
        ids=test_data['ids'],
        X=test_data['X'].flatten().tolist(),
        pi=pi,
        y=test_data['Y'],
        init_params=init_params,
        bounds=bnds,
        periods=test_data['periods'],
        max_period=periods,
        discounting_factor=discounting_factor
    )
    return results

# ------------------------------
# Step 4: Main script
# ------------------------------

def simulation():
    print("Generating discretization and transition model...")
    discretization, q_transition = get_discretization()

    print("Generating data...")
    train_df, data, valid_df, val_data, test_df, test_data = get_data(discretization, q_transition)

    print("Estimating discretization...")
    discretization_est, report = estimate_discretization(data, val_data, test_data)

    print("Estimating model parameters...")
    results = estimate_model(discretization_est, test_data)

    estimated_util = results[:-1]
    estimated_price_coeff = results[-1]
    print("Estimated utility values: ", np.round(estimated_util, 3))
    print("True utility values:      ", np.round(buy_util, 3))
    print("Estimated price coefficient:", np.round(estimated_price_coeff, 3))
    print("True price coefficient:     ", price_coeff)

    return discretization, discretization_est, report, results

# Run the pipeline
np.random.seed(0)
simulation()