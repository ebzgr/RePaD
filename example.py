import numpy as np
import data_generator as dag
import discretizer as dis
import utility as utl
import discretization_generator as dig
import estimator as st

'''
Here we present a simple example to generate a random discretization using the discretization generator module, 
generate bus engine replacement problem using the generated random discretization by data generator module, 
recover the random partitioning using data-driven discretization algorithm model, and finally recover 
the bus engine replacement problem parameters using the nested fixed point estimated module.
'''

## Generate the discretization
dim_q = 5
dim_active_q = 2
max_q = 10
dim_pi = 4
replacement_cost = -np.arange(4,8)

par_gen = dig.RandomDiscretizationGenerator(dim_q, dim_active_q, dim_pi, max_q)
discretization = par_gen.generate_random_discretization(2)
discretization['f_dc'] = replacement_cost  # Replacement cost in each state of the discretization
discretization['f_tr'] = np.arange(dim_pi) #The mileage transition in each state of the discretization

## Generate the data

max_mileage = 30
mileage_coefficient=-0.2
discounting_factor=0.9
q_transition = utl.generate_pi_transition(dim_pi, 3, 2) # The state transition in Pi(Q) dimension

data_gen = dag.EngineReplacementDataGenerator(max_mileage=max_mileage, mileage_coefficient=mileage_coefficient, 
                                     discretization=discretization, q_transition=q_transition, 
                                     max_q=max_q, dim_q=dim_q, discounting_factor=discounting_factor)

buses = 500
periods = 100
train_df = data_gen.generate(buses=buses, periods=periods)
data = utl.get_partitioning_variables(train_df) # Convert the dataframe into a dictionary that is usable by discretizer

valid_df = data_gen.generate(buses=buses, periods=periods)
val_data = utl.get_partitioning_variables(valid_df)

test_df = data_gen.generate(buses=buses, periods=periods)
test_data = utl.get_partitioning_variables(test_df)

## Recover the discretization
'''
The default values for lambda and smoothing del in the currect case is good enough, so we do not 
use extensive hyper-parameter tuning to tune those parameter, and use the simple hyper-param tuning. 
However, you may want to tune lamb and smoothing_del as well using hyperopt. In this case, we 
use a simple hyper-param optimization. we set a small value for delta small and a maximum number 
of partitions. We then use the performance on the validation set to choose the optimal number of 
partitions.
'''
discretizer = dis.DataDriveDiscretizer(delta=0.001, max_pi=10) 
_ , report = discretizer.discretize(data, val_data)
# Find the optimal number of partitions, by choosing the one that minimizes score in the validation set.
optimal_parts = report.iloc[report.test_score.argmax()].part.astype(int) 

discretizer = dis.DataDriveDiscretizer(max_pi = optimal_parts)
discretization_est, _ = discretizer.discretize(test_data, None)
## Recover the engine replacement parameters
estimator = st.BusEngineNFXP()
pi = utl.q_to_pi_states(discretization_est, test_data['Q'], dim_q) #Recover the partition of each observation in Q
bestll,f,alpha = estimator.estimate_theta(test_data['ids'], test_data['periods'], test_data['X'].flatten().tolist(), pi, test_data['Y'], discounting_factor = discounting_factor)
discretization_est['f_dc'] = f