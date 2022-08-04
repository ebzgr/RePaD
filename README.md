# RePaD

Repad is an algorithm that controls a high-dimensional variable set in the dynamic discrete choice modeling settings. It uses the idea behind recursive partitioning to discretize the state space and reduce the dimensionality of a high-dimensional control variable set to a one-dimensional categorical variable using a weighted sum of decision probabilities and state transition probabilities.
For more information about the algorithm, please read the first draft of the paper in the following.

https://arxiv.org/pdf/2208.01476.pdf

This repository contains five modules. The discretization algorithm is implemented in the file "discretizer." The rest of the modules are provided to experiment with the algorithm using the extended version of the canonical Rust (1987)  bus engine replacement problem. The modules are as the following:
1. discretization_generator: This module can generate a random discretization in a high-dimensional state space. It can be used to generate a discretized high-dimensional variable set.
2. data_generator: This module can generate simulation data from the extended version of Rust's bus engine replacement problem.
3. discretizer: This module is the main algorithm and discretizes a high-dimensional state space to a one-dimensional categorical variable.
4. estimator: This module can be used to estimate the parameters of the extended version of Rust's bus engine replacement problem.
5. utility: This module provides some functionalities that are used across all other modules.

Use the provided example file as well as in-code documentation as a guideline for using the algorithm. Please reach out to Ebi Barzegary (ebar@uw.edu) or Hema Yoganarasimhan (hemay@uw.edu) for questions regarding the algorithm or package.

### References:
Rust, John. "Optimal replacement of GMC bus engines: An empirical model of Harold Zurcher." _Econometrica: Journal of the Econometric Society_ (1987): 999-1033.
