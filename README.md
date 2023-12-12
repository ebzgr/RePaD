# RePaD

Repad is an algorithm that captures and controls for high-dimensional variables in the dynamic discrete choice models. It builds on the idea behind recursive partitioning to discretize the state space and reduce the dimensionality of a high-dimensional control variable set to a lower-dimensional categorical variable using a weighted sum of decision probabilities and state transition probabilities.
For more information about the algorithm, please read the first draft of the paper in the following.

https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4640309

This repository contains five modules. The discretization algorithm is implemented in the file "discretizer." The rest of the modules are provided to experiment with the algorithm using the extended version of the canonical Rust (1987)  bus engine replacement problem. The modules are as follows:
1. discretization_generator: This module can generate a random discretization in a high-dimensional state space. It can be used to generate a discretized high-dimensional variable set.
2. data_generator: This module can generate simulation data from the extended version of Rust's bus engine replacement problem.
3. discretizer: This module is the main algorithm and discretizes a high-dimensional state space to a one-dimensional categorical variable.
4. estimator: This module can be used to estimate the parameters of the extended version of Rust's bus engine replacement problem.
5. utility: This module provides some functionalities that are used across all other modules.

## Installation

To install the required dependencies for this project, follow these steps:

1. Clone this repository to your local machine.

```
git clone https://github.com/ebzgr/RePaD.git
```

2. It's recommended to create a virtual environment to isolate the dependencies for this project. You can do this with the following commands:

```
python3 -m venv environment_name
source environment_name/bin/activate # On Windows, use environment_name\Scripts\activate
```

3. Once the virtual environment is activated, you can install the dependencies with:

```
pip install -r requirements.txt
```


## Usage

Please use the provided `example.py` file as a guideline for using the algorithm. This script demonstrates how to generate a random discretization, generate data from the Rust bus engine replacement problem, recover the random partitioning using the data-driven discretization algorithm, and estimate the parameters of the problem.

## Contact

Please reach out to Ebi Barzegary (ebzgry@gmail.com) or Hema Yoganarasimhan (hemay@uw.edu) for questions regarding the algorithm or package.

## References:

Rust, John. "Optimal replacement of GMC bus engines: An empirical model of Harold Zurcher." _Econometrica: Journal of the Econometric Society_ (1987): 999-1033.