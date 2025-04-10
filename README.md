# RePaD

Repad is an algorithm that captures and controls for high-dimensional variables in the dynamic discrete choice models. It builds on the idea behind recursive partitioning to discretize the state space and reduce the dimensionality of a high-dimensional control variable set to a lower-dimensional categorical variable using a weighted sum of decision probabilities and state transition probabilities.
For more information about the algorithm, please read the first draft of the paper in the following.

https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4640309

This repository contains five modules. The discretization algorithm is implemented in the file "discretizer." The rest of the modules are provided to experiment with the algorithm using two canonical settings:
- An infinite-horizon Rust (1987) bus engine replacement problem.
- A finite-horizon durable good adoption problem similar to Song & Chintagunta (2003). The modules are as follows:
1. discretization_generator: This module can generate a random discretization in a high-dimensional state space. It can be used to generate a discretized high-dimensional variable set.
2. data_generator: This module provides two classes to generate simulation data:
   - `EngineReplacementDataGenerator` for infinite-horizon bus engine replacement problem.
   - `DurablesAdoptionDataGenerator` for finite-horizon durable goods adoption problem.
3. discretizer: This module is the main algorithm and discretizes a high-dimensional state space to a one-dimensional categorical variable.
4. estimator: This module provides two classes to estimate model parameters:
   - `BusEngineNFXP` for infinite-horizon bus engine replacement problem.
   - `DurablesAdoptionEstimate` for finite-horizon durable goods adoption problem.
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

Please refer to the following example files to see how to use the package for each setting:
- `Bus engine replacement example.py`
- `Durable good adoption example.py` 
This script demonstrates how to generate a random discretization, generate data, recover the partitioning using the data-driven discretization algorithm, and estimate the parameters of the problem.

## Contact

Please reach out to Ebrahim Barzegary (barzegary@essec.edu) or Hema Yoganarasimhan (hemay@uw.edu) for questions regarding the algorithm or package.

## References:

Rust, John. "Optimal replacement of GMC bus engines: An empirical model of Harold Zurcher." _Econometrica: Journal of the Econometric Society_ (1987): 999-1033.
Song, Inseong, and Pradeep K. Chintagunta. "A Micromodel of New Product Adoption with Heterogeneous and Forward-Looking Consumers: Application to the Digital Camera Category." *Quantitative Marketing and Economics* 1, no. 4 (2003): 371–407.