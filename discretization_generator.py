import numpy as np
import pandas as pd

class RandomDiscretizationGenerator:
    
    def __init__(self, dim_q, dim_active_q, dim_pi, max_q):
        """
        This function genrate a random discretization that can be used in the extended version of Rust(1987) bus 
        engine problem. This algorithm randomly discretize the state space of a variable set of size dim_q ino 
        dim_pi partition where each varaible is in range (0, max_q) using only the first dim_active_q variables.

        Parameters
        ----------
        dim_q : int
            The number of variables in the variabl set.
        dim_active_q : int
            The number of variables that is used to partition the state space. If dim_active_q is 10, it means
            that the partitioning only split the first 10 variables to generate the discretization.
        dim_pi : int
            Number of partitions in the final discretization.
        max_q : int
            The maximum value for each variable.

        Returns
        -------
        None.

        """
        self.dim_q = dim_q
        self.dim_active_q = dim_active_q
        self.dim_pi = dim_pi
        self.max_q = max_q


    @staticmethod 
    def _select_split(discretization, balance):
        """
        Select the next partition for split based on current discretization and the balance variable.
        If it is 0, there is not balancing concerns.
        If it is 1, the algorithm weight to partitions based on their number of current split for the next split.
        If it is 2, at each iteration the partition with the minimum number of current split is chosen for the next split.


        Parameters
        ----------
        discretization : dataframe
            Current discretization.
        balance: int
            Specifies how balance the discretization must be.

        Returns
        -------
        pi : int
            The selected partition number.

        """
        if(balance == 0):
            pi = discretization.sample(n=1)
        elif(balance == 1):
            pi = discretization.sample(n=1, weights=(1/discretization.total_split)**2)
        else:
            pi = discretization.loc[discretization.total_split == discretization.total_split.min()][:1]
            
        return pi
            

    def _add_random_split(self, discretization, balance):
        """
        Get a discretizationing, randomly select one of the partitions and split it into two partitions along
        a randomly selected variable

        Parameters
        ----------
        discretization : dataframe
            Current discretization.

        Returns
        -------
        discretization : dataframe
            New discretization with one more partition.

        """
        new_state = discretization.state.max() + 1
        pi = self._select_split(discretization, balance)
        var = np.random.randint(0, self.dim_active_q)
        val_min, val_max = pi[['q_{}_min'.format(var),'q_{}_max'.format(var)]].values.flatten().tolist()
        while((val_min+1) == val_max):
            pi = self._select_split(discretization, balance)
            var = np.random.randint(0,self.dim_active_q)
            val_min, val_max = pi[['q_{}_min'.format(var),'q_{}_max'.format(var)]].values.flatten().tolist()
            
        split_val = np.int((val_min + val_max)/2)
        discretization.loc[new_state] = discretization.loc[pi.state].values.flatten()
        discretization.loc[new_state,'state']=new_state
        discretization.loc[pi.state,["q_{}_max".format(var),'total_split']] = [split_val,pi.total_split.values[0]+1]
        discretization.loc[new_state,["q_{}_min".format(var),'total_split']] = [split_val, pi.total_split.values[0]+1]
    
        return discretization

    def generate_random_discretization(self, balance = 0):
        """
        Generate a random discretization

        Returns
        -------
        discretization : dataframe
            Dataframe that contains the discretization.
        balance: int
            Specifies how balance the discretization must be. The default value is 0.
            If it is 0, there is not balancing concerns.
            If it is 1, the algorithm weight to partitions based on their number of current split for the next split.
            If it is 2, at each iteration the partition with the minimum number of current split is chosen for the next split.

        """
        discretization =  pd.DataFrame(data={'state':[0],'total_split':[1]})
        for i in range(self.dim_q):
            discretization['q_{}_min'.format(i)]=0
            discretization['q_{}_max'.format(i)]=self.max_q
    
        for i in range(1,self.dim_pi):
            discretization = self._add_random_split(discretization, balance)
        
        return discretization