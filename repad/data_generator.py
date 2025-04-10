import numpy as np
import pandas as pd
from . import utility as utl

def get_data_generator(generator_type, **kwargs):
    if generator_type == 'rust':
        return EngineReplacementDataGenerator(**kwargs)
    elif generator_type == 'durables':
        return DurablesAdoptionDataGenerator(**kwargs)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


class EngineReplacementDataGenerator:

    def __init__(self, max_mileage, mileage_coefficient, discretization, q_transition, max_q=10, dim_q=10, discounting_factor=0.9, ev_eps=10**-9):
        """
        A class to generate sample data from the extended version of Rust(1987) bus engine replacement problem. See more details about the extended version at.

        Parameters
        ----------
        max_mileage : int
            Maximum mileage that a bus can have. After that the maintenance decision does not increase the mileage.
        mileage_coefficient : float
            The coefficient of mileage in the maintenance cost.
        discretization : dataframe
            The given discretization of high-dimensional variable.
        q_transition : enum
            The transition in the high-dimensional state space.
        max_q : int, optional
            Maximum value for each variable in the high-dimensional variable set. The default is 10.
        dim_q : int, optional
            The dimension of the high-dimensional variable set. The default is 10.
        discounting_factor : float, optional
            The discounting factor for the future utility in utility function. The default is 0.9.
        ev_eps : float, optional
            The epsilon value for value function iteration step. The default is 10**-9.

        Returns
        -------
        None.

        """
        self.max_m = max_mileage
        self.alpha = mileage_coefficient
        self.discretization = discretization
        self.dim_q = dim_q
        self.beta = discounting_factor
        self.ev_eps = ev_eps
        self.max_q = max_q

        self.discretization['f_dc'] = self.discretization['f_dc'].astype(float)
        if(isinstance(q_transition,int)):
            self.qst=utl.generate_pi_transition(len(self.discretization), q_transition)
        else:
            self.qst = q_transition
      
        self.st_matrix = self._generate_base_transition_matrix(self.max_m, self.qst, self.discretization)
        self.ev_df = self._calculate_expected_value(self.discretization, self.qst, self.max_m, self.alpha, self.beta, self.st_matrix, self.ev_eps)
    
    
    @staticmethod 
    def _generate_base_transition_matrix(max_m, qst, discretization):
        """
        Generate a base matrix for state transition.

        Parameters
        ----------
        max_m : int
            Maximum mileage that a bus can have. After that the maintenance decision does not increase the mileage.
        qst : matrix
            The state transition in Q.
        discretization : dataframe
            The descritization dataframe.

        Returns
        -------
        output: matrix
            The state transition matrix.

        """
        dim_pi = qst.shape[0]
        mileage = pd.DataFrame(data={'x':np.arange(0,max_m),'pr':0})
        prt = pd.DataFrame(data={'pi':np.arange(0,dim_pi),'pr':0})
        decision = pd.DataFrame(data={'d':np.arange(0,2),'pr':0})
        next_m = pd.DataFrame(data={'next_x':np.arange(0,max_m*2),'pr':0})
        next_state = pd.DataFrame(data={'next_pi':np.arange(0,dim_pi),'pr':0})

        tmp = mileage.merge(prt.merge(decision.merge(next_m.merge(next_state))))
        tmp['f_tr'] = discretization.f_tr[tmp.pi].values
        
        replace = tmp[(tmp.d==1)&(tmp.next_x==tmp.f_tr)].copy()
        tmp['pr'] = tmp['pr'].astype(float)
        tmp.loc[(tmp.d==1)&(tmp.next_x==tmp.f_tr),'pr']=np.squeeze(np.asarray(qst[replace.pi,replace.next_pi]))
        maintain = tmp[(tmp.d==0)&((tmp.x+tmp.f_tr)==tmp.next_x)]
        tmp.loc[(tmp.d==0)&((tmp.x+tmp.f_tr)==tmp.next_x),'pr']=np.squeeze(np.asarray(qst[maintain.pi,maintain.next_pi]))

        extra = tmp[tmp.next_x>max_m-2].groupby(['x','pi','d','next_pi']).pr.sum().reset_index()
        extra['next_x']=max_m-1
        tmp = tmp[tmp.next_x<max_m-1]
        tmp = pd.concat([tmp, extra], axis=0)

        base_transition_matrix = pd.pivot_table(tmp,index=['x','pi','d'],columns = ['next_x','next_pi'], values='pr', aggfunc='sum',fill_value=0)
        return base_transition_matrix.values

    @staticmethod 
    def _calculate_expected_value(discretization, qst, max_m, alpha, beta, st_matrix, eps):
        """
        Generate the value function dataframe for decision probability estimation. The dataframe includes 
        value function, flow utility and decision probability for each point in state space.

        Parameters
        ----------
        discretization : dataframe
            The descritization dataframe.
        qst : matrix
            The state transition in Q.
        max_m : int
            Maximum mileage value.
        alpha : float
            The coefficient of mileage for maintenance cost.
        beta : float
            Value of discounting factor.
        st_matrix : matrix
            State transition matrix.
        eps : float
            The epsilon value for value function convergance in the iteration step.

        Returns
        -------
        data : TYPE
            DESCRIPTION.

        """
        dim_pi = len(discretization)
        dt = pd.DataFrame(data={'pi':np.arange(0,dim_pi),'tmp':1})
        mileage = pd.DataFrame(data={'x':np.arange(0,max_m),'tmp':1})
        des = pd.DataFrame(data={'d':np.arange(0,2),'tmp':1})
        data = mileage.merge(dt.merge(des))
        data = data.drop('tmp', axis=1)
        
        ev = np.zeros(max_m*dim_pi*2).reshape(max_m*dim_pi,2)
        nev = np.zeros(max_m*dim_pi*2).reshape(max_m*dim_pi,2)
        u = np.matrix((1-data['d'])*(alpha*data['x'])+(data['d']*(pd.get_dummies(data['pi']).dot(discretization.f_dc)))).T.reshape(max_m*dim_pi,2).astype(float)

        max_diff = 1
        while(eps<max_diff):
            v = np.log(np.exp(ev).sum(axis=1))
            nev = u + (beta*st_matrix.dot(v)).reshape(max_m*dim_pi,2)
            max_diff = np.max(np.abs(ev-nev))
            ev = nev.copy()


        ev_exp = np.exp(ev)
        data['u']=u.reshape(max_m*dim_pi*2,1)
        data['ev']=ev.reshape(max_m*dim_pi*2,1)
        data['pr']=(ev_exp/ev_exp.sum(axis=1)).reshape(max_m*dim_pi*2,1)
        return data

    @staticmethod     
    def _random_q_from_pi_states(discretization, partition, dim_q):
        """
        Generate a random value for the high-dimensional variable given partitions using discretization

        Parameters
        ----------
        discretization : dataframe
            The dataframe of discretization.
        partition : array
            A list of partitions
        dim_q : int
            dimension of q.

        Returns
        -------
        output: matrix
            A matrix containing the generated high-dimensional values for q.

        """
        repeats = np.unique(partition, return_counts=True)[1]
        result = np.zeros(len(partition)*dim_q).reshape(len(partition),dim_q)
        base_row = 0
        for i in range(0,len(repeats)):
            row = discretization[discretization.state == i].iloc[0]
            for q in range(0,dim_q):
                qs = np.random.randint(low=row['q_{}_min'.format(q)], high=row['q_{}_max'.format(q)],size=repeats[i]).reshape(repeats[i],1)
                result[base_row:base_row+repeats[i],q:q+1] = qs
            base_row += repeats[i]


        resort = np.zeros(len(partition))
        tmp = np.argsort(partition)
        for i in range(0,len(partition)):
            resort[tmp[i]]=i

        return result[resort.astype(int)]

    def _sample_generator(self, ev_df, buses, periods, q_int):
        """
        The private function that generate a sample data using the state space value functions, number of buses, 
        number of periods, and the initial value for high-dimensional variable set

        Parameters
        ----------
        ev_df : dataframe
            The state space value function dataframe.
        buses : int
            Number of buses.
        periods : int
            Number of periods.
        q_int : matrix
            Initial values for high-dimensional variable set.

        Returns
        -------
        output : dataframe
            The dataframe containing the data for buses in each period including their dimension in q,
            and the decision for replacement or maintenance.
        """
        baseDf = pd.DataFrame(data={'id':np.arange(0,buses),'x':0})
        baseDf['pi'] = utl.q_to_pi_states(self.discretization, q_int, self.dim_q)
        q_matrix_titles = ['q_'+str(i) for i in range(0, self.dim_q)]
        baseDf[q_matrix_titles] =  pd.DataFrame(q_int,columns=[q_matrix_titles])

        dataframes = []  # Create an empty list to store the temporary DataFrames
        for t in range(periods):
            tmp = baseDf.merge(ev_df[ev_df.d==1][['pi','x','pr']], on=(['pi','x']),how='left').copy()
            tmp['d']=np.random.binomial(1,tmp['pr'],len(tmp['pr']))
            tmp['t']=t
            dataframes.append(tmp[['id','t','x','d','pr','pi']+q_matrix_titles])  # Append the temporary DataFrame to the list

            baseDf['x'] = utl.mileage_transition(baseDf['x'].values, tmp['d'].values, self.max_m, self.discretization.f_tr[baseDf.pi].values)

            baseDf['pi'] = utl.pi_state_transition( baseDf['pi'].values, self.qst)
            baseDf[q_matrix_titles]=self._random_q_from_pi_states(self.discretization, baseDf['pi'].values, self.dim_q)

        results = pd.concat(dataframes, ignore_index=True)  # Concatenate all the DataFrames at once

        int_cols = ['id','t','x','d','pi'] + q_matrix_titles
        results[int_cols] = results[int_cols].astype(int)
        return results



    def generate(self, buses, periods, q_initial=None):
        """
        Generate a data sample with the number of given buses for the given number of time periods.

        Parameters
        ----------
        buses : int
            The number of buses.
        periods : int
            The number of data periods.
        q_initial : matrix, optional
            The initial value for the high-dimensional variable for each bus at period 0. The default is None.
            If the input in None, the values are randomly generated. q_initial is neglected is the dimension
            of q is set to zero.

        Raises
        ------
        ValueError
            Return error if the dimension of the initial_q is not correct.

        Returns
        -------
        output : dataframe
            The dataframe containing the data for buses in each period including their dimension in q,
            and the decision for replacement or maintenance.

        """
        if(q_initial==None):
            q_int=utl.generate_initial_q(buses, self.dim_q, self.max_q)
        else:
            q_int = q_initial
            bus_dim, q_dim = q_initial.shape
            if(bus_dim != buses):
                raise ValueError('The dimension of initial q state is not aligned with the number of buses.')

            if(q_dim != self.dim_q):
                raise ValueError('The dimension of initial q state is not aligned with the given dimension of q.')
  
        output = self._sample_generator(self.ev_df, buses, periods, q_int)
        return output
    
class DurablesAdoptionDataGenerator:

    def __init__(self, max_price, min_price, price_coefficient, discretization, q_transition, max_period, max_q=10, dim_q=10, discounting_factor=0.9, ev_eps=10**-9):
        """
        A class to generate sample data from the extended version of Song & Chintagunta (2003) durables purchase problem.

        Parameters
        ----------
        max_price : int
            Maximum price that a good can have. To avoid floating point issues, the bounds on price is denoted in integer 
            multiples of 0.05. So a maximum price of 20 would correspond to max_price = 400.
        min_price : int
            Minimum price that a good can have. To avoid floating point issues, the bounds on price is denoted in integer 
            multiples of 0.05. So a minimum price of 0.05 would correspond to min_price = 1.
        price_coefficient : float
            The coefficient of price in the flow utility/marginal utility of income.
        discretization : dataframe
            The given discretization of high-dimensional variable.
        q_transition : enum
            The transition in the high-dimensional state space.
        max_q : int, optional
            Maximum value for each variable in the high-dimensional variable set. The default is 10.
        dim_q : int, optional
            The dimension of the high-dimensional variable set. The default is 10.
        discounting_factor : float, optional
            The discounting factor for the future utility in utility function. The default is 0.9.
        ev_eps : float, optional
            The epsilon value for value function iteration step. The default is 10**-9.

        Returns
        -------
        None.

        """
        self.max_p = max_price
        self.min_p = min_price
        self.beta = price_coefficient
        self.discretization = discretization
        self.dim_q = dim_q
        self.delta = discounting_factor
        self.ev_eps = ev_eps
        self.max_q = max_q
        self.max_period = max_period

        self.discretization['f_dc'] = self.discretization['f_dc'].astype(float)
        if(isinstance(q_transition,int)):
            self.qst=utl.generate_pi_transition(len(self.discretization), q_transition)
        else:
            self.qst = q_transition
      
        self.st_matrix = self._generate_base_transition_matrix(self.max_p, self.min_p, self.qst, self.discretization)
        self.ev_df = self._calculate_expected_value(self.discretization, self.qst, self.max_p, self.min_p, self.max_period, self.delta, self.beta, self.st_matrix, self.ev_eps)
    
    @staticmethod 
    def _generate_base_transition_matrix(max_p, min_p, qst, discretization):
        """
        Generate a base matrix for state transition.

        Parameters
        ----------
        max_p : int
            Maximum price that a good can have. To avoid floating point issues, the bounds on price is denoted in integer 
            multiples of 0.05. So a maximum price of 20 would correspond to max_p = 400.
        qst : matrix
            The state transition in Q.
        discretization : dataframe
            The descritization dataframe.

        Returns
        -------
        output: matrix
            The state transition matrix.

        """
        dim_pi = qst.shape[0]
        x = pd.DataFrame(data={'x':np.arange(min_p,max_p+1),'pr':0}) 
        prt = pd.DataFrame(data={'pi':np.arange(0,dim_pi),'pr':0})
        decision = pd.DataFrame(data={'d':np.arange(0,1),'pr':0}) # only wait, as if purchase there is no state transition (i.e. consumer drops out of the market)
        next_x = pd.DataFrame(data={'next_x':np.arange(min_p,max_p+1),'pr':0})
        next_state = pd.DataFrame(data={'next_pi':np.arange(0,dim_pi),'pr':0})

        tmp = x.merge(prt.merge(decision.merge(next_x.merge(next_state))))
        tmp['f_tr'] = discretization.f_tr[tmp.pi].values
        
        wait = tmp[(tmp.d==0)&(tmp.next_x==np.maximum(tmp.x-(tmp.f_tr),min_p))].copy() 
        tmp['pr'] = tmp['pr'].astype(float)
        tmp.loc[(tmp.d==0)&(tmp.next_x==np.maximum(tmp.x-(tmp.f_tr),min_p)),'pr']=np.squeeze(np.asarray(qst[wait.pi,wait.next_pi])) 
    
        base_transition_matrix = pd.pivot_table(tmp,index=['x','pi','d'],columns = ['next_x','next_pi'], values='pr', aggfunc='sum',fill_value=0)

        return base_transition_matrix.values

    @staticmethod 
    def _calculate_expected_value(discretization, qst, max_p, min_p, max_period, discounting_factor, beta, st_matrix, eps):
        """
        Generate the value function dataframe for decision probability estimation. The dataframe includes 
        value function, flow utility and decision probability for each point in state space.

        Parameters
        ----------
        discretization : dataframe
            The descritization dataframe.
        qst : matrix
            The state transition in Q.
        max_p : int
            Maximum price that a good can have. To avoid floating point issues, the bounds on price is denoted in integer 
            multiples of 0.05. So a maximum price of 20 would correspond to max_p = 400.
        beta  : float
            Marginal utility of income.
        delta : float
            Value of discounting factor.
        st_matrix : matrix
            State transition matrix.
        eps : float
            The epsilon value for value function convergance in the iteration step.

        Returns
        -------
        data : TYPE
            DESCRIPTION.

        """
        dim_pi = len(discretization)
        dt = pd.DataFrame(data={'pi':np.arange(0,dim_pi),'tmp':1})
        x = pd.DataFrame(data={'x':np.arange(min_p,max_p+1),'tmp':1}) 
        des = pd.DataFrame(data={'d':np.arange(0,2),'tmp':1})
        period = pd.DataFrame(data={'t':np.arange(0,max_period),'tmp':1})
        data = period.merge(x.merge(dt.merge(des)))
        data = data.drop('tmp', axis=1)

        data['u'] = data['d']*(((pd.get_dummies(data['pi']).dot(discretization.f_dc)))/(1-discounting_factor) + beta*data['x'])

        rows, cols = st_matrix.shape
        new_matrix = np.zeros((2 * rows, cols), dtype=st_matrix.dtype)
        new_matrix[::2] = st_matrix
        st_matrix = new_matrix

        chunk_size = len(data[data.t==0])
        new_v = np.zeros(len(data)).reshape(-1,1)
        new_v[(max_period-1)*chunk_size:max_period*chunk_size] = data[(max_period-1)*chunk_size:max_period*chunk_size].u.values.reshape(-1,1)
        data['v'] = new_v
        for i in np.arange(max_period-1,0,-1):
            update_start = (i-1)*chunk_size
            target = data[update_start+chunk_size:update_start+2*chunk_size]
            target_v = target.v.values.reshape(-1, 2)
            ev = np.matrix(np.log(np.exp(target_v).sum(axis=1))).T
            new_v[update_start:update_start+chunk_size] = data[update_start:update_start+chunk_size].u.values.reshape(-1,1) + discounting_factor*(st_matrix*ev)
            data['v'] = new_v


        prs = utl.get_soft_max(data.v.values.reshape(-1,2))
        data['pr'] = prs.reshape(-1,1)
        return data

    @staticmethod     
    def _random_q_from_pi_states(discretization, partition, dim_q):
        """
        Generate a random value for the high-dimensional variable given partitions using discretization

        Parameters
        ----------
        discretization : dataframe
            The dataframe of discretization.
        partition : array
            A list of partitions
        dim_q : int
            dimension of q.

        Returns
        -------
        output: matrix
            A matrix containing the generated high-dimensional values for q.

        """
        repeats = np.unique(partition, return_counts=True)[1]
        result = np.zeros(len(partition)*dim_q).reshape(len(partition),dim_q)
        base_row = 0
        for i in range(0,len(repeats)):
            row = discretization[discretization.state == i].iloc[0]
            for q in range(0,dim_q):
                qs = np.random.randint(low=row['q_{}_min'.format(q)], high=row['q_{}_max'.format(q)],size=repeats[i]).reshape(repeats[i],1)
                result[base_row:base_row+repeats[i],q:q+1] = qs
            base_row += repeats[i]


        resort = np.zeros(len(partition))
        tmp = np.argsort(partition)
        for i in range(0,len(partition)):
            resort[tmp[i]]=i

        return result[resort.astype(int)]

    def _sample_generator(self, ev_df, consumers, periods, q_int):
        """
        The private function that generate a sample data using the state space value functions, number of consumers, 
        number of periods, and the initial value for high-dimensional variable set

        Parameters
        ----------
        ev_df : dataframe
            The state space value function dataframe.
        consumers : int
            Number of consumers.
        periods : int
            Number of periods.
        q_int : matrix
            Initial values for high-dimensional variable set.

        Returns
        -------
        output : dataframe
            The dataframe containing the data for consumers in each period including their dimension in q,
            and the decision for purchase or waiting.
        """

        baseDf = pd.DataFrame(data={'id':np.arange(0,consumers),'x':self.max_p})
        baseDf['pi'] = utl.q_to_pi_states(self.discretization, q_int, self.dim_q)
        q_matrix_titles = ['q_'+str(i) for i in range(0, self.dim_q)]
        baseDf[q_matrix_titles] =  pd.DataFrame(q_int,columns=[q_matrix_titles])

        dataframes = []  # Create an empty list to store the temporary DataFrames
        for t in range(periods):
            baseDf['t']=t
            tmp = baseDf.merge(ev_df[ev_df.d==1][['t','pi','x','pr']], on=(['t','pi','x']),how='left').copy()
            tmp['d']=np.random.binomial(1,tmp['pr'],len(tmp['pr']))
            dataframes.append(tmp[['id','t','x','d','pr','pi']+q_matrix_titles])  # Append the temporary DataFrame to the list
            
            # prepare next x and next pi
            baseDf['x'] = utl.price_transition(baseDf['x'].values, tmp['d'].values, self.min_p, self.discretization.f_tr[baseDf.pi].values)
            baseDf['pi'] = utl.pi_state_transition( baseDf['pi'].values, self.qst)
            baseDf[q_matrix_titles]=self._random_q_from_pi_states(self.discretization, baseDf['pi'].values, self.dim_q)

        results = pd.concat(dataframes, ignore_index=True)  # Concatenate all the DataFrames at once

        int_cols = ['id','t','d','pi'] + q_matrix_titles
        results[int_cols] = results[int_cols].astype(int)

        first_d1 = results[results['d'] == 1].groupby('id')['t'].min().reset_index()
        first_d1.columns = ['id', 'first_t_d1']

        # Step 2: Merge this information back with the original DataFrame
        results = results.merge(first_d1, on='id', how='left')

        # Step 3: Filter out rows where `t` is greater than the first `t` where `d = 1`
        results = results[results['t'] <= results['first_t_d1']].drop(columns='first_t_d1')
          
        return results



    def generate(self, consumers, periods, q_initial=None):
        """
        Generate a data sample with the number of given buses for the given number of time periods.

        Parameters
        ----------
        consumers : int
            The number of consumers.
        periods : int
            The number of data periods.
        q_initial : matrix, optional
            The initial value for the high-dimensional variable for each bus at period 0. The default is None.
            If the input in None, the values are randomly generated. q_initial is neglected is the dimension
            of q is set to zero.

        Raises
        ------
        ValueError
            Return error if the dimension of the initial_q is not correct.

        Returns
        -------
        output : dataframe
            The dataframe containing the data for consumers in each period including their dimension in q,
            and the decision for purchase or waiting.

        """
        if(q_initial==None):
            q_int=utl.generate_initial_q(consumers, self.dim_q, self.max_q)
        else:
            q_int = q_initial
            cons_dim, q_dim = q_initial.shape
            if(cons_dim != consumers):
                raise ValueError('The dimension of initial q state is not aligned with the number of buses.')

            if(q_dim != self.dim_q):
                raise ValueError('The dimension of initial q state is not aligned with the given dimension of q.')
  
        output = self._sample_generator(ev_df=self.ev_df,consumers=consumers, periods=periods, q_int=q_int)
        return output