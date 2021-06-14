import numpy as np
import pandas as pd
import utility as utl
import statsmodels.formula.api as smf
import statsmodels.api as sm

class BusEngineNFXP:
    def __init__(self, method='rlogit', min_diff=10**-6, max_iter=100):
        """
        Estimate the extended version of Rust(1987) bus engine problem using nested fixed point

        Parameters
        ----------
        method : string, optional
            The algorithm that is used for estimating parameters. There are two methods implemented at the moment.
            In 'rlogit' we repeatedly use estimated values for static model to calculate value functions, and use them
            as offset for the next iterations static model estimation.
            In the direct model we calculate the deratives of the nested fixed point and use Adam accelerator for 
            estimating using gradient descent. The default method is 'rlogit'.
        min_diff : float, optional
            The minimum difference between estimated parameters in two consecutive iteration for convergence. 
            The default is 10**-6.
        max_iter : int, optional
            Maximum number of iteration for parameter estimation. The default is 100.

        Returns
        -------
        None.

        """
        self.method = method
        self.min_diff = min_diff
        self.max_iter = max_iter
        self.history = []

    def _optimize_theta_rlogit(self, df, alpha, f, st_matrix, states_df, discounting_factor, max_x, dim_pi, J):
        """
        Estimate parameters using repeated static model estimation.

        Parameters
        ----------
        df : dataframe
            The data dataframe.
        alpha : float
            The maintenance coefficient.
        f : array
            The replacement cost in each partition of the discretization.
        st_matrix : matrix
            The state transition matrix.
        states_df : dataframe
            The dataframe of states.
        discounting_factor : float
            The discounting coefficient.
        max_x : int
            The maximum value for x variables.        
        dim_pi : int
            The total number of partitions in the discretization.
        J : int
            The number of possible values for dependent variable.

        Raises
        ------
        ValueError
            In some cases the algorithm diverges and become problematic. This issue needs more analysis.

        Returns
        -------
        alpha : float
            The maintenance coefficient.
        f : array
            The replacement cost in each partition of the discretization.
        ll : float
            The likelihood of observing the data in the given coefficients.
            
        """
        max_diff = np.inf
        last_params = np.concatenate((f,alpha))
        i = 0
        while((max_diff>self.min_diff)&(i<self.max_iter)):
            df, state_ev_df = _calculate_observation_ev(df, alpha, f, st_matrix, states_df, discounting_factor, max_x, dim_pi, J, self.history)         
            model = _logit_estimate(df,discounting_factor,last_params)
            new_params = model.params.values
            
            if(np.isnan(new_params).any()):
                raise ValueError('The estimated parameters contain nan. Estimated parameters: {}. Alpha history: {}'.format(new_params,self.history))
                
            alpha = -np.array([new_params[-1]])
            f = np.array(new_params[:-1])
            
            self.history.append(alpha)
            
            df = df.drop(['delta_ev'],1)
            max_diff = max(np.abs((last_params-new_params)))
            last_params = new_params
            i = i + 1
            
        ll = utl.calculate_ll(model.fittedvalues, df.d.values)
        return alpha, f, ll
    
    def _optimize_theta_direct(self, df, alpha, f, st_matrix, states_df, discounting_factor, max_x, dim_pi, J, learning_rate):
        """
        Estimate parameters using gradient descent with Adam booster.

        Parameters
        ----------
        df : dataframe
            The data dataframe.
        alpha : float
            The maintenance coefficient.
        f : array
            The replacement cost in each partition of the discretization.
        st_matrix : matrix
            The state transition matrix.
        states_df : dataframe
            The dataframe of states.
        discounting_factor : float
            The discounting coefficient.
        max_x : int
            The maximum value for x variables.        
        dim_pi : int
            The total number of partitions in the discretization.
        J : int
            The number of possible values for dependent variable.
        learning_rate : float
            The learning rate in gradient descent.

        Raises
        ------
        ValueError
            In some cases the algorithm diverges and become problematic. This issue needs more analysis.

        Returns
        -------
        alpha : float
            The maintenance coefficient.
        f : array
            The replacement cost in each partition of the discretization.
        ll : float
            The likelihood of observing the data in the given coefficients.
            
        """
        df, state_ev_df  = _calculate_observation_ev(df, alpha, f, st_matrix, states_df, discounting_factor, max_x, dim_pi, J, [])
        init = np.concatenate((f,alpha))

        params, ll = gradient_descent(df, discounting_factor, states_df, st_matrix, J, dim_pi, max_x, w_init=init,
                                      learning_rate=learning_rate, max_iterations=self.max_iter, threshold=self.min_diff)
        alpha = np.array([params[-1]])
        f = np.array(params[:-1])
        return alpha, f, ll
    
    
    @staticmethod
    def _calculate_transition_matrix(df, dim_pi, J, max_x):
        """
        Calculate the state transition matrix.

        Parameters
        ----------
        df : dataframe
            The data dataframe.
        dim_pi : int
            The total number of partitions in the discretization.
        J : int
            The number of possible values for dependent variable.
        max_x : int
            The maximum value for x variables.

        Returns
        -------
        matrix
            The state transition matrix.

        """
        x = pd.DataFrame(data={'x' : np.arange(0,max_x),'pr':0})
        pi_state = pd.DataFrame(data={'pi':np.arange(0,dim_pi),'pr':0})
        decision = pd.DataFrame(data={'d':np.arange(0,J),'pr':0})

        next_x = pd.DataFrame(data={'next_x' : np.arange(0,max_x),'pr':0})
        next_state = pd.DataFrame(data={'next_pi':np.arange(0,dim_pi),'pr':0})

        blank = x.merge(pi_state.merge(decision.merge(next_x.merge(next_state))))
        base_st_df = pd.pivot_table(blank,index=['x','pi','d'], columns = ['next_x','next_pi'], values='pr', aggfunc='sum',fill_value=0)
        
        max_t = df.t.max()
        df = df.sort_values(['id','t'])
        df['next_pi'] = df['pi'].shift(-1)
        df['next_x'] = df['x'].shift(-1)
        df['tmp'] = 1
        cols = ['x','d','pi','next_x','next_pi']
        sumto = lambda x: x / x.sum()
        count = df[df.t!=max_t].groupby(cols)['id'].count().rename("counts")
        pr = count.groupby(level=cols[:3]).transform(sumto).rename("pr")
        st = pd.concat([count, pr], axis=1).reset_index()

        st['next_x'] = st['next_x'].astype('int64')
        st['next_pi'] = st['next_pi'].astype('int64')

        tmp = pd.pivot_table(st,index=['x', 'pi', 'd']
                     ,columns=['next_x','next_pi'], values='pr', aggfunc='sum',fill_value=0)

        base_st_df.update(tmp)

        return base_st_df.values


    def estimate_theta(self, ids, periods, X, pi, y, discounting_factor = 0.9, learning_rate=0.2):
        """
        Estimate bus engine replacement coefficient.

        Parameters
        ----------
        ids : array
            An array containing observation's id.
        periods : array
            An array containing observation's period.
        X : array
            An array containing observation's X variable set.
        pi : array
            An array containing observation's discretizes high-dimensional value.
        y : array
            An array containing observation's dependent variable.
        discounting_factor : float
            The discounting coefficient. The default is 0.9.
        learning_rate : float
            The learning rate in gradient descent. The default is 0.2.

        Returns
        -------
        bestll : float
            The likelihood of observing the data in the given coefficients.
        f : array
            The replacement cost in each partition of the discretization.
        alpha : float
            The maintenance coefficient.

        """
        self.history = []
        df = pd.DataFrame(data={'id':ids, 't':periods, 'd':y, 'x':X, 'pi':pi})
        J = int(df.d.max()) + 1
        max_x = int(df.x.max()) + 2
        dim_pi = int(df.pi.max()) + 1

        alpha = np.array([0])
        f = np.ones(dim_pi)
        
        st_matrix = self._calculate_transition_matrix(df, dim_pi, J, max_x)           
        states_df = utl.generate_base_state_dataframe(dim_pi, max_x, J)
        if(self.method == 'rlogit'):
            alpha, f, bestll = self._optimize_theta_rlogit(df, alpha, f, st_matrix, states_df, discounting_factor, max_x, dim_pi, J) #Optimize the theta with the given partitioning
        else:
            alpha, f, bestll = self._optimize_theta_direct(df, alpha, f, st_matrix, states_df, discounting_factor, max_x, dim_pi, J, learning_rate)
            
        return bestll,f,alpha


def _calculate_observation_ev(df, alpha, f, st_matrix, states_df, discounting_factor, max_x, dim_pi, J, history):
    """
    Calculate the EVs for each observation using the calculates states value functions. Since the logit estimation 
    uses difference in utilities, we set the maintenance cost to zero.

    Parameters
    ----------
    df : dataframe
        The data dataframe.
    alpha : float
        The maintenance coefficient.
    f : array
        The replacement cost in each partition of the discretization.
    st_matrix : matrix
        The state transition matrix.
    states_df : dataframe
        The dataframe of states.
    discounting_factor : float
        The discounting coefficient.
    max_x : int
        The maximum value for x variables.        
    dim_pi : int
        The total number of partitions in the discretization.
    J : int
        The number of possible values for dependent variable.
    history : array
        The history of estimated alpha. It is used for debugging code.
        
    Returns
    -------
    df : dataframe
        The dataframe that contains value functions.
    state_ev_df : dataframe
        The dataframe containing states, their flow utility, value function and decision probabilities.

    """
    state_ev_df = _calculate_state_ev(alpha, f, st_matrix, states_df, discounting_factor, max_x, dim_pi, J, history)
    state_ev_df['delta_ev']=state_ev_df['ev']-state_ev_df['ev'].shift(1)
    if('delta_ev' in df):
        df = df.drop(['delta_ev'],1)
    if('pr' in df):
        df = df.drop(['pr'],1)        
    df = df.merge(state_ev_df[state_ev_df.d==1][['delta_ev','x','pi','pr']],on=['x','pi'],how='left')
    
    return df, state_ev_df


def _calculate_state_ev(alpha, f, st_matrix, states_df, discounting_factor, max_x, dim_pi, J, history, min_diff = 10**-10): 
    """
    Calculate the expected value of being in each state

    Parameters
    ----------
    alpha : float
        The maintenance coefficient.
    f : array
        The replacement cost in each partition of the discretization.
    st_matrix : matrix
        The state transition matrix.
    states_df : dataframe
        The dataframe of states.
    discounting_factor : float
        The discounting coefficient.
    max_x : int
        The maximum value for x variables.        
    dim_pi : int
        The total number of partitions in the discretization.
    J : int
        The number of possible values for dependent variable.
    history : array
        The history of estimated alpha. It is used for debugging code.
    min_diff : float, optional
        The minimum two consecutive iteration difference for bellman iteration convergence. The default is 10**-10.

    Returns
    -------
    states_df : dataframe
        The dataframe containing states, their flow utility, value function and decision probabilities.

    """
    
    v = np.zeros(max_x*dim_pi*J).reshape(max_x*dim_pi,J)
    u = np.matrix((1-states_df['d'])*(alpha*states_df['x']+utl.hot_encode(states_df['pi']).dot(-f))).T.reshape(max_x*dim_pi,J)

    max_diff = np.inf
    while(min_diff<max_diff):
        nv = u + (discounting_factor*(st_matrix.dot(np.log(np.exp(v).sum(axis=1))))).reshape(max_x*dim_pi,J) # Bellman 
        max_diff = np.max(np.abs(v-nv))
        v = nv.copy()
        
    ev = st_matrix.dot(np.log(np.exp(v).sum(axis=1))).reshape(max_x*dim_pi,J)
    
    if(np.isnan(ev).any()):
        raise ValueError('The estimated EV contains nan. The proposed alpha and f are {} and {}. Alpha history: {}'.format(alpha, f, history))    
    
    states_df['u']=u.reshape(max_x*dim_pi*J,1)
    states_df['ev']=ev.reshape(max_x*dim_pi*J,1)
    states_df['pr']= (np.exp(v)/np.exp(v).sum(axis=1)).reshape(max_x*dim_pi*J,1)
    return states_df


def _logit_estimate(df, discounting_factor,init_params):
    """
    Estimate an static model using future values as offset.

    Parameters
    ----------
    df : dataframe
        Dataframe of data.
    discounting_factor : float
        The discounting coefficient.
    init_params : array
        Array of initial values for maintenance mileage coefficient and replacement costs.

    Returns
    -------
    classifier : model
        The classifier model.

    """
    offset = discounting_factor*df['delta_ev']
    classifier = smf.glm(formula='d ~ C(pi) + x - 1', data=df, offset=offset,
                    family = sm.families.Binomial()).fit(start_params=init_params, method='bfgs')
    return classifier


def func(params, df, discounting_factor, states_df, st_matrix, J, dim_pi, max_x):
    """
    The objective function for minimization in the gradient descent algorithm

    Parameters
    ----------
    params : array
        Array of maintenance mileage coefficient and replacement costs.
    df : dataframe
        Dataframe of data.
    discounting_factor : float
        The discounting coefficient.
    states_df : dataframe
        The dataframe of states.
    st_matrix : matrix
        The state transition matrix.
    J : int
        The number of possible values for dependent variable.
    dim_pi : int
        The total number of partitions in the discretization.
    max_x : int
        The maximum value for x variables.

    Returns
    -------
    ll : float
        The objective function value at the given point.  
    derivatives : array
        The derativates of the objective function at the given point.

    """
    alpha = np.array([params[-1]])
    f = np.array(params[:-1])

    df, state_ev_df  = _calculate_observation_ev(df, alpha, f, st_matrix, states_df, discounting_factor, max_x, dim_pi, J, [])
    derives = np.squeeze(np.asarray(get_derivatives(st_matrix, state_ev_df, discounting_factor, df)))/len(df)
    ll = -utl.calculate_ll(df.pr, df.d.values)/len(df)  
    derivatives = -np.squeeze(np.asarray(derives))
    return ll, derivatives

 
def gradient_descent(df, discounting_factor, states_df, st_matrix, J, dim_pi, max_x,
                     w_init, learning_rate, max_iterations, threshold):
    """
    Calculate the mileage coefficient and replacement costs using gradient descent with Adaptive Moment Estimation.
    Read Kingma, D. P., & Ba, J. L. (2015) for more info about Adam.

    Parameters
    ----------
    df : dataframe
        The dataframe of data.
    discounting_factor : float
        The discounting coefficient.
    states_df : dataframe
        The dataframe of states.
    st_matrix : matrix
        The state transition matrix.
    J : int
        The number of possible values for dependent variable.
    dim_pi : int
        The total number of partitions in the discretization.
    max_x : int
        The maximum value for x variables.
    w_init : array
        Initial values for parameters.
    learning_rate : float
        The learning rate in gradient descent.
    max_iterations : int
        The maximum number of iteration in the gradient descent algorithm.
    threshold : float
        The minimum difference between estimated parameters in two consecutive iteration for convergence.

    Returns
    -------
    alpha : float
        The maintenance coefficient.
    f : array
        The replacement cost in each partition of the discretization.

    """
    
    m_dw = np.zeros(w_init.shape)
    v_dw = np.zeros(w_init.shape)
    beta1=0.9
    beta2=0.999
    w = w_init
    w_history = w
    f_history = 0

    i = 1
    diff = np.inf
    
    while  i<max_iterations and diff>threshold:
        ll, derives = func(w, df, discounting_factor, states_df, st_matrix, J, dim_pi, max_x)
        
        m_dw = beta1*m_dw + (1-beta1)*derives
        v_dw = beta2*v_dw + (1-beta2)*(derives**2)
        m_dw_corr = m_dw/(1-beta1**i)
        v_dw_corr = v_dw/(1-beta2**i)
        
        w = w - learning_rate*(m_dw_corr/(np.sqrt(v_dw_corr)+threshold))
        w_history = np.vstack((w_history,w))
        f_history = np.vstack((f_history,ll))
        
        i+=1
        diff = np.max(np.absolute(w_history[-1]-w_history[-2]))
        
    alpha = w_history[-1]
    f = f_history[-1]
    
    return alpha, f


def _get_observations_score(st_matrix, state_ev_df, discounting_factor, df):
    """
    Calculate the observation scores using method explain in 
    https://mark-ponder.com/tutorials/discrete-choice-models/dynamic-discrete-choice-nested-fixed-point-algorithm/
    
    Parameters
    ----------
    st_matrix : matrix
        The state transition matrix.
    state_ev_df : dataframe
        The dataframe of state's value function using the optimal coefficients.
    discounting_factor : float
        The discounting coefficient.
    df : dataframe
        The dataframe containing the data.

    Returns
    -------
    score : array
        An array of observation scores.

    """
    J = state_ev_df.d.max()+1
    dim_pi = state_ev_df.pi.max()+1
    prs = state_ev_df.pr.values
    
    df.pi=df.pi.astype(int)
    
    s = np.zeros((st_matrix.shape[0],st_matrix.shape[1]*J))
    s[:,::2] = st_matrix
    s[:,1::2] = st_matrix
    delt = discounting_factor*np.multiply(s,prs)
    
    dF = np.zeros((s.shape[0],dim_pi))
    for i in range(0,dim_pi):
        dF[:,i]= s.dot(((state_ev_df.pi==i)&(state_ev_df.d==1))*prs)
        
    dAlpha = s.dot(np.tile([1,0], int(st_matrix.shape[0]/J))*prs*state_ev_df.x)    
    dEV = np.linalg.inv(np.identity(st_matrix.shape[1]*J)-delt).dot(np.concatenate((dF,np.matrix(dAlpha).T),axis=1))
    
    score = -np.concatenate((pd.get_dummies(df.pi.values,drop_first=False),np.matrix(df.x).T),axis=1) # = dU/dTheta
    ddEV = (dEV[1::2] - dEV[::2]) # = (dEV(x,1) - dEV(x,0)) / dTheta
    
    score = score+discounting_factor*ddEV[df.x*dim_pi+df.pi]

    score = np.multiply(score,np.matrix(df.d-prs[df.x*dim_pi*J+df.pi*J+1]).T)
    return score

def get_derivatives(st_matrix, state_ev_df, discounting_factor, df):
    """
    Calculate the derivatives of likelihood to theta

    Parameters
    ----------
    st_matrix : matrix
        The state transition matrix.
    state_ev_df : dataframe
        The dataframe of state's value function using the optimal coefficients.
    discounting_factor : float
        The discounting coefficient.
    df : dataframe
        The dataframe containing the data.

    Returns
    -------
    derives : array
        The array of derivatives.

    """
    score = _get_observations_score(st_matrix, state_ev_df, discounting_factor, df)
    derives = score.sum(axis = 0)
    return derives

def get_standard_errors(st_matrix, state_ev_df, discounting_factor, df):
    """
    Calculate the standard errors of estimated coefficients

    Parameters
    ----------
    st_matrix : matrix
        The state transition matrix.
    state_ev_df : dataframe
        The dataframe of state's value function using the optimal coefficients.
    discounting_factor : float
        The discounting coefficient.
    df : dataframe
        The dataframe containing the data.

    Returns
    -------
    se : array
        An array containing the standard errors.

    """
    score = _get_observations_score(st_matrix, state_ev_df, discounting_factor, df)
    H = np.linalg.inv(score.T.dot(score))
    se = np.sqrt(np.diag(H))
    return se