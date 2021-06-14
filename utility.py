import numpy as np
import pandas as pd

def get_partitioning_variables(df):
    """
    Convert the return dataframe from data generator to data dictionary that is compatible with discretizer

    Parameters
    ----------
    df : dataframe
        The data dataframe.

    Returns
    -------
    dict
        The data dictionary.

    """
    df=df.rename(columns={'m':'x0'})
    ids = df.id.values
    periods = df.t.values
    X = df[['x0']].values
    Q = df[df.columns[df.columns.str.contains('q_')]].values
    Y = df.d.values
    return {'ids':ids, 'periods':periods, 'X':X, 'Q':Q, 'Y':Y}

def create_dataframe_from_data(ids, periods, X, Q, Y):
    """
    Create a Dataframe of the observations where for each column i in X and Q we have xi and qi in the dataframe

    Parameters
    ----------
    ids : array
        An array containing observation's id.
    periods : array
        An array containing observation's period.
    X : array
        An array containing observation's X variable set.
    Q : array
        An array containing observation's Q variable set.
    Y : array
        An array containing observation's dependent variable.

    Returns
    -------
    df : dataframe
        The data dataframe.

    """
    df = pd.DataFrame(data={'id':ids, 't':periods, 'd':Y})
    for i in range(0,X.shape[1]):
        df['x{}'.format(i)] = X[:,i]
    for i in np.arange(0,Q.shape[1]):
        df['q{}'.format(i)] = Q[:,i]
    return df


def generate_base_state_dataframe(dim_pi, max_x, J):
    """
    Generate base state dataframe. It contains the interaction of all decisions in all the states.

    Parameters
    ----------
    dim_pi : int
        The total number of partitions in the discretization.
    max_x : int
        The maximum value for x variables.        
    J : int
        The number of possible values for dependent variable.

    Returns
    -------
    data : dataframe
        The base dataframe that contains all the states and decisions.

    """
    dt = pd.DataFrame(data={'pi':np.arange(0,dim_pi),'tmp':1})
    mileage = pd.DataFrame(data={'x':np.arange(0,max_x),'tmp':1})
    des = pd.DataFrame(data={'d':np.arange(0,J),'tmp':1})
    data = mileage.merge(dt.merge(des)).drop('tmp',1)
    return data

def calculate_ll(dHats, d): 
    """
    Calculate the log likelihood of observed decisions using estimated decision probabilities

    Parameters
    ----------
    dHats : array
        Estimated decision probabilities.
    d : array
        Actual decision probabilities.

    Returns
    -------
    float
        The likelihood.

    """
    return (np.log(dHats+1e-200)*d + np.log(1-dHats+1e-200)*(1-d)).sum()


def hot_encode(x):
    """
    Returns the hotencoding of a categorical variable

    Parameters
    ----------
    x : array
        Categorical variable values.

    Returns
    -------
    matrix
        The hotencoding matrix.

    """
    return pd.get_dummies(x).values

def q_to_pi_states(discretization, q, dim_q):
    """
    Pi function: Zero based transformation from Q to Pi. Returns the partition of observations given their q, 
    and the discretization dataframe
    
    Parameters
    ----------
    discretization : dataframe
        DESCRIPTION.
    q : matrix
        The q values of observations.
    dim_q : int
        dimension of q.

    Returns
    -------
    array
        An array of observations partition in discretization.

    """
    qs = pd.DataFrame(data=q)
    qs['states'] = 0
    for i in range(0,len(discretization)):
        row = discretization[discretization.state == i].iloc[0]
        selects = np.arange(0,len(qs))
        for j in range(0,dim_q):
            df = qs.iloc[selects]
            selects = df[(df[j]<row['q_{}_max'.format(j)])&(df[j]>=row['q_{}_min'.format(j)])].index.values
        qs.loc[selects,'states'] = i

    return qs.states.values

 
def mileage_transition(mileage, replace, max_m, increment=None):
    """
    Given observations mileage, replacement transition, and mileage increment amount, this function returns 
    the next value for mileage

    Parameters
    ----------
    mileage : array
        Array of observations mileages.
    replace : array
        The boolean values for replacement or maintenance. replace[i] = 1 means that the replacement decision is chosen
    max_m : int
        Maximum mileage that a bus can have. After that the maintenance decision does not increase the mileage.
    increment : Array, optional
        The amount of mileage increment value in a period. The default is None. If the default value is chosen, 
        the mileage increases is 1.

    Returns
    -------
    mileage : array
        The new mileage values.

    """
    if(increment is None):
        mileage = np.minimum(mileage + 1, max_m-1)
        mileage[replace==1]=1
    else:
        mileage = np.minimum(mileage + increment, max_m-1)
        mileage[replace==1]=increment[replace==1]
        
    return mileage

def pi_state_transition(old_states, transition):
    """
    Given transition matrix in PI space, transit a given old_states in PI to new_states in PI    

    Parameters
    ----------
    old_states : array
        Old state in the discretization.
    transition : array
        The state transition in the discretized space.

    Returns
    -------
    new_states : array
        The new states in the discretization.

    """
    new_states = 0
    for i in range(0,transition.shape[0]):
        new_states = new_states + (old_states==i)*np.random.choice(transition.shape[0],size=len(old_states),p=np.squeeze(np.asarray(transition[i])))
    return new_states

def base_split_dataframe(Q):
    """
    Generate the base discretization dataframe with only one partition that contains all Q

    Parameters
    ----------
    Q : array
        The Q part of data.

    Returns
    -------
    discretization : dataframe
        The base discretization dataframe.

    """
    discretization = pd.DataFrame(data={'state':[0],'next_var':[0],'next_val':[0],'next_improve':[0], 'next_dec_ll':[0], 'next_trans_ll':[0]})

    for i in range(0,Q.shape[1]):
        discretization['q_'+str(i)+'_min'] = Q[:,i].min()
        discretization['q_'+str(i)+'_max'] = Q[:,i].max()+1

    return discretization

def generate_initial_q(n,dim_q,max_q):
    """
    Randomly generate an initial values for observations in Q space

    Parameters
    ----------
    n : int
        Number of observations.
    dim_q : int
        Dimension of Q.
    max_x : TYPE
        Maximum value for q.

    Returns
    -------
    matrix
        The matrix of observations values in Q.

    """
    return np.random.randint(low=0, high=max_q,size=n*dim_q).reshape(n,dim_q)

 
def generate_pi_transition(dim, ttype, ttype_ext=2):
    """
    Generate a PI(Q) transition function given the dimension of PI(Q) and the type of transition

    Parameters
    ----------
    dim : int
        dimension of pi.
    ttype : int
        type of transition. It can be fully random, partial, and no transition. See the reference paper for more info.
    ttype_ext : int, optional
        The number of next state and observations can go given a state. It is only used when ttype is 3. The default is 2.

    Returns
    -------
    matrix
        The state transition in Q matrix.

    """
    if(ttype==1):
        return np.identity(dim)

    if(ttype==2):
        return np.ones((dim, dim))/dim

    if(ttype==3):
        res = np.identity(dim)
        for i in range(1, ttype_ext):
            res = res + np.roll(np.identity(dim), i, axis=1)
        return res/ttype_ext


def map_to_single_dimension(X):
    """
    Generate a mapping function f(X)->one dimension. It returns the new dimension of Xs as well as the Mapper

    Parameters
    ----------
    X : array
        The array of X independent variables.

    Returns
    -------
    x_df : dataframe
        The dataframe of X and its one dimensional.
    mapper : dataframe
        The mapping dataframe.

    """
    x_cols = ['orig_x{}'.format(i) for i in range(0,X.shape[1])]
    x_df = pd.DataFrame(data=X, columns = x_cols)
    mapper = x_df.drop_duplicates().copy()
    mapper['x'] = np.arange(len(mapper))
    x_df = x_df.merge(mapper, on=x_cols, how='left')
    return x_df, mapper


def add_next_state(df):
    """
    Add the next state to the dataframe

    Parameters
    ----------
    df : dataframe
        The dataframe of data.

    Returns
    -------
    df : dataframe
        The dataframe of data containing next state.

    """
    df = df.sort_values(['id','t'])
    df['next_x'] = df.x.shift(-1)
    df['next_pi'] = df.pi.shift(-1)
    df.loc[df.t==df.t.max(),['next_x','next_pi']] = [np.nan,np.nan]
    return df