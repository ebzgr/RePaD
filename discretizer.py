import numpy as np
import pandas as pd
import utility as utl
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

class DataDriveDiscretizer:
    def __init__(self, lamb = 1, delta=0.01, min_size=0, smoothing_del = 10**-5, max_pi = None):
        """
        The algorithm discretibed in XXX. This algorithm discretizes a high-dimensional variable set Q to control
        for potential biases in dynamic discrete choice modeling.

        Parameters
        ----------
        lamb : float
            The relative importance of state transition likelihood versus decision likelihood.
        delta : float
            Minumum accepted improvement in the weighted likelihood. If the increase in lower than delta the 
            algorithm stops the discretization
        min_size : int, optional
            Minimum number of observations in a partition to be considered for a split. The default is 0.
        smoothing_del : float, optional
            The value for additive smoothing. The default is 1. Check additive smoothing online for more info.
        max_pi : int, optional
            Restrict total number of partition in the discretization to max_pi. The default is None. The 
            algorithm do not restrict based on number of partitions if it is set to None.

        Returns
        -------
        None.

        """
        self.lamb = lamb
        self.delta = delta 
        self.min_size = min_size
        self.smoothing_del = smoothing_del
        if(max_pi is None):
            self.max_pi = np.inf
        else:
            self.max_pi = max_pi

    def _find_next_split(self, q_cols, total_pi, parallel=True):
        if(parallel):
            element_run = Parallel(n_jobs=-1)(delayed(_update_best_split)(self.df.copy(), self.D, total_pi, pi, q_cols, self.current_adj_ll, self.min_size, self.lamb*self.lamb_adj, self.smoothing_del) for pi, row in self.parts.iterrows())
            for i in range(len(self.parts)):
                self.parts.loc[self.parts.state==i,['next_var','next_val','next_improve','next_dec_ll','next_trans_ll']]=element_run[i]
        else:
            for pi, row in self.parts.iterrows():
                self.parts.loc[self.parts.state==pi,['next_var','next_val','next_improve','next_dec_ll','next_trans_ll']] = _update_best_split(self.df.copy(), self.D, total_pi, pi, q_cols, self.current_adj_ll, self.min_size, self.lamb*self.lamb_adj, self.smoothing_del)

        # Find the partition with the split with highest amount of likelihood improvement
        if(self.parts['next_improve'].max()>0):
            row = self.parts.iloc[self.parts.next_improve.idxmax()]
            df, parts = _add_split(self.df.copy(), self.parts.copy(), row['state'], row['next_var'], row['next_val'], total_pi)
        else:
            return self.df, self.parts, 0, 0, 0

        return df, parts, parts['next_improve'].max(), row['next_dec_ll'],row['next_trans_ll']

    def discretize(self, data, test_data=None, parallel=True):
        """
        Discretizes the state space of data using the algorithm

        Parameters
        ----------
        data : dictionary
            The data dictionary. It must contain ids, periods, X, Q and Y variables.
        test_data : dic, optional
            Test data dictionary. It is used to reporting out of sample performance. You can pass the validation
            set to your algorithm. The default is None.
        parallel : bool, optional
            Whether the algorithm should run in parallel or not. The default is True.

        Returns
        -------
        dataframe
            The generated data-driven discretization.
        dataframe
            Performance report on the training and validation set.

        """

        self.df, self.mapper, q_cols = _create_dataset(data['ids'], data['periods'], data['X'], data['Q'], data['Y'])
        self.D = np.unique(self.df.d.astype(int))
        self.parts = utl.base_split_dataframe(data['Q'])

        total_pi = 1
        counter, next_size, current_decision, current_size = _generate_counters(self.df, self.D, total_pi, self.smoothing_del)
        dec_ll =  _decision_likelihood(current_decision, current_size)
        trans_ll =  _transition_likelihood(counter, current_decision, next_size)
        self.lamb_adj = dec_ll/trans_ll
        self.current_adj_ll = dec_ll + (self.lamb_adj*self.lamb) * trans_ll
        self.score = self.current_adj_ll/(self.lamb+1)

        if(test_data is None):
            self.report = pd.DataFrame(data={'part':[1], 'dec_likelihood':[dec_ll], 'trans_likelihood':[trans_ll],
                                             'adjusted_likelihood':[self.current_adj_ll],
                                             'min_node_size':[len(data['ids'])], 'increase_ratio':[0], 'score':[self.score]})
        else:
            test_adj_ll, test_dec_ll, test_trans_ll = self.test_likelihood(test_data)
            self.test_lamb_adj = test_dec_ll/test_trans_ll
            test_adj_ll = test_dec_ll + (self.test_lamb_adj*self.lamb) * test_trans_ll
            test_score = test_adj_ll/(self.lamb+1)

            self.report = pd.DataFrame(data={'part':[1], 'dec_likelihood':[dec_ll], 'trans_likelihood':[trans_ll],
                                             'adjusted_likelihood':[self.current_adj_ll], 
                                             'min_node_size':[len(data['ids'])], 'increase_ratio':[0], 'score':[self.score],
                                             'test_dec_likelihood':[test_dec_ll], 'test_trans_likelihood':[test_trans_ll],
                                             'test_adjusted_likelihood':[test_adj_ll], 'test_increase_ratio':[0],'test_score':[test_score]})


        new_df, new_parts, improve, next_dec_ll, next_trans_ll = self._find_next_split(q_cols, total_pi, parallel=parallel)
        improve_rate = -improve/self.current_adj_ll
        print('Partition {}: Improvement = {}'.format(len(new_parts),improve_rate))
        while((improve_rate>self.delta) & (total_pi<self.max_pi)):
            self.current_adj_ll = next_dec_ll + (self.lamb_adj*self.lamb) * next_trans_ll
            self.score = self.current_adj_ll/(self.lamb+1)
            total_pi += 1
            self.df = new_df
            self.parts = new_parts
            if(test_data!=None):
                new_test_adj_ll, test_dec_ll, test_trans_ll = self.test_likelihood(test_data)
                test_score = new_test_adj_ll/(self.lamb+1)
                self.report.loc[total_pi-1] = [total_pi, next_dec_ll, next_trans_ll, self.current_adj_ll, 
                                               self.df.pi.value_counts().min(), improve_rate, self.score,
                                               test_dec_ll, test_trans_ll, new_test_adj_ll,
                                               -(new_test_adj_ll-test_adj_ll)/test_adj_ll, test_score]
                test_adj_ll = new_test_adj_ll
            else:
                self.report.loc[total_pi-1] = [total_pi, next_dec_ll, next_trans_ll, self.current_adj_ll,
                                               self.df.pi.value_counts().min(), improve_rate, self.score]

            if(total_pi==self.max_pi):
                break
            new_df, new_parts, improve, next_dec_ll, next_trans_ll = self._find_next_split(q_cols, total_pi, parallel=parallel)
            improve_rate = -improve/self.current_adj_ll

            print('Partition {}: Improvement = {}'.format(len(new_parts),improve_rate))

        parts_cols = ['state']+self.parts.columns[self.parts.columns.str.contains('q_')].tolist()
        return self.parts[parts_cols].copy(), self.report.copy()


    def test_likelihood(self, data):
        """
        Calculate the weighted likelihood of observing the data.

        Parameters
        ----------
        data : dictionary
            The data dictionary. It must contain ids, periods, X, Q and Y variables.

        Raises
        ------
        ValueError
            Error raised of we have zero probabilities in the training data and smoothing coefficient is zero.

        Returns
        -------
        adj_likelihood : float
            The adjusted likelihood.
        dec_likelihood : float
            The likelihod of decision part of data.
        trans_likelihood : float
            The likelihood of state transition part of data.

        """
        df  = _create_test_dataset(self.mapper, self.parts, data['ids'], data['periods'], data['X'], data['Q'], data['Y'])
        trans_pr, decision_pr = _get_prediction_tables(self.df, self.D, len(self.parts), self.smoothing_del)
        df = df.merge(trans_pr, how='left').merge(decision_pr, how='left')
        if((np.min(df.trans_pr_smooth)==0)&(self.lamb>0)):
            raise ValueError("A transition probability is zero. Please either set smoothing_del to a non-zero positive number or choose a bigger delta to prevent over-splitting.")
        if(np.min(df.dec_pr_smooth)==0):
            raise ValueError("A decision probability is zero. Please either set smoothing_del to a non-zero positive number or choose a bigger delta to prevent over-splitting.")

        df = df[df.t!=df.t.max()]
        dec_likelihood = np.sum(np.log(df.dec_pr_smooth))
        
        trans_likelihood = np.sum(np.log(df.trans_pr_smooth))
        if(self.lamb>0):
            adj_likelihood =  dec_likelihood + self.lamb*self.lamb_adj * trans_likelihood
        else:
            adj_likelihood =  dec_likelihood
        return adj_likelihood, dec_likelihood, trans_likelihood


    def cross_validate_discretization(self, data, folds):
        """
        Perform a cross validation on the data.

        Parameters
        ----------
        data : dictionary
            The data dictionary. It must contain ids, periods, X, Q and Y variables.
        folds : int
            Number of folds in the cross-validation.

        Returns
        -------
        summary : dataframe
            A summary report of cross-validation.
        full_report : dataframe
            The full report of partitionings.

        """
        kf = KFold(n_splits=folds, random_state=0, shuffle=True)
        i = 0
        train_data, test_data = {},{}
        cross_parts, cross_reports = {}, {}
        for train_index, test_index in kf.split(np.unique(data['ids'])):
            for x in data:
                train_data[x] = data[x][np.isin(data['ids'], train_index)]
                test_data[x] = data[x][~np.isin(data['ids'], train_index)]
            cross_parts[i], cross_reports[i] = self.partition(train_data, test_data)
            i+=1

        full_report = pd.concat(cross_reports).reset_index()
        summary = full_report.groupby('part').mean().drop(columns=['level_0','level_1'])
        return summary, full_report

def _create_dataset(ids, periods, X, Q, Y):
    """
    Transform the given data into a dataframe. It also transform the X variable into a one dimensional variable 
    to make operations easier.

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
        The dataframe of data.
    mapper : dataframe
        A mapper that map original values for X to its one dimensional transformation.
    q_cols : list
        The list of q column names in the dataframe df.

    """
    x_transform, mapper = utl.map_to_single_dimension(X)
    q_cols = ['q_{}'.format(i) for i in range(0,Q.shape[1])]
    df = pd.DataFrame(data={'id':ids, 't':periods, 'd':Y, 'x':x_transform['x'].values})
    df = pd.concat([df,pd.DataFrame(data=Q, columns = q_cols)],axis=1)
    df['pi'] = 0
    df = utl.add_next_state(df)

    return df, mapper, q_cols

def _create_test_dataset(mapper, parts, ids, periods, X, Q, Y):
    """
    Create the test dataset using mapper and test data.

    Parameters
    ----------
    mapper : dataframe
        A mapper that map original values for X to its one dimensional transformation.
    parts : dataframe
        The state space discretization.
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
        The dataframe of data.

    """
    x_cols = ['orig_x{}'.format(i) for i in range(0,X.shape[1])]
    q_cols = ['q_{}'.format(i) for i in range(0,Q.shape[1])]
    x_transform = pd.DataFrame(data=X, columns = x_cols)
    x_transform = x_transform.merge(mapper, on=x_cols, how='left')
    df = pd.DataFrame(data={'id':ids, 't':periods, 'd':Y, 'x':x_transform['x'].values})
    df = pd.concat([df,pd.DataFrame(data=Q,columns = q_cols)],axis=1)
    df['pi'] = utl.q_to_pi_states(parts, Q, Q.shape[1])
    df = utl.add_next_state(df)
    return df

def _get_prediction_tables(df, D, total_pi, smoothing_del):
    """
    Given a dataframe, returns the probability of state transition and decisions
    
    Parameters
    ----------
    df : dataframe
        The dataframe of data.
    D : array
        An array of different possible values for dependent variable.
    total_pi : int
        Total number of partitions in the discretization.
    smoothing_del : float
        The value for additive smoothing. The default is 1. Check additive smoothing online for more info.

    Returns
    -------
    counter : dataframe
        A dataframe containing state transition probabilities, and their smoothed version using additive smoothing.
    current_decision : dataframe
        A dataframe containing decision probabilities, and their smoothed version using additive smoothing.

    """
    counter, next_size, current_decision, current_size = _generate_counters(df, D, total_pi, smoothing_del)
    counter = counter.merge(next_size, how='left').merge(current_decision, how='left')
    counter['trans_pr'] = counter['move_count'] /counter['next_count'] /counter['current_dec_count']
    counter['trans_pr_smooth'] = counter['move_count_smooth'] /counter['next_count_smooth'] /counter['current_dec_count_smooth']
    current_decision = current_decision.merge(current_size, how='left')
    current_decision['dec_pr']=current_decision['current_dec_count'] / current_decision['current_count']
    current_decision['dec_pr_smooth']=current_decision['current_dec_count_smooth'] / current_decision['current_count_smooth']

    return counter, current_decision


def _transition_likelihood(counter, current_decision, next_size):
    """
    Calculate State Transition Likelihood Part using N(x',pi',d',x,pi), N(d,x,pi) and N(x',pi')

    Parameters
    ----------
    counter : dataframe
        A dataframe containing N(x',pi',d',x,pi).
    current_decision : dataframe
        A dataframe containing N(d,x,pi).
    next_size : dataframe
        A dataframe containing N(x',pi').

    Returns
    -------
    flaot
        The likelihood of state transition part of data.

    """
    nom = np.sum(counter.move_count*np.log(counter.move_count_smooth))
    den1 = np.sum(current_decision.current_dec_count*np.log(current_decision.current_dec_count_smooth)) 
    den2 = np.sum(next_size.next_count*np.log(next_size.next_count_smooth))
    return nom - den1 - den2



def _decision_likelihood(current_decision, current_size):
    """
    Calculate Decision Likelihood Part using N(d,x,pi) and N(x,pi)

    Parameters
    ----------
    current_decision : dataframe
        A dataframe containing N(d,x,pi).
    current_size : dataframe
        A dataframe containing N(x,pi).

    Returns
    -------
    flaot
        The likelihood of decision part of data.

    """
    nom = np.sum(current_decision.current_dec_count*np.log(current_decision.current_dec_count_smooth))
    den = np.sum(current_size.current_count*np.log(current_size.current_count_smooth))
    return nom - den



def _generate_counters_base(df, D, total_pi):
    """
    Generate an empty counter dataframe where N(.) = 0 for any combination of X and PI(Q)

    Parameters
    ----------
    df : dataframe
        The dataframe of data.
    D : array
        An array of different possible values for dependent variable.
    total_pi : int
        Total number of partitions in the discretization.

    Returns
    -------
    dataframe
        The empty dataframe.

    """
    x_df = pd.DataFrame(data={'x':df.x.unique(),'move_count':0})
    next_x_df = pd.DataFrame(data={'next_x':df.x.unique(),'move_count':0})
    D_df = pd.DataFrame(data={'d':D,'move_count':0})
    pi_df = pd.DataFrame(data={'pi':np.arange(total_pi),'move_count':0})
    next_pi_df = pd.DataFrame(data={'next_pi':np.arange(total_pi),'move_count':0})
    counter_base = x_df.merge(pi_df.merge(D_df.merge(next_x_df.merge(next_pi_df))))
    return counter_base.drop(columns=['move_count'])


def _generate_counters(df, D, total_pi, smoothing_del):
    """
    Get the empty counter dataframe and populate it. The function add smoothing value to remove any zero probability

    Parameters
    ----------
    df : dataframe
        The dataframe of data.
    D : array
        An array of different possible values for dependent variable.
    total_pi : int
        Total number of partitions in the discretization.
    smoothing_del : float
        The value for additive smoothing. The default is 1. Check additive smoothing online for more info.

    Returns
    -------
    counter : dataframe
        A dataframe containing N(x',pi',d',x,pi).
    next_size : dataframe
        A dataframe containing N(x',pi').
    current_decision : dataframe
        A dataframe containing N(d,x,pi).
    current_size : dataframe
        A dataframe containing N(x,pi).

    """
    counter_base = _generate_counters_base(df, D, total_pi)
    total = df.groupby(['x','pi','next_x','next_pi','d'])['id'].count().reset_index().rename(columns={'id':'move_count'}).astype('int64')
    counter = counter_base.merge(total, how='left').fillna(0).astype("int64")
    counter['move_count_smooth'] = counter['move_count'] + smoothing_del

    next_size = counter.groupby(['next_x','next_pi'])[['move_count','move_count_smooth']].sum().reset_index().rename(columns={'move_count':'next_count','move_count_smooth':'next_count_smooth'})
    current_decision = counter.groupby(['x','pi','d'])[['move_count','move_count_smooth']].sum().reset_index().rename(columns={'move_count':'current_dec_count','move_count_smooth':'current_dec_count_smooth'})
    current_size = current_decision.groupby(['x','pi'])[['current_dec_count','current_dec_count_smooth']].sum().reset_index().rename(columns={'current_dec_count':'current_count', 'current_dec_count_smooth':'current_count_smooth'})
    return counter, next_size, current_decision, current_size


# 
def _check_split(df, D, total_pi, pi, col, val, current_ll, min_size, lamb, smoothing_del):
    """
    Check the likelihood increase for the split in "pi" on "col" at "val"

    Parameters
    ----------
    df : dataframe
        The dataframe of data.
    D : array
        An array of different possible values for dependent variable.
    total_pi : int
        Total number of partitions in the discretization.
    pi : int
        Selected partition for split.
    col : int
        Selected variable for split.
    val : float
        Selected value for split on col.
    current_ll : float
        Current likelihood of observing the data.
    min_size : int, optional
        Minimum number of observations in a partition to be considered for a split. The default is 0.
    lamb : float
        The relative importance of state transition likelihood versus decision likelihood. The lamb 
        here is equal to relative_lamb*adjusted_lamb.
    smoothing_del : float, optional
        The value for additive smoothing. The default is 1. Check additive smoothing online for more info.

    Returns
    -------
    float
        The adjusted likelihood after the split.
    float
        The likelihood of the decision part after the split.
    float
        The likelihood of the state transition part after the split..

    """
    df.loc[(df.pi == pi) & (df[col]>=val), "pi"] = total_pi

    if(min(len(df[df.pi==total_pi]),len(df[df.pi==pi]))<min_size):
        return current_ll, 0, 0
    df['next_pi'] = df.pi.shift(-1)
    df.loc[df.t==df.t.max(),'next_pi'] = np.nan
    counter, next_size, current_decision, current_size = _generate_counters(df, D, total_pi+1, smoothing_del)

    dec_ll =  _decision_likelihood(current_decision, current_size)
    trans_ll =  _transition_likelihood(counter, current_decision, next_size)
    new_ll = dec_ll + lamb * trans_ll

    return new_ll, dec_ll, trans_ll


def _update_best_split(df, D, total_pi, pi ,q_cols, current_ll, min_size, lamb, smoothing_del):
    """
    Find the best next split in partition pi given the current partitioning

    Parameters
    ----------
    df : dataframe
        The dataframe of data.
    D : array
        An array of different possible values for dependent variable.
    total_pi : int
        Total number of partitions in the discretization.
    pi : int
        Selected partition for split.
    q_cols : array
        List of q column names in dataframe.
    current_ll : float
        Current likelihood of observing the data.
    min_size : int, optional
        Minimum number of observations in a partition to be considered for a split. The default is 0.
    lamb : float
        The relative importance of state transition likelihood versus decision likelihood. The lamb 
        here is equal to relative_lamb*adjusted_lamb.
    smoothing_del : float, optional
        The value for additive smoothing. The default is 1. Check additive smoothing online for more info.


    Returns
    -------
    best_var : int
        Best variable in Q for next split in pi.
    best_val : float
        Best value for next split in pi.
    likelihood_delta : float
        Likelihood increase after the split.
    best_dec_ll : float
        The likelihood of the decision part of data after 0the best split in pi.
    best_trans_ll : float
        The likelihood of the state transition part of data after the best split in pi.

    """
    best_ll, best_dec_ll, best_trans_ll, best_var, best_val = current_ll, 0, 0, 0, 0

    part_df = df[(df.pi == pi)&(df.t!=df.t.max())]
    for col in q_cols:
        vals = part_df[col].unique().astype("int64")
        vals.sort()
        for val in vals[1:]:
            newll, dec_ll, trans_ll = _check_split(df.copy(), D, total_pi, pi, col, val, current_ll, min_size, lamb, smoothing_del)
            if(newll>best_ll):
                best_ll = newll
                best_dec_ll = dec_ll
                best_trans_ll = trans_ll
                best_var = col
                best_val = val

    likelihood_delta = best_ll-current_ll
    return best_var, best_val, likelihood_delta, best_dec_ll, best_trans_ll

def _add_split(df, parts, pi, col, val, total_pi):
    """
    Add split pi,col,val to the current partitioning and updates df and parts dataframes

    Parameters
    ----------
    df : dataframe
        The dataframe of data.
    parts : dataframe
        The dataframe of discretization.
    pi : int
        The partition that is to be split.
    col : int
        The variable in Q that is used for the split.
    val : float
        The value for splitting pi by col.
    total_pi : int
        Total number of partitions in the discretization..

    Returns
    -------
    df : dataframe
        The updated data dataframe.
    parts : dataframe
        The updated discretization dataframe.

    """

    df.loc[(df.pi == pi) & (df[col]>=val), "pi"] = total_pi
    df['next_pi'] = df.pi.shift(-1)
    df.loc[df.t==df.t.max(),'next_pi'] = np.nan
    parts.loc[total_pi] = parts.loc[pi].copy()
    parts.loc[total_pi,'state']=total_pi
    parts.loc[parts.state==pi,"{}_max".format(col)] = val
    parts.loc[parts.state==total_pi,"{}_min".format(col)] = val

    return df, parts
