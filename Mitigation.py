from itertools import repeat
import numpy as np
import scipy.optimize as opt
import pandas as pd
import pickle
from collections import namedtuple
from scipy.stats import norm
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import torch
import multiprocessing as mp
#from __future__ import print_function
import functools
import numpy as np
import pandas as pd
import time

import pickle


# Global Variables
TEST_SIZE = 0.5  # fraction of observations from each protected group
Theta = np.linspace(0, 1.0, 101)
alpha = (Theta[1] - Theta[0])/2

_SMALL = True  # small scale dataset for speed and testing




# A multiplier controlling the automatic setting of nu.
_ACCURACY_MUL = 0.5

# Parameters controlling adaptive shrinking of the learning rate.
_REGR_CHECK_START_T = 5
_REGR_CHECK_INCREASE_T = 1.6
_SHRINK_REGRET = 0.8
_SHRINK_ETA = 0.8


_LOGISTIC_C = 5  # Constant for rescaled logisitic loss
_QEO_EVAL = False  # For now not handling the QEO disparity

# The smallest number of iterations after which expgrad terminates.
_MIN_T = 0

# If _RUN_LP_STEP is set to True, then each step of exponentiated gradient is
# followed by the saddle point optimization over the convex hull of
# classifiers returned so far.
_RUN_LP_STEP = True


_PRECISION = 1e-12


def read_result_list(result_list):
    """
    Parse the experiment a list of experiment result and print out info
    """

    for result in result_list:
        learner = result['learner']
        dataset = result['dataset']
        train_eval = result['train_eval']
        test_eval = result['test_eval']
        loss = result['loss']
        constraint = result['constraint']
        learner = result['learner']
        dataset = result['dataset']
        eps_vals = train_eval.keys()
        train_disp_dic = {}
        test_disp_dic = {}
        train_err_dic = {}
        test_err_dic = {}
        test_loss_std_dic = {}
        test_disp_dev_dic = {}
        for eps in eps_vals:
            train_disp = train_eval[eps]["DP_disp"]
            test_disp = test_eval[eps]["DP_disp"]
            train_disp_dic[eps] = train_disp
            test_disp_dic[eps] = test_disp
            test_loss_std_dic[eps] = test_eval[eps]['loss_std']
            test_disp_dev_dic[eps] = test_eval[eps]['disp_std']

            if loss == "square":
                # taking the RMSE
                train_err_dic[eps] = np.sqrt(train_eval[eps]['weighted_loss'])
                test_err_dic[eps] = np.sqrt(test_eval[eps]['weighted_loss'])

            else:
                train_err_dic[eps] = (train_eval[eps]['weighted_loss'])
                test_err_dic[eps] = (test_eval[eps]['weighted_loss'])

        # taking the pareto frontier
        train_disp_list = [train_disp_dic[k] for k in eps_vals]
        test_disp_list = [test_disp_dic[k] for k in eps_vals]
        train_err_list = [train_err_dic[k] for k in eps_vals]
        test_err_list = [test_err_dic[k] for k in eps_vals]

        if loss == "square":
            show_loss = 'RMSE'
        else:
            show_loss = loss


        info = str('Dataset: '+dataset + '; loss: ' + loss + '; Solver: '+ learner)
        print(info)

        train_data = {'specified epsilon': list(eps_vals), 'SP disparity':
                      train_disp_list, show_loss : train_err_list}
        train_performance = pd.DataFrame(data=train_data)
        test_data = {'specified epsilon': list(eps_vals), 'SP disparity':
                      test_disp_list, show_loss : test_err_list}
        test_performance = pd.DataFrame(data=test_data)

        # Print out experiment info.
        print('Train set trade-off:')
        print(train_performance)
        print('Test set trade-off:')
        print(test_performance)


def pmf2disp(pmf1, pmf2):
    """
    Take two empirical PMF vectors with the same support and calculate
    the K-S stats
    """
    cdf_1 = pmf1.cumsum()
    cdf_2 = pmf2.cumsum()
    diff = cdf_1 - cdf_2
    diff = abs(diff)
    return max(diff)

def get_histogram(pred, theta_indices):
    """
    Given a list of discrete predictions and Theta, compute a histogram
    pred: discrete prediction Series vector
    Theta: the discrete range of predictions as a Series vector
    """
    theta_counts = pd.Series(np.zeros(len(theta_indices)))
    for theta in theta_indices:
        theta_counts[theta_indices == theta] = len(pred[pred == theta])
    return theta_counts

def weighted_pmf(pred, classifier_weights, Theta):
    """
    Given a list of predictions and a set of weights, compute pmf.
    pl: a list of prediction vectors
    result_weights: a vector of weights over the classifiers
    """
    width = Theta[1] - Theta[0]
    theta_indices = pd.Series(Theta + width/2)
    weights = list(classifier_weights)
    weighted_histograms = [(get_histogram(pred.iloc[:, i],
                                          theta_indices)) * weights[i]
                           for i in range(pred.shape[1])]

    theta_counts = sum(weighted_histograms)
    pmf = theta_counts / sum(theta_counts)
    return pmf

def loss_vec(tp, y, result_weights, loss='square'):
    """
    Given a list of predictions and a set of weights, compute
    (weighted average) loss for each point
    """
    
    num_h = len(result_weights)
    if loss == 'square':
        loss_list = [(tp.iloc[:, i] - y)**2 for i in range(num_h)]
    elif loss == 'absolute':
        loss_list = [abs(tp.iloc[:, i] - y) for i in range(num_h)]
    elif loss == 'logistic':
        logistic_prob_list = [1/(1 + np.exp(- _LOGISTIC_C * (2 * tp[i]
                                                             - 1))) for i in range(num_h)]
        # logistic_prob_list = [tp[i] for i in range(num_h)]
        loss_list = [log_loss_vec(y, prob_pred, eps=1e-15) for
                     prob_pred in logistic_prob_list]
    else:
        raise Exception('Loss not supported: ', str(loss))
    df = pd.concat(loss_list, axis=1)
    weighted_loss_vec = pd.DataFrame(np.dot(df,
                                            pd.DataFrame(result_weights)))
    return weighted_loss_vec.iloc[:, 0]

def log_loss_vec(y_true, y_pred, eps=1e-15):
    """
    return the vector of log loss over the examples
    """
    # Clipping
    y_pred = np.clip(y_pred, eps, 1 - eps)
    # If y_pred is of single dimension, assume y_true to be binary
    # and then check.
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    if y_pred.shape[1] == 1:
        y_pred = np.append(1 - y_pred, y_pred, axis=1)

    # Renormalize
    y_pred /= y_pred.sum(axis=1)[:, np.newaxis]
    trans_label = pd.concat([1-y_true, y_true], axis=1)
    loss = -(trans_label * np.log(y_pred)).sum(axis=1)
    return loss


def KS_confbdd(n, alpha=0.05):
    """
    Given sample size calculate the confidence interval width on K-S stats
    n: sample size
    alpha: failure prob
    ref: http://www.math.utah.edu/~davar/ps-pdf-files/Kolmogorov-Smirnov.pdf
    """
    return np.sqrt((1/(2 * n)) * np.log(2/alpha))

def extract_group_pred(total_pred, a):
    """
    total_pred: predictions over the data
    a: protected group attributes
    extract the relevant predictions for each protected group
    """
    groups = list(pd.Series.unique(a))
    pred_per_group = {}
    for g in groups:
        pred_per_group[g] = total_pred[a == g]
    return pred_per_group

def extract_pred(X, pred_aug, Theta):
    """
    Given a list of pred over the augmented dataset, produce
    the real-valued predictions over the original dataset
    """
    width = Theta[1] - Theta[0]
    Theta_mid = Theta + (width / 2)

    num_t = len(Theta)
    n = int(len(X) / num_t)  # TODO: check whether things divide
    pred_list = [pred_aug[((j) * n):((j+1) * n)] for j in range(num_t)]
    total_pred_list = []
    for i in range(n):
        theta_index = max(0, (sum([p_vec.iloc[i] for p_vec in pred_list]) - 1))
        total_pred_list.append(Theta_mid[theta_index])
    return total_pred_list

data_list = []
loss_vec_list = []
def evaluate_FairModel(x, a, y, loss, result, Theta):
    """
    Evaluate the performance of the fair model on a dataset

    Input:
    - X, Y: augmented data
    - loss: loss function name
    - result returned by exp_grad
    - Theta: list of thresholds
    - y: original labels
    """

    if loss == "square":  # squared loss reweighting
        X, A, Y, W = augment_data_sq(x, a, y, Theta)
   # elif loss == "absolute":  # absolute loss reweighting (uniform)
    #    X, A, Y, W = augment.augment_data_ab(x, a, y, Theta)
    #elif loss == "logistic":  # logisitic reweighting
     #   X, A, Y, W = augment.augment_data_logistic(x, a, y, Theta)
    else:
        raise Exception('Loss not supported: ', str(loss))

    hs = result.hs
    weights = result.weights

    # first make sure the lengths of hs and weights are the same;
    off_set = len(hs) - len(weights)
    if (off_set > 0):
        off_set_list = pd.Series(np.zeros(off_set), index=[i +
                                                           len(weights)
                                                           for i in
                                                           range(off_set)])
        result_weights = weights.append(off_set_list)
    else:
        result_weights = weights

    # second filter out hypotheses with zero weights
    hs = hs[result_weights > 0]
    result_weights = result_weights[result_weights > 0]

    num_h = len(hs)
    num_t = len(Theta)
    n = int(len(X) / num_t)

    # predictions
    pred_list = [pd.Series(extract_pred(X, h(X), Theta),
                           index=range(n)) for h in hs]
    total_pred = pd.concat(pred_list, axis=1, keys=range(num_h))

    # predictions across different groups
    pred_group = extract_group_pred(total_pred, a)


    data_list.append([pred_list,total_pred,y])
  
    loss_vec_list.append([total_pred, y, result_weights, loss])
    weighted_loss_vec = loss_vec(total_pred, y, result_weights, loss)

    # Fit a normal distribution to the sq_loss vector
    loss_mean, loss_std = norm.fit(weighted_loss_vec)

    # DP disp
    PMF_all = weighted_pmf(total_pred, result_weights, Theta)
    PMF_group = [weighted_pmf(pred_group[g], result_weights, Theta) for g in pred_group]
    DP_disp = max([pmf2disp(PMF_g, PMF_all) for PMF_g in PMF_group])

   
    # TODO: make sure at least one for each subgroup
    evaluation = {}
    evaluation['pred'] = total_pred
    evaluation['classifier_weights'] = result_weights
    evaluation['weighted_loss'] = loss_mean
    evaluation['loss_std'] = loss_std / np.sqrt(n)
    evaluation['disp_std'] = KS_confbdd(n, alpha=0.05)
    evaluation['DP_disp'] = DP_disp
    evaluation['n_oracle_calls'] = result.n_oracle_calls

    return evaluation

class Moment:
    """Generic moment"""
    
    def __init__(self, dataX, dataA, dataY):
        self.X = dataX
        self.tags = pd.DataFrame({"attr": dataA, "label": dataY})
        self.n = dataX.shape[0]

class _CondOpportunity(Moment):
    """Generic fairness metric including DP and EO"""

    def __init__(self, dataX, dataA, dataY, dataGrp):
        super().__init__(dataX, dataA, dataY)
        self.tags["grp"] = dataGrp
        self.prob_grp = self.tags.groupby("grp").size()/self.n
        self.prob_attr_grp = self.tags.groupby(["grp", "attr"]).size()/self.n
        signed = pd.concat([self.prob_attr_grp, self.prob_attr_grp],
                           keys=["+", "-"],
                           names=["sign", "grp", "attr"])
        
        self.index = signed.index
        
    def gamma(self, predictor):
        pred = predictor(self.X)
        self.tags["pred"] = pred
        expect_grp = self.tags.groupby("grp").mean()
        expect_attr_grp = self.tags.groupby(["grp", "attr"]).mean()
        expect_attr_grp["diff"] = expect_attr_grp["pred"] - expect_grp["pred"]
        g_unsigned = expect_attr_grp["diff"]
        g_signed = pd.concat([g_unsigned, -g_unsigned],
                             keys=["+","-"],
                             names=["sign", "grp", "attr"])
        self._gamma_descr = str(expect_attr_grp[["pred", "diff"]])
        return g_signed

    def signed_weights(self, lambda_vec):
        lambda_signed = lambda_vec["+"] - lambda_vec["-"]
        adjust = lambda_signed.sum(level="grp")/self.prob_grp \
                 - lambda_signed/self.prob_attr_grp
        signed_weights = self.tags.apply(
            lambda row: adjust[row["grp"], row["attr"]], axis=1
        )
        return signed_weights
    
class DP(_CondOpportunity):
    """Demographic parity"""
    short_name = "DP"

    def __init__(self, dataX, dataA, dataY):
        super().__init__(dataX, dataA, dataY,
                         dataY.apply(lambda y : "all"))

       

class MisclassError(Moment):
    """Misclassification error"""
    short_name = "Err"

    def __init__(self, dataX, dataA, dataY, dataW=None):
        super().__init__(dataX, dataA, dataY)
        if dataW is None:
            self.tags["weight"] = 1
        else:
            self.tags["weight"] = dataW
        self.index = ["all"]

    def gamma(self, predictor):
        pred = predictor(self.X)
        error = pd.Series(data=(self.tags["weight"]*(self.tags["label"]-pred).abs()).mean(),
                          index=self.index)
        self._gamma_descr = str(error)
        return error

    def signed_weights(self, lambda_vec=None):
        if lambda_vec is None:
            return self.tags["weight"]*(2*self.tags["label"]-1)
        else:
            return lambda_vec["all"]*self.tags["weight"]*(2*self.tags["label"]-1)



class _GapResult:
    # The result of a duality gap computation
    def __init__(self, L, L_low, L_high, gamma, error):
        self.L = L
        self.L_low = L_low
        self.L_high = L_high
        self.gamma = gamma
        self.error = error

    def gap(self):
        return max(self.L-self.L_low, self.L_high-self.L)


class _Lagrangian:
    # Operations related to the Lagrangian
    def __init__(self, dataX, dataA, dataY, learner, dataW, cons_class, eps, B,
                 opt_lambda=True, debug=False, init_cache=[]):
        self.X = dataX
        self.obj = MisclassError(dataX, dataA, dataY, dataW)
        self.cons = cons_class(dataX, dataA, dataY)
        self.pickled_learner = pickle.dumps(learner)
        self.eps = eps
        self.B = B
        self.opt_lambda = opt_lambda
        self.debug = debug
        self.hs = pd.Series()
        self.classifiers = pd.Series()
        self.errors = pd.Series()
        self.gammas = pd.DataFrame()
        self.n = self.X.shape[0]
        self.n_oracle_calls = 0
        self.last_linprog_n_hs = 0
        self.last_linprog_result = None
        for classifier in init_cache:
            self.add_classifier(classifier)
        
    def eval_from_error_gamma(self, error, gamma, lambda_vec):
        # Return the value of the Lagrangian.
        #
        # Returned values:
        #   L -- value of the Lagrangian
        #   L_high -- value of the Lagrangian under the best response of the
        #             lambda player
        
        lambda_signed = lambda_vec["+"] - lambda_vec["-"]
        if self.opt_lambda:
            L = error + np.sum(lambda_vec*gamma) \
                - self.eps*np.sum(lambda_signed.abs())
        else:
            L = error + np.sum(lambda_vec*gamma) \
                - self.eps*np.sum(lambda_vec)
        max_gamma = gamma.max()
        if max_gamma < self.eps:
            L_high = error
        else:
            L_high = error + self.B*(max_gamma-self.eps)
        return L, L_high
    
    def eval(self, h, lambda_vec):
        # Return the value of the Lagrangian.
        #
        # Returned values:
        #   L -- value of the Lagrangian
        #   L_high -- value of the Lagrangian under the best response of the
        #             lambda player
        #   gamma -- vector of constraint violations
        #   error -- the empirical error
        
        if callable(h):
            error = self.obj.gamma(h)[0]
            gamma = self.cons.gamma(h)
        else:
            error = self.errors[h.index].dot(h)
            gamma = self.gammas[h.index].dot(h)
        L, L_high = self.eval_from_error_gamma(error, gamma, lambda_vec)
        return L, L_high, gamma, error

    def eval_gap(self, h, lambda_hat, nu):
        # Return the duality gap object for the given h and lambda_hat
        
        L, L_high, gamma, error \
            = self.eval(h, lambda_hat)
        res = _GapResult(L, L, L_high, gamma, error)
        for mul in [1.0, 2.0, 5.0, 10.0]:
            h_hat, h_hat_idx = self.best_h(mul*lambda_hat)
            if self.debug:
                print("%smul=%.0f" % (" "*9, mul))
            L_low_mul, tmp, tmp, tmp \
                = self.eval(pd.Series({h_hat_idx: 1.0}), lambda_hat)
            if (L_low_mul < res.L_low):
                res.L_low = L_low_mul
            if res.gap() > nu+_PRECISION:
                break
        return res
    
    def solve_linprog(self, nu):
        n_hs = len(self.hs)
        n_cons = len(self.cons.index)
        if self.last_linprog_n_hs == n_hs:
            return self.last_linprog_res
        c = np.concatenate((self.errors, [self.B]))
        A_ub = np.concatenate(
            (self.gammas-self.eps, -np.ones((n_cons, 1))), axis=1)
        b_ub = np.zeros(n_cons)
        A_eq = np.concatenate(
            (np.ones((1, n_hs)), np.zeros((1, 1))), axis=1)
        b_eq = np.ones(1)
        res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
        h = pd.Series(res.x[:-1], self.hs.index)
        dual_c = np.concatenate((b_ub, -b_eq))
        dual_A_ub = np.concatenate(
            (-A_ub.transpose(), A_eq.transpose()), axis=1)
        dual_b_ub = c
        dual_bounds = [
            (None, None) if i==n_cons else (0, None) for i in range(n_cons+1)]
        res_dual = opt.linprog(dual_c, A_ub=dual_A_ub, b_ub=dual_b_ub,
                               bounds=dual_bounds)
        lambda_vec = pd.Series(res_dual.x[:-1], self.cons.index)
        self.last_linprog_n_hs = n_hs
        self.last_linprog_res = (h, lambda_vec,
                                 self.eval_gap(h, lambda_vec, nu))
        return self.last_linprog_res

    def best_h(self, lambda_vec):
        # Return the classifier that solves the best-response problem
        # for the vector of Lagrange multipliers lambda_vec.
    
        signed_weights = self.obj.signed_weights() \
                         + self.cons.signed_weights(lambda_vec)
        
        redY = 1*(signed_weights > 0)
        redW = signed_weights.abs()
        
        redW = self.n*redW/redW.sum()

        if self.debug:
            print("%sclassifier start" % ("_"*9,))
        classifier = pickle.loads(self.pickled_learner)
        classifier.fit(self.X, redY, redW)
        self.n_oracle_calls += 1
        if self.debug:
            print("%sclassifier end" % ("_"*9,))
        
        h = lambda X: classifier.predict(X)
        h_error = self.obj.gamma(h)[0]
        h_gamma = self.cons.gamma(h)
        h_val = h_error + h_gamma.dot(lambda_vec)

        if not self.hs.empty:
            vals =  self.errors + self.gammas.transpose().dot(lambda_vec)
            best_idx = vals.idxmin()
            best_val = vals[best_idx]
        else:
            best_idx = -1
            best_val = np.PINF

        if h_val < best_val-_PRECISION:
            if self.debug:
                print("%sbest_h: val improvement %f" % ("_"*9, best_val-h_val))
                print("%snclassifiers: %d" % (" "*9, len(self.hs)))
            h_idx = len(self.hs)
            self.hs.at[h_idx] = h
            self.classifiers.at[h_idx] = classifier
            self.errors.at[h_idx] = h_error
            self.gammas[h_idx] = h_gamma
            best_idx = h_idx

        return self.hs[best_idx], best_idx

    def add_classifier(self, classifier):
        h = lambda X: classifier.predict(X)
        h_error = self.obj.gamma(h)[0]
        h_gamma = self.cons.gamma(h)
        h_idx = len(self.hs)
        self.hs.at[h_idx] = h
        self.classifiers.at[h_idx] = classifier
        self.errors.at[h_idx] = h_error
        self.gammas[h_idx] = h_gamma


def _mean_pred(dataX, hs, weights):
    # Return a weighted average of predictions produced by classifiers in hs
    
    pred = pd.DataFrame()
    for t in range(len(hs)):
        pred[t] = hs[t](dataX)
    return pred[weights.index].dot(weights)


### Explicit optimization parameters of expgrad

# A multiplier controlling the automatic setting of nu.
_ACCURACY_MUL = 0.5

# Parameters controlling adaptive shrinking of the learning rate.
_REGR_CHECK_START_T = 5
_REGR_CHECK_INCREASE_T = 1.6
_SHRINK_REGRET = 0.8
_SHRINK_ETA = 0.8

# The smallest number of iterations after which expgrad terminates.
_MIN_T = 0

# If _RUN_LP_STEP is set to True, then each step of exponentiated gradient is
# followed by the saddle point optimization over the convex hull of
# classifiers returned so far.
_RUN_LP_STEP = True


def expgrad(dataX, dataA, dataY, learner, dataW=None, cons_class=DP, eps=0.01,
            T=50, nu=None, eta_mul=2.0, debug=False, init_cache=[]):
    """
    Return a fair classifier under specified fairness constraints
    via exponentiated-gradient reduction.
    
    Required input arguments:
      dataX -- a DataFrame containing covariates
      dataA -- a Series containing the protected attribute
      dataY -- a Series containing labels in {0,1}
      learner -- a learner implementing methods fit(X,Y,W) and predict(X),
                 where X is the DataFrame of covariates, and Y and W
                 are the Series containing the labels and weights,
                 respectively; labels Y and predictions returned by
                 predict(X) are in {0,1}

    Optional keyword arguments:
      cons_class -- the fairness measure (default moments.DP)
      eps -- allowed fairness constraint violation (default 0.01)
      T -- max number of iterations (default 50)
      nu -- convergence threshold for the duality gap (default None,
            corresponding to a conservative automatic setting based on the
            statistical uncertainty in measuring classification error)
      eta_mul -- initial setting of the learning rate (default 2.0)
      debug -- if True, then debugging output is produced (default False)

    Returned named tuple with fields:
      best_classifier -- a function that maps a DataFrame X containing
                         covariates to a Series containing the corresponding
                         probabilistic decisions in [0,1]
      best_gap -- the quality of best_classifier; if the algorithm has
                  converged then best_gap<= nu; the solution best_classifier
                  is guaranteed to have the classification error within
                  2*best_gap of the best error under constraint eps; the
                  constraint violation is at most 2*(eps+best_gap)
      last_t -- the last executed iteration; always last_t < T
      best_t -- the iteration in which best_classifier was obtained
      n_oracle_calls -- how many times the learner was called
      n_classifiers -- how many distinct classifiers have been generated
    """

    ExpgradResult = namedtuple("ExgradResult",
                               "best_classifier best_gap last_t best_t"
                               " n_oracle_calls n_classifiers"
                               " hs classifiers weights")

    n = dataX.shape[0]
    assert dataA.shape[0]==n & dataY.shape[0]==n, \
        "the number of rows in all data fields must match"

    if dataW is None:
        dataW = pd.Series(1, dataY.index)
    else:
        dataW = n*dataW / dataW.sum()

    if debug:
        print("...EG STARTING")

    B = 1/eps
    lagr = _Lagrangian(dataX, dataA, dataY, learner, dataW, cons_class, eps, B,
                       debug=debug, init_cache=init_cache)

    theta  = pd.Series(0, lagr.cons.index)
    Qsum = pd.Series()
    lambdas  = pd.DataFrame()
    gaps_EG = []
    gaps = []
    Qs = []
    last_regr_checked = _REGR_CHECK_START_T
    last_gap = np.PINF
    
    for t in range(0, T):
        
        if debug:
            print("...iter=%03d" % t)

        lambda_vec = B*np.exp(theta) / (1+np.exp(theta).sum())
        

        lambdas[t] = lambda_vec
        
        lambda_EG = lambdas.mean(axis=1)
        
        if t == 0:
            h, h_idx = lagr.best_h(0*lambda_vec)
   

        h, h_idx = lagr.best_h(lambda_vec)

        pred_h = h(dataX)
        if t == 0:
            
            if nu is None:
                
                nu = _ACCURACY_MUL * (dataW*(pred_h-dataY).abs()).std() / np.sqrt(n)
            eta_min = nu / (2*B)
            eta = eta_mul / B
            if debug:
                print("...eps=%.3f, B=%.1f, nu=%.6f, T=%d, eta_min=%.6f"
                      % (eps, B, nu, T, eta_min))
                #print(lagr.cons.index)
        
        # if not Qsum.index.contains(h_idx):  # OLD CODE
        if h_idx not in Qsum.index:
            Qsum.at[h_idx] = 0.0
        Qsum[h_idx] += 1.0
        gamma = lagr.gammas[h_idx]
        
        Q_EG = Qsum / Qsum.sum()
        res_EG = lagr.eval_gap(Q_EG, lambda_EG, nu)
        gap_EG = res_EG.gap()
        gaps_EG.append(gap_EG)
          
        if (t == 0) or not _RUN_LP_STEP:
            gap_LP = np.PINF
        else:
            Q_LP, lambda_LP, res_LP = lagr.solve_linprog(nu)
            gap_LP = res_LP.gap()
            
        if gap_EG < gap_LP:
            Qs.append(Q_EG)
            gaps.append(gap_EG)
        else:
            Qs.append(Q_LP)
            gaps.append(gap_LP)

        if debug:
            print("%seta=%.6f, L_low=%.3f, L=%.3f, L_high=%.3f"
                  ", gap=%.6f, disp=%.3f, err=%.3f, gap_LP=%.6f"
                  % (" "*9, eta, res_EG.L_low, res_EG.L, res_EG.L_high,
                     gap_EG, res_EG.gamma.max(), res_EG.error, gap_LP))

        if (gaps[t] < nu) and (t >= _MIN_T):
            break

        if t >= last_regr_checked*_REGR_CHECK_INCREASE_T:
            best_gap = min(gaps_EG)

            if best_gap > last_gap*_SHRINK_REGRET:
                eta *= _SHRINK_ETA
            last_regr_checked = t
            last_gap = best_gap
            
        theta += eta*(gamma-eps)
       
    last_t = len(Qs)-1
    gaps_series = pd.Series(gaps)
    gaps_best = gaps_series[gaps_series<=gaps_series.min()+_PRECISION]
    best_t = gaps_best.index[-1]
    best_Q = Qs[best_t]
    hs = lagr.hs
    best_classifier = lambda X : _mean_pred(X, hs, best_Q)
    best_gap = gaps[best_t]

    res = ExpgradResult(best_classifier=best_classifier,
                        hs=lagr.hs,
                        classifiers=lagr.classifiers,
                        weights=best_Q,
                        best_gap=best_gap,
                        last_t=last_t,
                        best_t=best_t,
                        n_oracle_calls=lagr.n_oracle_calls,
                        n_classifiers=len(lagr.hs))
    if debug:
        print("...eps=%.3f, B=%.1f, nu=%.6f, T=%d, eta_min=%.6f"
              % (eps, B, nu, T, eta_min))
        print("...last_t=%d, best_t=%d, best_gap=%.6f"
              ", n_oracle_calls=%d, n_hs=%d"
              % (res.last_t, res.best_t, res.best_gap,
                 res.n_oracle_calls, res.n_classifiers))
        tmp, tmp, best_gamma, best_error = lagr.eval(best_classifier, 0*lambda_vec)
        print("...disp=%.6f, err=%.6f"
              % (best_gamma.max(), best_error))

    return res



class DP_theta(_CondOpportunity):
    """DP for regression"""
    short_name = "DP-reg"
    def __init__(self, dataX, dataA, dataY):
        super().__init__(dataX, dataA, dataY,
                         dataX["theta"])

def augment_data_sq(x, a, y, Theta):
    """
    Augment the dataset so that the x carries an additional feature of theta
    Then also attach appropriate weights to each data point.

    Theta: Assume uniform grid Theta
    """
    n = np.shape(x)[0]  # number of original data points
    num_theta = len(Theta)
    width = Theta[1] - Theta[0]
    X_aug = pd.concat(repeat(x, num_theta))
    A_aug = pd.concat(repeat(a, num_theta))
    Y_values = pd.concat(repeat(y, num_theta))

    theta_list = [s for theta in Theta for s in repeat(theta, n)]
    # Adding theta to the feature
    X_aug['theta'] = pd.Series(theta_list, index=X_aug.index)
    X_aug.index = range(n * num_theta)
    # Y_aug.index = range(n * num_theta)
    A_aug.index = range(n * num_theta)
    Y_values.index = range(n * num_theta)

    # two helper functions
    sq_loss = lambda a, b: (a - b)**2  # square loss function
    weight_assign = lambda theta, y: (sq_loss(theta + width/2, y) - sq_loss(theta - width/2, y))
    W = weight_assign(X_aug['theta'], Y_values)
    Y_aug = 1*(W < 0)
    W = abs(W)
    # Compute the weights
    return X_aug, A_aug, Y_aug, W


def train_FairRegression(x, a, y, eps, Theta, learner,
                                constraint="DP", loss="square", init_cache=[]):
    """
    Run fair algorithm on the training set and then record
    the metrics on the training set.

    x, a, y: the training set input for the fair algorithm
    eps: the desired level of fairness violation
    Theta: the set of thresholds (z's in the paper)
    """
    alpha = (Theta[1] - Theta[0])/2

    if loss == "square":  # squared loss reweighting
        X, A, Y, W = augment_data_sq(x, a, y, Theta)
    #elif loss == "absolute":  # absolute loss reweighting (uniform)
     #   X, A, Y, W = augment.augment_data_ab(x, a, y, Theta)
    #elif loss == "logistic":  # logisitic reweighting
     #   X, A, Y, W = augment.augment_data_logistic(x, a, y, Theta)
    else:
        raise Exception('Loss not supported: ', str(loss))
 
    if constraint == "DP":  # DP constraint
        result = expgrad(X, A, Y, learner, dataW=W,
                             cons_class=DP_theta, eps=eps,
                             debug=False, init_cache=init_cache)
      
    else:  # exception
        raise Exception('Constraint not supported: ', str(constraint))
    

    print('epsilon value: ', eps, ': number of oracle calls', result.n_oracle_calls)

    model_info = {}  # dictionary for saving data
    model_info['loss_function'] = loss
    model_info['constraint'] = constraint
    model_info['exp_grad_result'] = result
  
    return model_info



def fair_train_test(x_train, a_train, y_train, x_test, a_test, y_test, eps_list, learner, constraint="DP",
                   loss="square",  init_cache=[]):
    """
    Input:
    - dataset name
    - size parameter for data parser
    - eps_list: list of epsilons for exp_grad
    - learner: the solver for CSC
    - constraint: fairness constraint name
    - loss: loss function name
    - random_seed

    Output: Results for
    - exp_grad: (eps, loss) for training and test sets
    - benchmark method: (eps, loss) for training and test sets
    """
    '''
    if dataset == 'lawschool':
        x, a, y = parser.clean_lawschool_full()
    elif dataset == 'communities':
        x, a, y = parser.clean_communities_full()
    elif dataset == 'adult_full':
        x, a, y = parser.clean_adult_full()
    else:
        raise Exception('DATA SET NOT FOUND!')
  
    if _SMALL:
        x, a, y = subsample(x, a, y, size)

    x_train, a_train, y_train, x_test, a_test, y_test = train_test_split_groups(x, a, y, random_seed=DATA_SPLIT_SEED)
    '''


    fair_model = {}
    train_evaluation = {}
    test_evaluation = {}
    for eps in eps_list:
        fair_model[eps] = train_FairRegression(x_train,
                                                         a_train,
                                                         y_train, eps,
                                                         Theta,
                                                         learner,
                                                         constraint,
                                                         loss,
                                                         init_cache=init_cache)

        train_evaluation[eps] =evaluate_FairModel(x_train,
                                                            a_train,
                                                            y_train,
                                                            loss,
                                                            fair_model[eps]['exp_grad_result'],
                                                            Theta)
        
        test_evaluation[eps] = evaluate_FairModel(x_test,
                                                           a_test,
                                                           y_test,
                                                           loss,
                                                           fair_model[eps]['exp_grad_result'],
                                                           Theta)

    result = {}
    result['dataset'] = 'dataset'
    result['learner'] = learner.name
    result['loss'] = loss
    result['constraint'] = constraint
    result['train_eval'] = train_evaluation
    result['test_eval'] = test_evaluation
    result['fair_model'] = fair_model
    return result


def approximate_data(X, Y, W, Theta):
    """
    Given the augmented data (X, Y, W), recover for each example the
    prediction in Theta + alpha/2 that minimizes the cost;
    Thus we reduce the size back to the same orginal size
    """

    start = time.time()
    n = int(len(X) / len(Theta))  # size of the dataset
    alpha = (Theta[1] - Theta[0])/2
    x = X.iloc[:n, :].drop(['theta'], 1)
    pred_vec = Theta + alpha  # the vector of possible preds

    minimizer = {}

    pred_vec = {}  # mapping theta to pred vector
    for pred in (Theta + alpha):
        pred_vec[pred] = (1 * (pred >= pd.Series(Theta)))

    for i in range(n):
        index_set = [i + j * n for j in range(len(Theta))]  # the set of rows for i-th example
        W_i = W.iloc[index_set]
        Y_i = Y.iloc[index_set]
        Y_i.index = range(len(Y_i))
        W_i.index = range(len(Y_i))
        cost_i = {}
        for pred in (Theta + alpha):
            cost_i[pred] = abs(Y_i - pred_vec[pred]).dot(W_i)
        minimizer[i] = min(cost_i, key=cost_i.get)

    end = time.time()
    print('orginal method time = ', end - start)

    start = time.time()
    xx , yy = approximate_data_aligned(X, Y, W, Theta)
    end = time.time()
    print('optimized method time = ', end - start)

    print('X equall :', xx.equals(x),' y equall :', yy.equals(pd.Series(minimizer)))

    return x, pd.Series(minimizer)


def approximate_data_aligned(X, Y, W, Theta):
    n = int(len(X) / len(Theta))
    alpha = (Theta[1] - Theta[0]) / 2
    pred_vec = Theta + alpha
    x = X.iloc[:n, :].drop(['theta'], 1)
    pred_dict = {}
    for pred in pred_vec:
        pred_dict[pred] = 1 * (pred >= Theta)
    minimizer = {}
    for i in range(n):
        index_set = [i + j * n for j in range(len(Theta))]
        W_i = W.iloc[index_set].values
        Y_i = Y.iloc[index_set].values
        cost_i = []
        for pred in pred_vec:
            costs = np.abs(Y_i - pred_dict[pred]) * W_i
            cost_i.append(np.sum(costs))
        best_pred_index = np.argmin(cost_i)
        minimizer[i] = pred_vec[best_pred_index]
    return x, pd.Series(minimizer)



    
class XGB_Regression_Learner:

    def __init__(self, Theta):
        self.Theta = Theta
        self.name = "XGB Regression"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params = {'max_depth': 12,  
                  'n_estimators': 50, 
                  'eta' : 0.01,
                  'n_jobs':mp.cpu_count()
                  }
        print(params)
        
        self.regr = xgb.XGBRegressor(**params)

    def fit(self, X, Y, W):
        print('XGBoost training begins')
        matX, vecY = approximate_data_aligned(X, Y, W, self.Theta)
        
       # matX = torch.from_numpy(matX.values).float().to(self.device)
        
        self.regr.fit(matX, vecY)
        print('XGBoost training ends')
        
       

    def predict(self, X):
        #t = torch.from_numpy(X.drop(['theta'], axis=1).values).float().to(self.device)
        y_values = self.regr.predict(X.drop(['theta'], axis=1))

        pred = 1*(y_values - X['theta'] >= 0)  # w * x - theta
        return pred


class RF_Regression_Learner:

    def __init__(self, Theta):
        self.Theta = Theta
        self.name = "RF Regression"
        self.regr = RandomForestRegressor(max_depth=4, 
                                          n_estimators=200,
                                          n_jobs= mp.cpu_count())
        print('mp.cpu_count()', mp.cpu_count())

    def fit(self, X, Y, W):
        print('RF training begins')
        matX, vecY = approximate_data_aligned(X, Y, W, self.Theta)
        self.regr.fit(matX, vecY)
        print('RF training ends')

    def predict(self, X):
        y_values = self.regr.predict(X.drop(['theta'], axis=1))
        pred = 1*(y_values - X['theta'] >= 0)  # w * x - theta
        return pred

class MLP_Regression_Learner:

    def __init__(self, Theta):
        self.Theta = Theta
        self.name = "MLP Regression"
        print('MLP fitting')
        self.regr = MLPRegressor(hidden_layer_sizes= (256) , 
                                          alpha=0.001,
                                          verbose=False,
                                          early_stopping=True,
                                          learning_rate = 'adaptive',
                                          learning_rate_init = 0.01)


    def fit(self, X, Y, W):
        print('MLP training begins')
        matX, vecY = approximate_data_aligned(X, Y, W, self.Theta)
        self.regr.fit(matX, vecY)
        print('MLP training ends')

    def predict(self, X):
        y_values = self.regr.predict(X.drop(['theta'], axis=1))
        pred = 1*(y_values - X['theta'] >= 0)  # w * x - theta
        return pred


class TabNet_Regression_Learner:

    def __init__(self, Theta):
        self.Theta = Theta
        self.name = "TabNet Regression"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        params = {'n_d': 16,'n_a': 16, 'n_steps': 6, 'verbose': 0, 'device_name': device}
        print('Tabnet ',params,'  ', device)
        self.regr = TabNetRegressor(**params)

    def fit(self, X, Y, W):
        start_t = time.time()  
        print('TabNet approximate_data_aligned begins')      
        matX, vecY = approximate_data_aligned(X, Y, W, self.Theta)
        end_t = time.time()
        print('TabNet approximate_data_aligned ends', ' time = ', end_t-start_t)
        print('TabNet training begins')
        start_t = time.time()
        self.regr.fit(matX.values, np.array(vecY).reshape(-1, 1), max_epochs = 25)
        end_t = time.time()
        print('TabNet training ends', 'training time = ', end_t-start_t)

    def predict(self, X):
        print('TabNet predict begins')
        start_t = time.time()  
        y_values = self.regr.predict(X.drop(['theta'], axis=1).values)
        pred = 1*(y_values.flatten() - X['theta'] >= 0)  # w * x - theta
        end_t = time.time()
        print('TabNet predict ends', 'training time = ', end_t-start_t)
        return pred
    

 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size = 2, output_size=1, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)



    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        lstm_out, _ = self.lstm(x, (h0, c0))
        output = self.fc(lstm_out[:, -1, :])
        return output

    def fit(self, X_train, y_train,  epochs=25, learning_rate=0.00001, batch_size=64, patience=10, device='cpu'):

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        static = [0,1,2,471,472]
        X_train = X_train.values
        x_train_new = np.delete(X_train, static, 1)
        new_data2 = np.reshape(x_train_new, (X_train.shape[0], -1, 52))
    
        static_data = X_train[:, static]
        static_data = np.repeat(static_data[:, None, :], 9, axis=1)
        X_train = np.concatenate((static_data,new_data2 ), axis=-1)

        dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


        for epoch in range(epochs):
            self.train()  # Set the model to training mode
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()




    def predict(self, X_test):
 
        dataset = TensorDataset(torch.from_numpy(X_test).float())
        dataloader = DataLoader(dataset, batch_size=64)  # Example batch size, adjust as needed
        predictions = []

        self.eval()
        with torch.no_grad():
            for inputs in dataloader:
                outputs = self(inputs[0])
                predictions.append(outputs.cpu().numpy())

        return np.concatenate(predictions, axis=0)

 

class LSTM_Regression_Learner:

    def __init__(self, Theta):
        self.device =  "cpu"
        self.Theta = Theta
        self.name = "LSTM Regression"
        self.regr = LSTM(input_size=57)

    def fit(self, X, Y, W):

        matX, vecY = approximate_data_aligned(X, Y, W, self.Theta)

        self.regr.fit(matX, np.array(vecY))

    
    def predict(self, X):

        x_theta = X['theta']
        X = X.drop(['theta'], axis=1).values
        static = [0,1,2,471,472]
        static_data = X[:, static]
       
        static_data = np.repeat(static_data[:, None, :], 9, axis=1)
   
        X = np.concatenate((static_data,np.reshape(( np.delete(X, static, 1)), (X.shape[0], -1, 52)) ), axis=-1)
        y_values = self.regr.predict(X)
        
        pred = 1*(y_values.flatten() - x_theta >= 0)  # w * x - theta

        return pred
    




class GRU(nn.Module):
    def __init__(self, input_size, hidden_size=2, output_size=1, num_layers=1):

        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        gru_out, _ = self.gru(x, h0)
        output = self.fc(gru_out[:, -1, :])
        return output

    def fit(self, X_train, y_train, epochs=25, learning_rate=0.00001, batch_size=64, device='cpu'):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        static = [0,1,2,471,472]
        X_train = X_train.values
        x_train_new = np.delete(X_train, static, 1)
        new_data2 = np.reshape(x_train_new, (X_train.shape[0], -1, 52))
    
        static_data = X_train[:, static]
        static_data = np.repeat(static_data[:, None, :], 9, axis=1)
        X_train = np.concatenate((static_data,new_data2 ), axis=-1)


        dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


        for epoch in range(epochs):
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

    
    def predict(self, X_test, device='cpu'):
        dataset = TensorDataset(torch.from_numpy(X_test).float())
        dataloader = DataLoader(dataset, batch_size=64)  # Example batch size, adjust as needed
        predictions = []

        self.eval()
        with torch.no_grad():
            for inputs in dataloader:
                outputs = self(inputs[0])
                predictions.append(outputs.cpu().numpy())

        return np.concatenate(predictions, axis=0)


class GRU_Regression_Learner:

    def __init__(self, Theta):
        self.device =  "cpu"
        self.Theta = Theta
        self.name = "GRU Regression"
        self.regr = GRU(input_size=57)

    def fit(self, X, Y, W):

        matX, vecY = approximate_data_aligned(X, Y, W, self.Theta)

        self.regr.fit(matX, np.array(vecY))

    
    def predict(self, X):

        x_theta = X['theta']
        X = X.drop(['theta'], axis=1).values
        static = [0,1,2,471,472]
        static_data = X[:, static]
       
        static_data = np.repeat(static_data[:, None, :], 9, axis=1)
   
        X = np.concatenate((static_data,np.reshape(( np.delete(X, static, 1)), (X.shape[0], -1, 52)) ), axis=-1)
        y_values = self.regr.predict(X)
        
        pred = 1*(y_values.flatten() - x_theta >= 0)  # w * x - theta

        return pred
