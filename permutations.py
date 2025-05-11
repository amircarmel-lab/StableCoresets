import numpy as np
import os
from numpy.random import random
from itertools import combinations, permutations
from time import time
import pulp
from pulp import LpProblem, LpVariable, LpMaximize, value, lpSum
# Added missing tqdm import
from tqdm import tqdm

# Fixed all caps to lowercase for proper module exports
__all__ = [
    "rank_parity",
    "rank_equality",
    "rank_calibration"
]

def _pairs(n):
    return n*(n-1)/2.



#calibration
def _merge_cal(h1,h2,g):
    count = 0
    arr = []
    i=0
    j=0
    while i < len(h1) and j < len(h2):
        #pairs that are correctly ordered
        if h1[i][1] < h2[j][1]:
            arr.append(h1[i])
        else:
            while(j<len(h2) and h1[i][1] > h2[j][1]):
                # count inverted pairs containing group
                if h2[j][2] == g:
                    count += len(h1[i:])
                else:
                    count += np.bincount(np.array(h1, dtype=int)[i:,2], minlength=2)[g]
                arr.append(h2[j])
                j+=1
            arr.append(h1[i])
        i+=1
#     add any remaining elements
    while i < len(h1):
        arr.append(h1[i])
        i+=1
    while j < len(h2):
        arr.append(h2[j])
        j+=1
    return arr, count


#equality
def _merge_eq(h1,h2,g):
    count = 0
    arr = []
    i=0
    j=0
    while i < len(h1) and j< len(h2):
        #pairs that are correctly ordered
        if h1[i][1] < h2[j][1]:
            arr.append(h1[i])
        else:
            while(j<len(h2) and h1[i][1] > h2[j][1]):
                if(h2[j][2] != g):
                # count inverted pairs favoring group
                    count += np.bincount(np.array(h1, dtype=int)[i:,2], minlength=2)[g]
                arr.append(h2[j])
                j+=1
            arr.append(h1[i])
        i+=1
    while i < len(h1):
        arr.append(h1[i])
        i+=1
    while j < len(h2):
        arr.append(h2[j])
        j+=1
    return arr, count

# parity
def _merge_parity(h1,h2,g):
    count1 = 0
    count2 = 0
    i=0
    while i < len(h1):
        if h1[i]== g:
            count1 +=1
        i+=1
    i=0
    while i < len(h2):
        if h2[i] != g:
            count2 +=1
        i+=1
    count = count1*count2
    return np.concatenate([h1,h2]), count


#compute FARE error metrics using adaptation of mergesort to perform pair counting.
def _count_inversions(data, s, e, merge_fnc, g):
    if s == e: #base case
        return [data[s]], 0
    else:
        m = s + int((e-s)/2)
        h1,c1 = _count_inversions(data, s, m, merge_fnc, g)
        h2,c2 = _count_inversions(data, m+1, e, merge_fnc, g)
        merged, c = merge_fnc(h1,h2,g)
        return merged, (c1+c2+c)


def rank_equality(y_true, y_pred, groups):
    """Compute the rank equality error between two rankings.

    Parameters
    ----------
    y_true : array-like of shape = (n_samples)
        Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples)
        Estimated target values.

    groups : array-like of shape = (n_samples)
        Binary integer array with group labels for each sample.

    Returns
    ----------
    error0 : float
        The rank parity error for group 0.

    error1 : float
        The rank parity error for group 1.

    Examples
    --------
    >>> y_true = [1,2,3,4]
    >>> y_pred = [1,3,4,2]
    >>> groups =[0,1,0,1]
    >>> rank_equality(y_true,y_pred,groups)
    (0.0, 0.25)
    """
    #sort instances by y_pred
    r = np.transpose([y_true,y_pred,groups])
    r = r[r[:,0].argsort()]
    #count the items in each group for narmalization
    len_groups = np.bincount(np.array(groups, dtype=int), minlength=2)
    p = len_groups[0]*len_groups[1]
    e0 = 0 if p == 0 else _count_inversions(r, 0, len(r)-1, _merge_eq, 0)[1] / p
    e1 = 0 if p == 0 else _count_inversions(r, 0, len(r)-1, _merge_eq, 1)[1] / p
    return e0, e1


def rank_calibration(y_true, y_pred, groups):
    """Compute the rank calibration error between two rankings.

    Parameters
    ----------
    y_true : array-like of shape = (n_samples)
        Ground truth (correct) target values.

    y_pred : array-like of shape = (n_samples)
        Estimated target values.

    groups : array-like of shape = (n_samples)
        Binary integer array with group labels for each sample.

    Returns
    -------
    error0 : float
        The rank parity error for group 0.

    error1 : float
        The rank parity error for group 1.

    Examples
    --------
    >>> y_true = [1,2,3,4]
    >>> y_pred = [1,3,4,2]
    >>> groups =[0,1,0,1]
    >>> rank_calibration(y_true,y_pred,groups)
    (0.20000000000000001, 0.40000000000000002)
    """
    #sort instances by y_pred
    r = np.transpose([y_true,y_pred,groups])
    r = r[r[:,0].argsort()]
    #count the items in each group for normalization
    len_groups = np.bincount(np.array(groups, dtype=int), minlength=2)
    p0 = _pairs(len(r)) - _pairs(len_groups[1])
    p1 = _pairs(len(r)) - _pairs(len_groups[0])
    # count pairs
    e0 = 0 if p0 == 0 else _count_inversions(r, 0, len(r)-1, _merge_cal, 0)[1] / p0
    e1 = 0 if p1 == 0 else _count_inversions(r, 0, len(r)-1, _merge_cal, 1)[1] / p1
    return e0, e1

def rank_parity(y,groups):
    """Compute the rank parity error for one ranking.

    Parameters
    ----------
    y: array-like of shape = (n_samples)
        Rank values.

    groups : array-like of shape = (n_samples)
        Binary integer array with group labels for each sample.

    Returns
    -------
    error0 : float
        The rank parity error for group 0.

    error1 : float
        The rank partiy error for group 1.

    Examples
    --------
    >>> y = [1,3,4,2]
    >>> groups =[0,1,0,1]
    >>> rank_parity(y, groups)
    (0.5,0.5)
    """
    r = np.transpose([y,np.array(groups)])
    r = r[r[:,0].argsort()]
    g= np.array(r[:,1], dtype=int)
    e0 = _count_inversions(g, 0, len(g)-1, _merge_parity, 0)[1]
    e1 = _count_inversions(g, 0, len(g)-1, _merge_parity, 1)[1]

    return e0,e1


"""
utils
"""

def kendalltau_dist(rank_a, rank_b):
    """Compute Kendall tau distance between two rankings."""
    tau = 0
    n_candidates = len(rank_a)
    for i, j in combinations(range(n_candidates), 2):
        tau += (np.sign(rank_a[i] - rank_a[j]) == -np.sign(rank_b[i] - rank_b[j]))
    return tau

def build_graph(ranks):
    """Build edge weights graph from input rankings."""
    n_voters, n_candidates = ranks.shape
    edge_weights = np.zeros((n_candidates, n_candidates))
    for i, j in combinations(range(n_candidates), 2):
        preference = ranks[:, i] - ranks[:, j]
        h_ij = np.sum(preference < 0)  # prefers i to j
        h_ji = np.sum(preference > 0)  # prefers j to i
        if h_ij > h_ji:
            edge_weights[i, j] = h_ij - h_ji
        elif h_ij < h_ji:
            edge_weights[j, i] = h_ji - h_ij
    return edge_weights

def gen_groups(d, p):
    """
    Generate group assignments for n candidates with bias parameter p.
    p = 0.5 gives random fair assignment
    p > 0.5 biases towards group 1
    """
    len1 = int(d/2)
    len0 = d - len1
    groups = []
    while len1 > 0 and len0 > 0:
        if random() < p:
            groups.append(1)
            len1 -= 1
        else:
            groups.append(0)
            len0 -= 1
    while len1 > 0:
        groups.append(1)
        len1 -= 1
    while len0 > 0:
        groups.append(0)
        len0 -= 1
    return groups


"""
Ranking aggregation algorithms with and without fairness constraints.
"""

def aggregate_kemeny_new(n_candidates, ranks):
    """
    Kemeny rank aggregation using PuLP.

    Args:
        n_candidates: Number of candidates
        ranks: Input rankings (n_samples × n_candidates)

    Returns:
        aggregated_rank: Aggregated ranking
        runtime: Computation time in seconds
    """
    print("Setting up Kemeny aggregation problem...")
    t0 = time()

    # Create optimization problem
    prob = LpProblem("Kemeny_Aggregation", LpMaximize)

    # Helper function for indexing
    idx = lambda i, j: n_candidates * i + j

    # Create binary variables for each pair
    x = {}
    print("Creating binary variables...")
    for i in tqdm(range(n_candidates), desc="Creating variables"):
        for j in range(n_candidates):
            x[idx(i,j)] = LpVariable(f"x_{i}_{j}", 0, 1, 'Binary')

    # Add pairwise constraints (xij + xji = 1)
    print("Adding pairwise constraints...")
    pairs = list(combinations(range(n_candidates), 2))
    for i, j in tqdm(pairs, desc="Adding pairwise constraints"):
        prob += x[idx(i,j)] + x[idx(j,i)] == 1

    # Add transitivity constraints (xij + xjk + xki >= 1)
    print("Adding transitivity constraints...")
    perms = list(permutations(range(n_candidates), 3))
    for i, j, k in tqdm(perms, desc="Adding transitivity constraints"):
        prob += x[idx(i,j)] + x[idx(j,k)] + x[idx(k,i)] >= 1

    # Set objective (maximize agreement with input rankings)
    print("Building graph for objective function...")
    edge_weights = build_graph(ranks)
    c = -1 * edge_weights.ravel()

    # Define objective function
    prob += sum(c[i] * x[i] for i in range(n_candidates * n_candidates))

    try:
        # Solve the problem
        print("Solving optimization problem...")
        t0 = time()
        solver = pulp.GUROBI(msg=False)
        prob.solve(solver)
        t1 = time()
        solve_time = t1 - t0
        print(f"Solved in {solve_time:.2f} seconds")

        # Get solution if optimal
        if prob.status == 1:  # 1 indicates optimal solution found
            print("Extracting solution...")
            sol = []
            for i in tqdm(range(n_candidates * n_candidates), desc="Extracting solution"):
                sol.append(value(x[i]))
            sol = np.array(sol)
            sol[sol == None] = 0
            aggr_rank = np.sum(sol.reshape((n_candidates, n_candidates)), axis=1)
            return aggr_rank, solve_time
        else:
            print(f"Optimal solution not found. Status: {prob.status}")
            return None, solve_time

    except Exception as e:
        print(f"Optimization failed: {e}")
        t1 = time()
        return None, t1-t0


def build_parity_constraints(groups):
    """Build matrix for parity constraints."""
    n_candidates = len(groups)
    edges = np.zeros((n_candidates, n_candidates))
    for i, j in combinations(range(n_candidates), 2):
        edges[i, j] = (groups[i] - groups[j])
        edges[j, i] = -(groups[i] - groups[j])
    return edges.ravel()
    
def aggregate_parity_new(n_candidates, ranks, groups, delta):
    """
    Fair rank aggregation using PuLP with parity constraints.

    Args:
        n_candidates: Number of candidates
        ranks: Input rankings (n_samples × n_candidates)
        groups: Group assignments for candidates
        delta: Fairness threshold

    Returns:
        aggregated_rank: Aggregated ranking
        runtime: Computation time in seconds
    """
    print("Setting up fair rank aggregation problem...")
    init_time = time()

    try:
        # Create optimization problem
        prob = LpProblem("Fair_Kemeny_Aggregation", LpMaximize)

        # Helper function for indexing
        idx = lambda i, j: n_candidates * i + j

        # Create binary variables for each pair
        x = {}
        # print("Creating binary variables..")
        for i in tqdm(range(n_candidates), desc="Creating variables"):
            for j in range(n_candidates):
                x[idx(i,j)] = LpVariable(f"x_{i}_{j}", 0, 1, 'Binary')

        # Add pairwise constraints (xij + xji = 1)
        # print("Adding pairwise constraints...")
        pairs = list(combinations(range(n_candidates), 2))
        for i, j in tqdm(pairs, desc="Adding pairwise constraints"):
            prob += x[idx(i,j)] + x[idx(j,i)] == 1

        # Add transitivity constraints (xij + xjk + xki >= 1)
        # print("Adding transitivity constraints...")
        perms = list(permutations(range(n_candidates), 3))
        for i, j, k in tqdm(perms, desc="Adding transitivity constraints"):
            prob += x[idx(i,j)] + x[idx(j,k)] + x[idx(k,i)] >= 1

        # Add parity constraints
        # print("Building parity constraints...")
        parity = build_parity_constraints(groups)
        parity_sum = lpSum(parity[i] * x[i] for i in range(n_candidates * n_candidates))
        delta_adj = delta * ((sum(groups))*(len(groups)-sum(groups)))
        prob += parity_sum <= delta_adj    # Upper bound
        prob += parity_sum >= -delta_adj   # Lower bound

        # Set objective (maximize agreement with input rankings)
        # print("Building graph for objective function...")
        edge_weights = build_graph(ranks)
        c = -1 * edge_weights.ravel()
        prob += lpSum(c[i] * x[i] for i in range(n_candidates * n_candidates))

        # Solve
        # print("Solving optimization problem...")
        t0 = time()
        prob.solve()
        t1 = time()
        solve_time = t1 - t0
        # print(f"Solved in {solve_time:.2f} seconds")

        # Check if solution was found
        if prob.status == 1:  # 1 indicates optimal solution found
            # print("Extracting solution...")
            # Get solution values
            sol = []
            for i in tqdm(range(n_candidates * n_candidates), desc="Extracting solution"):
                sol.append(value(x[i]))

            sol = np.array(sol)
            sol[sol == None] = 0
            aggr_rank = np.sum(sol.reshape((n_candidates, n_candidates)), axis=1)
            return aggr_rank, solve_time
        else:
            print(f"Optimal solution not found. Status: {prob.status}")
            return None, solve_time

    except Exception as e:
        print(f"Optimization failed: {e}")
        return None, -1