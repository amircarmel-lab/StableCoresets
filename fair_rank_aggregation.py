from random import random
import numpy as np
import pulp
from itertools import combinations, permutations


def borda_sort(partial_lists, ground_set):
    n_total = len(ground_set)
    scores = {elem: 0 for elem in ground_set}  # Initialize scores for all elements
    
    for l in partial_lists:
        # Compute scores for ranked elements
        for idx, elem in enumerate(reversed(l)):
            scores[elem] += idx
        
        # Compute scores for unranked elements
        ranked_elements = set(l)
        unranked_elements = ground_set - ranked_elements
        
        if unranked_elements:  # Only process if there are unranked elements
            n_ranked = len(l)
            
            # Calculate excess score
            total_score_full = n_total * (n_total - 1) / 2  # Sum of 0 to n_total-1
            total_score_used = n_ranked * (n_ranked - 1) / 2  # Sum of 0 to n_ranked-1
            excess_score = total_score_full - total_score_used
            
            # Distribute excess score equally
            score_per_unranked = excess_score / len(unranked_elements)
            for elem in unranked_elements:
                scores[elem] += score_per_unranked
    
    # Sort elements by their scores in descending order
    return sorted(scores.keys(), key=lambda elem: scores[elem], reverse=True)

def kendalltau_dist(rank_a, rank_b):
    """Compute Kendall tau distance between two rankings."""
    tau = 0
    n_candidates = len(rank_a)
    for i, j in combinations(range(n_candidates), 2):
        tau += (np.sign(rank_a[i] - rank_a[j]) == -np.sign(rank_b[i] - rank_b[j]))
    return tau
    
def compute_kendall_tau_cost(ranking, all_rankings, sample_size = None):
    """
    Compute the average Kendall tau distance between a ranking and a sampled subset of rankings.

    Args:
        ranking: A candidate consensus ranking
        all_rankings: A set of rankings (list of lists or numpy array of lists)
        sample_size: Number of rankings to sample for cost estimation (default: 2000)

    Returns:
        The average Kendall tau distance over the sampled rankings
    """
    # Adjust sample size if it exceeds the size of all_rankings

    # Randomly sample indices without replacement
    if sample_size is not None and sample_size < len(all_rankings):
        sampled_indices = np.random.choice(len(all_rankings), sample_size, replace=True)
        sampled_rankings = [all_rankings[i] for i in sampled_indices]
    else:
        sampled_rankings = all_rankings

    # Compute total distance on sampled rankings
    total_distance = 0
    for rank in sampled_rankings:
        partial_elements = set(rank)
        induced_ranking = [elem for elem in ranking if elem in partial_elements]
        total_distance += kendalltau_dist(induced_ranking, rank)

    # Return average distance
    return total_distance / len(sampled_rankings)

def find_best_perm(rankings):
    min_cost = float('inf')
    min_perm = []
    for rank in tqdm(rankings):
        cost = compute_kendall_tau_cost(rank, rankings)
        if cost < min_cost:
            min_cost = cost
            min_perm = rank
    return rank

def kemeny_rank_aggregation(rankings, ground_set):
    """
    Kemeny rank aggregation using Integer Linear Programming.
    
    Args:
        rankings: List of rankings, each ranking is a list of items
        solver_name: Name of the solver to use ('GUROBI', 'CBC', etc.)
    
    Returns:
        aggregated_ranking: The optimal aggregation
    """
    # First convert items to indices for ILP processing
    all_items = sorted(list(ground_set))
    
    # Create mapping from items to indices
    item_to_idx = {item: idx for idx, item in enumerate(all_items)}
    idx_to_item = {idx: item for idx, item in enumerate(all_items)}
    
    # Convert rankings to use indices
    idx_rankings = []
    for ranking in rankings:
        idx_ranking = [item_to_idx[item] for item in ranking]
        idx_rankings.append(idx_ranking)
    
    n_candidates = len(all_items)
    
    # Create optimization problem
    prob = pulp.LpProblem("Kemeny_Aggregation", pulp.LpMinimize)
    
    # Create binary variables for each pair (i,j)
    x = {}
    for i in range(n_candidates):
        for j in range(n_candidates):
            if i != j:  # No need for variables comparing an item to itself
                x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", 0, 1, 'Binary')
    
    # Add pairwise constraints (xij + xji = 1)
    for i, j in combinations(range(n_candidates), 2):
        prob += x[(i, j)] + x[(j, i)] == 1
    
    # Add transitivity constraints (xij + xjk - xik <= 1)
    for i, j, k in permutations(range(n_candidates), 3):
        prob += x[(i, j)] + x[(j, k)] - x[(i, k)] <= 1
    
    # Build objective function - minimize disagreements
    objective_terms = []
    
    for voter_ranking in idx_rankings:
        # Create a mapping from item to its position in this voter's ranking
        item_positions = {item: pos for pos, item in enumerate(voter_ranking)}
        
        # For each pair of items, add a penalty if the aggregated ranking disagrees
        for i in range(n_candidates):
            for j in range(i+1, n_candidates):
                # If i comes before j in voter's ranking
                if i in item_positions and j in item_positions:
                    if item_positions[i] < item_positions[j]:
                        objective_terms.append(x[(j, i)])  # Penalty for reversing order
                    else:
                        objective_terms.append(x[(i, j)])
    
    # Set the objective
    prob += pulp.lpSum(objective_terms)
                # solver = pulp.GUROBI_CMD(path=gurobi_path, msg=True)
            # status = problem.solve(solver)
    # Solve the problem
    try:
        try:
            solver = pulp.GUROBI_CMD(msg=False)
        except:
            print("ERROR")
        prob.solve(solver)
        
        # Check if an optimal solution was found
        if prob.status == pulp.LpStatusOptimal:
            # Calculate scores
            scores = np.zeros(n_candidates)
            for i in range(n_candidates):
                for j in range(n_candidates):
                    if i != j and pulp.value(x[(i, j)]) > 0.5:  # i preferred to j
                        scores[i] += 1
            
            # Get aggregated ranking indices
            aggregated_idx = np.argsort(-scores).tolist()
            
            # Convert back to original items
            aggregated_ranking = [idx_to_item[idx] for idx in aggregated_idx]
            
            return aggregated_ranking
        else:
            print(f"Optimal solution not found. Status: {prob.status}")
            return None
            
    except Exception as e:
        print(f"Optimization failed: {e}")
        return None


def kemeny_rank_aggregation_with_parity(rankings, ground_set, groups, delta=0.1):
    """
    Kemeny rank aggregation with parity constraints using Integer Linear Programming.
    
    Args:
        rankings: List of rankings, each ranking is a list of items
        ground_set: Set of all possible items
        groups: Dictionary mapping items to their group (0 or 1)
        delta: Fairness threshold (smaller values enforce more fairness between groups)
    
    Returns:
        aggregated_ranking: The optimal aggregation
    """
    # Convert items to indices for ILP processing
    all_items = sorted(list(ground_set))
    
    # Create mapping from items to indices
    item_to_idx = {item: idx for idx, item in enumerate(all_items)}
    idx_to_item = {idx: item for idx, item in enumerate(all_items)}
    
    # Convert rankings to use indices
    idx_rankings = []
    for ranking in rankings:
        idx_ranking = [item_to_idx[item] for item in ranking]
        idx_rankings.append(idx_ranking)
    
    n_candidates = len(all_items)
    
    # Convert groups to array form using item indices
    group_array = np.zeros(n_candidates, dtype=int)
    for idx, item in enumerate(all_items):
        group_array[idx] = groups.get(item, 0)
    
    # Create optimization problem
    prob = pulp.LpProblem("Fair_Kemeny_Aggregation", pulp.LpMinimize)
    
    # Create binary variables for each pair (i,j)
    x = {}
    for i in range(n_candidates):
        for j in range(n_candidates):
            if i != j:  # No need for variables comparing an item to itself
                x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", 0, 1, 'Binary')
    
    # Add pairwise constraints (xij + xji = 1)
    for i, j in combinations(range(n_candidates), 2):
        prob += x[(i, j)] + x[(j, i)] == 1
    
    # Add transitivity constraints (xij + xjk - xik <= 1)
    for i, j, k in permutations(range(n_candidates), 3):
        prob += x[(i, j)] + x[(j, k)] - x[(i, k)] <= 1
    
    # Build objective function - minimize disagreements
    objective_terms = []
    
    for voter_ranking in idx_rankings:
        # Create a mapping from item to its position in this voter's ranking
        item_positions = {item: pos for pos, item in enumerate(voter_ranking)}
        
        # For each pair of items, add a penalty if the aggregated ranking disagrees
        for i in range(n_candidates):
            for j in range(n_candidates):
                if i != j and i in item_positions and j in item_positions:
                    if item_positions[i] < item_positions[j]:
                        objective_terms.append(x[(j, i)])  # Penalty for reversing order
                    else:
                        objective_terms.append(x[(i, j)])
    
    # Set the objective
    prob += pulp.lpSum(objective_terms)
    
    # Build parity constraints based on groups
    parity_terms = []
    for i in range(n_candidates):
        for j in range(n_candidates):
            if i != j:
                # If i and j are from different groups, add a term
                if group_array[i] != group_array[j]:
                    # +1 if item from group 1 is preferred over item from group 0
                    # -1 if item from group 0 is preferred over item from group 1
                    weight = group_array[i] - group_array[j]
                    parity_terms.append(weight * x[(i, j)])
    
    parity_sum = pulp.lpSum(parity_terms)
    
    # Calculate the adjusted delta threshold (assuming balanced groups)
    group_0_size = np.sum(group_array == 0)
    group_1_size = np.sum(group_array == 1)
    delta_adj = delta * (group_0_size * group_1_size)
    
    # Add upper and lower bounds for parity
    prob += parity_sum <= delta_adj    # Upper bound
    prob += parity_sum >= -delta_adj   # Lower bound
    
    # Solve the problem
    try:
        try:
            solver = pulp.GUROBI_CMD(msg=False)
        except:
            print("ERROR")
        prob.solve(solver)
        
        # Check if an optimal solution was found
        if prob.status == pulp.LpStatusOptimal:
            # Calculate scores
            scores = np.zeros(n_candidates)
            for i in range(n_candidates):
                for j in range(n_candidates):
                    if i != j and pulp.value(x[(i, j)]) > 0.5:  # i preferred to j
                        scores[i] += 1
            
            # Get aggregated ranking indices
            aggregated_idx = np.argsort(-scores).tolist()
            
            # Convert back to original items
            aggregated_ranking = [idx_to_item[idx] for idx in aggregated_idx]
            
            return aggregated_ranking
        else:
            return None
            
    except Exception as e:
        return None
        
def create_group_assignment(items, p=0.5):
    """
    Create a balanced binary group assignment for the given items with bias parameter p.
    
    Args:
        items: List of items to assign to groups
        p: Probability parameter (default=0.5)
           - p = 0.5 gives a fair random assignment
           - p > 0.5 biases toward putting items earlier in the list in group 1
           - p < 0.5 biases toward putting items earlier in the list in group 0
    
    Returns:
        Dictionary mapping each item to its group (0 or 1)
    """
    d = len(items)
    len1 = d // 2  # Number of items to put in group 1
    len0 = d - len1  # Number of items to put in group 0
    
    # Initialize empty group assignment
    groups = {}
    
    # Iterate through the items
    for item in items:
        # If we've filled either group completely, put the rest in the other group
        if len1 <= 0:
            groups[item] = 0
            len0 -= 1
        elif len0 <= 0:
            groups[item] = 1
            len1 -= 1
        # Otherwise, randomly assign based on probability p
        elif random() < p:
            groups[item] = 1
            len1 -= 1
        else:
            groups[item] = 0
            len0 -= 1
    
    return groups



















# # Example usage with {7, 3, 221, 12}
# if __name__ == "__main__":
#     # Example rankings using the items {7, 3, 221, 12}
#     # Each list represents one voter's preferences (first is most preferred)
#     rankings = [
#         [7, 3, 221, 12],    # Voter 1: prefers 7 > 3 > 221 > 12
#         [3, 7, 12, 221],    # Voter 2: prefers 3 > 7 > 12 > 221
#         [7, 12, 3, 221],    # Voter 3: prefers 7 > 12 > 3 > 221
#         [3, 221, 7, 12],    # Voter 4: prefers 3 > 221 > 7 > 12
#         [221, 12, 7, 3]     # Voter 5: prefers 221 > 12 > 7 > 3
#     ]
    
#     print("\nComputing Kemeny optimal aggregation...")
#     aggregated = kemeny_rank_aggregation(rankings)
    
#     print("\nKemeny optimal aggregation:")
#     print(aggregated)