import numpy as np
import time
import math

def uniform_sample_coreset(points, N=1000):
    """
    Construct a uniform random coreset by sampling points.

    Args:
        points: numpy array of shape (n, d) containing n d-dimensional points
        N: size of the coreset to construct

    Returns:
        coreset: numpy array of shape (N, d) containing the sampled points
        weights: uniform weights (all 1/N)
    """

    # Step 1: Randomly sample N points without replacement
    # If N is larger than the number of points, sample with replacement
    sampled_indices = np.random.choice(len(points), min(N, len(points)), replace=True)

    # Sample the points
    coreset = points[sampled_indices]

    # Create uniform weights
    weights = np.ones(N) / N

    return coreset, weights

def construct_strong_coreset_probabilities(points):
    n = len(points)

    # Step 1: Find approximate 1-median by sampling 100 points
    sample_indices = np.random.choice(n, min(1000, n), replace=True)
    samples = points[sample_indices]

    min_cost = float('inf')
    C_star = None

    for center in samples:
        # Using vectorized operations for speed
        distances = np.sum(np.abs(points - center), axis=1)
        cost = np.sum(distances)
        # print(cost)
        if cost < min_cost:
            min_cost = cost
            C_star = center

    # Vectorized computation of distances to C_star
    distances = np.sum(np.abs(points - C_star), axis=1)
    total_cost = np.sum(distances)
    sensitivities = distances / total_cost + 1/n

    # Step 3: Normalize sensitivities to probabilities
    probs = sensitivities / np.sum(sensitivities)
    # print(f"Sensitivities computed in {time.time() - start_time:.2f} seconds")

    # Step 4: Sample 100*N points according to these probabilities
    # print(f"Sampling {N} points...")
    first_sample_indices = np.random.choice(n, min(int(10_000 * math.log(n)), n), replace=True, p=probs)
    first_sample = points[first_sample_indices]
    first_weights = 1 / (len(first_sample_indices) * probs[first_sample_indices])

    # Second iteration
    # Find a better approximate 1-median using the first sample
    min_cost = float('inf')
    better_C_star = None

    for center in first_sample[np.random.choice(len(first_sample), min(1000, n), replace=True)]:
        # Using vectorized operations for weighted distance calculation
        distances = np.sum(np.abs(first_sample - center), axis=1)
        weighted_cost = np.sum(distances * first_weights)
        if weighted_cost < min_cost:
            min_cost = weighted_cost
            better_C_star = center

    # Compute new sensitivities for the first sample
    distances = np.sum(np.abs(first_sample - better_C_star), axis=1)
    total_weighted_cost = np.sum(distances * first_weights)
    
    # Compute the sum of weights for normalization
    total_weight = np.sum(first_weights)
    
    # New sensitivity formula: w_x*dist/cost + weight/sum(weights)
    new_sensitivities = (first_weights * distances / total_weighted_cost) + (first_weights / total_weight)
    
    # Normalize to get probabilities
    second_probs = new_sensitivities / np.sum(new_sensitivities)

    return second_probs, first_weights, first_sample
    

def construct_strong_coreset(points, N, probs=None, weights=None, first_sample=None):
    """
    Construct a coreset for the 1-median problem in L1 through 2-step sampling.
    Args:
        points: numpy array of shape (n, d) containing n d-dimensional points
        N: size of the coreset to construct

    Returns:
        coreset: numpy array of shape (N, d) containing the coreset points
        weights: numpy array of shape (N,) containing the weights
    """
    if probs is None:
        probs, weights, first_sample = construct_strong_coreset_probabilities(points)
    # Sample final N points from the first sample
    second_sample_indices = np.random.choice(
        len(first_sample), 
        N, 
        replace=True, 
        p=probs
    )
    
    # Final coreset and weights
    coreset = first_sample[second_sample_indices]
    
    # The weights need to account for both sampling steps
    final_weights = 1 / (N * probs[second_sample_indices])
    final_weights *= weights[second_sample_indices]  # Multiply by first stage weights
    
    return coreset, final_weights
        
    # n = len(points)

    # # Step 1: Find approximate 1-median by sampling 100 points
    # sample_indices = np.random.choice(n, min(1000, n), replace=True)
    # samples = points[sample_indices]

    # min_cost = float('inf')
    # C_star = None

    # for center in samples:
    #     # Using vectorized operations for speed
    #     distances = np.sum(np.abs(points - center), axis=1)
    #     cost = np.sum(distances)
    #     # print(cost)
    #     if cost < min_cost:
    #         min_cost = cost
    #         C_star = center

    # # Vectorized computation of distances to C_star
    # distances = np.sum(np.abs(points - C_star), axis=1)
    # total_cost = np.sum(distances)
    # sensitivities = distances / total_cost + 1/n

    # # Step 3: Normalize sensitivities to probabilities
    # probs = sensitivities / np.sum(sensitivities)
    # # print(f"Sensitivities computed in {time.time() - start_time:.2f} seconds")

    # # Step 4: Sample 100*N points according to these probabilities
    # # print(f"Sampling {N} points...")
    # first_sample_indices = np.random.choice(n, min(100*N, n), replace=True, p=probs)
    # first_sample = points[first_sample_indices]
    # first_weights = 1 / (len(first_sample_indices) * probs[first_sample_indices])

    # # Second iteration
    # # Find a better approximate 1-median using the first sample
    # min_cost = float('inf')
    # better_C_star = None

    # for center in first_sample[np.random.choice(len(first_sample), min(100, len(first_sample)), replace=False)]:
    #     # Using vectorized operations for weighted distance calculation
    #     distances = np.sum(np.abs(first_sample - center), axis=1)
    #     weighted_cost = np.sum(distances * first_weights)
    #     if weighted_cost < min_cost:
    #         min_cost = weighted_cost
    #         better_C_star = center

    # # Compute new sensitivities for the first sample
    # distances = np.sum(np.abs(first_sample - better_C_star), axis=1)
    # total_weighted_cost = np.sum(distances * first_weights)
    
    # # Compute the sum of weights for normalization
    # total_weight = np.sum(first_weights)
    
    # # New sensitivity formula: w_x*dist/cost + weight/sum(weights)
    # new_sensitivities = (first_weights * distances / total_weighted_cost) + (first_weights / total_weight)
    
    # # Normalize to get probabilities
    # second_probs = new_sensitivities / np.sum(new_sensitivities)
    
    # # Sample final N points from the first sample
    # second_sample_indices = np.random.choice(
    #     len(first_sample), 
    #     N, 
    #     replace=True, 
    #     p=second_probs
    # )
    
    # # Final coreset and weights
    # coreset = first_sample[second_sample_indices]
    
    # # The weights need to account for both sampling steps
    # final_weights = 1 / (N * second_probs[second_sample_indices])
    # final_weights *= first_weights[second_sample_indices]  # Multiply by first stage weights
    
    # return coreset, final_weights

def construct_strong_coreset_probabilities_one_iteration(points):
    n = len(points)

    # Step 1: Find approximate 1-median by sampling 1000 points
    sample_indices = np.random.choice(n, min(1000, n), replace=True)
    samples = points[sample_indices]

    min_cost = float('inf')
    C_star = None

    for center in samples:
        # Using vectorized operations for speed
        distances = np.sum(np.abs(points - center), axis=1)
        cost = np.sum(distances)
        # print(cost)
        if cost < min_cost:
            min_cost = cost
            C_star = center

    # Vectorized computation of distances to C_star
    distances = np.sum(np.abs(points - C_star), axis=1)
    total_cost = np.sum(distances)
    sensitivities = distances / total_cost + 1/n

    # Step 3: Normalize sensitivities to probabilities
    probs = sensitivities / np.sum(sensitivities)
    # print(f"Sensitivities computed in {time.time() - start_time:.2f} seconds")
    return probs

    # Step 4: Sample N points according to these probabilities
    # print(f"Sampling {N} points...")
    sampled_indices = np.random.choice(n, N, replace=True, p=probs)

    # Step 5: Reweight the sampled points
    coreset = points[sampled_indices]
    weights = 1 / (N * probs[sampled_indices])

    # print("Coreset construction complete")
    return coreset, weights


def construct_strong_coreset_one_iteration(points, N, probs=None):
    """
    Construct a coreset for the 1-median problem in L1 throught 2-step sampling.
    Args:
        points: numpy array of shape (n, d) containing n d-dimensional points
        N: size of the coreset to construct

    Returns:
        coreset: numpy array of shape (N, d) containing the coreset points
        weights: numpy array of shape (N,) containing the weights
    """    
    if probs is None:
        probs = construct_strong_coreset_probabilities_one_iteration(points)

    sampled_indices = np.random.choice(len(points), N, replace=True, p=probs)

    # Step 5: Reweight the sampled points
    coreset = points[sampled_indices]
    weights = 1 / (N * probs[sampled_indices])

    # print("Coreset construction complete")
    return coreset, weights
    
    # n = len(points)

    # # Step 1: Find approximate 1-median by sampling 1000 points
    # sample_indices = np.random.choice(n, min(1000, n), replace=True)
    # samples = points[sample_indices]

    # min_cost = float('inf')
    # C_star = None

    # for center in samples:
    #     # Using vectorized operations for speed
    #     distances = np.sum(np.abs(points - center), axis=1)
    #     cost = np.sum(distances)
    #     # print(cost)
    #     if cost < min_cost:
    #         min_cost = cost
    #         C_star = center

    # # Vectorized computation of distances to C_star
    # distances = np.sum(np.abs(points - C_star), axis=1)
    # total_cost = np.sum(distances)
    # sensitivities = distances / total_cost + 1/n

    # # Step 3: Normalize sensitivities to probabilities
    # probs = sensitivities / np.sum(sensitivities)
    # # print(f"Sensitivities computed in {time.time() - start_time:.2f} seconds")

    # # Step 4: Sample N points according to these probabilities
    # # print(f"Sampling {N} points...")
    # sampled_indices = np.random.choice(n, N, replace=True, p=probs)

    # # Step 5: Reweight the sampled points
    # coreset = points[sampled_indices]
    # weights = 1 / (N * probs[sampled_indices])

    # # print("Coreset construction complete")
    # return coreset, weights


def compute_median(points, weights=None):
    """
    Compute the weighted median of a coreset in ℓ₁ norm.
    
    Parameters:
    ----------
    points : np.ndarray
        Array of shape (m, d) where m is the number of points and d is the dimension.
    weights : np.ndarray
        Array of shape (m,) containing the weights of each point.
        Weights are assumed to be normalized (sum to 1).
    
    Returns:
    -------
    np.ndarray
        The weighted median point of shape (d,)
    """
    if len(points) == 0:
        raise ValueError("Empty points array")

    if weights is None:
        return np.median(points, axis=0)
    
    if len(points) != len(weights):
        raise ValueError(f"Mismatch between points ({len(points)}) and weights ({len(weights)})")
    
    # Normalize weights if they don't sum to 1
    if not np.isclose(np.sum(weights), 1.0):
        weights = weights / np.sum(weights)
    
    m, d = points.shape
    median_point = np.zeros(d)
    
    # Compute median for each coordinate
    for j in range(d):
        # Extract the j-th coordinate of all points
        coordinate_values = points[:, j]
        
        # Sort the values and rearrange weights accordingly
        sorted_indices = np.argsort(coordinate_values)
        sorted_values = coordinate_values[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        # Compute cumulative weights
        cumulative_weights = np.cumsum(sorted_weights)
        
        # Find the index k where cumulative weight crosses 0.5
        # This is the weighted median for this coordinate
        median_idx = np.searchsorted(cumulative_weights, 0.5, side='right')
        
        # Handle edge cases
        if median_idx >= m:
            median_idx = m - 1
            
        median_point[j] = sorted_values[median_idx]
    
    return median_point

def evaluate_coreset(original_points, coreset, weights=None):
    """
    Evaluate the quality of the coreset by comparing costs.

    Returns:
        Relative error of the coreset approximation
    """
    # Get the median of original points (coordinate-wise median)
    true_median = compute_median(original_points)
    original_cost = np.sum(np.sum(np.abs(original_points - true_median), axis=1))

    # Compute cost using coreset and weights
    coreset_median = compute_median(coreset, weights)
    coreset_cost = np.sum(np.sum(np.abs(original_points - coreset_median), axis=1))
    # Compute relative error
    relative_error = (coreset_cost - original_cost) / original_cost
    if relative_error < 0:
        print ("error: relative error is negative ", relative_error)

    return relative_error
