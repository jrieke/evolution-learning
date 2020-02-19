import numpy as np
import copy
import warnings

from networks import WeightLearningNetwork


def sorted_by_values(list_, values, reverse=False):
    """Sort a list by the values in another list."""
    return [x for _, x in sorted(zip(values, list_), reverse=reverse)]


def reproduce_tournament(population, ranks, tournament_size, new_population_size=None, cull_ratio=0,
                         elite_ratio=0, num_mutations=1):
    """
    Create new population via tournament selection based on ranks.

    Procedure:
        - Sort all individuals by rank
        - Eliminate lower percentage of individuals from breeding pool (culling)
        - Pass upper percentage of individuals to child population unchanged (elitism)
        - Select parents by tournament selection
        - Produce new population through mutation

    Args:
        population (iterable): The old population.
        ranks (iterable, same size as population): Rank of each individual (e.g. obtained via pareto
            ranking).
        tournament_size (int): The size of each tournament to determine a parent network.
        new_population_size (int): Size of the new poulation (if None (default), this is the old
            population size).
        cull_ratio (float): Which fraction of the worst networks to leave out from the breeding
            pool.
        elite_ratio (float): Which fraction of the best networks to pass directly w/o mutation to
            the new population.
        num_mutations (int, optional): The number of mutations to carry out on each child
            (default: 1).

    Returns:
        list: The new population.
    """
    if tournament_size > len(population):
        raise ValueError(
            f"tournament_size ({tournament_size}) needs to be smaller or equal to the size of the "
            f"population ({len(population)})")

    if new_population_size is None:
        new_population_size = len(population)

    new_population = []

    # Sort the population by rank.
    sorted_population = sorted_by_values(population, ranks)

    # Culling - remove the worst performing individuals.
    num_cull = int(len(population) * cull_ratio)
    if num_cull > 0:
        del sorted_population[-num_cull:]

    # Elitism - move the best performing individuals directly to the new population,
    # while deleting their torch_layers before.
    num_elite = int(len(population) * elite_ratio)
    for net in sorted_population[:num_elite]:
        net = copy.deepcopy(net)
        if type(net) == WeightLearningNetwork:
            net.delete_torch_layers()
        new_population.append(net)

    for i in range(new_population_size - num_elite):
        # Tournament selection: Select some random individuals for the tournament and choose the
        # best one as parent.
        tournament_indices = np.random.choice(len(sorted_population), size=tournament_size,
                                              replace=False)
        parent_index = np.min(
            tournament_indices)  # sorted_population is sorted by ranks, so just pick the first one
        parent = sorted_population[parent_index]

        # Create the child by copying the parent and mutating.
        child = copy.deepcopy(parent)
        for _ in range(num_mutations):
            child.mutate()
        new_population.append(child)

    return new_population


def rank_by_fitness_score(mean_rewards, max_rewards, complexities, alpha_mean=1, alpha_max=1,
                          alpha_complexity=1):
    """
    NOTE: THIS IS NOT TESTED PROPERLY.

    Ranks elements by combining all metrics into a single objective (the fitness score).

    The fitness score is: alpha_mean * mean_rewards + alpha_max * max_rewards - alpha_complexity
    * complexities. The element with the highest fitness score has rank 0, the element with the
    second-highest fitness score has rank 1, etc.

    Returns:
        ranks: list of length population size, where each element i indicates 
            the relative rank position of the network i, i.e. ranks[3] 
            corresponds to the rank position of the network 3
    """
    fitness_scores = alpha_mean * mean_rewards + alpha_max * max_rewards - alpha_complexity \
                     * complexities
    fitness_scores = fitness_scores.astype(float)
    ranks = np.zeros(len(mean_rewards), dtype=int)
    for i in range(len(mean_rewards)):
        best_element = np.argmax(fitness_scores)
        ranks[best_element] = i
        fitness_scores[best_element] = -np.inf
    return ranks


def rank_by_dominance(mean_rewards, max_rewards, complexities, p_complexity_objective=0.8):
    """
    Ranks elements by dominance relations on multiple objectives, similar to NSGA-II
    (Deb et al. 2002).

    Elements are compared pairwise based on two objectives (mean reward and complexity with 80 %
    chance, mean reward and max reward with 20 % chance). An element is dominant, if it is not worse
    than the other element for both objectives and better for at least one objective. Based on these
    dominance relations, elements are sorted into pareto fronts, and their crowding distances within
    the front are calculated. Ranks are given based on 1) the front, 2) the crowding distance. The
    element of front 1 with the highest crowding distance gets rank 0, the element of front 1 with
    the second-highest crowding distance gets rank 1, etc.
    """
    if np.random.rand() < p_complexity_objective:
        values1 = mean_rewards
        values2 = 1 / complexities
        # logging.info('Using mean reward and complexity for ranking')
    else:
        values1 = mean_rewards
        values2 = max_rewards
        # logging.info('Using mean reward and max reward for ranking')

    values = np.vstack([values1, values2]).T
    return nsga_sort(values)


def nsga_sort(objective_values, return_fronts=False):
    """
    Return ranking of objective values based on non-dominated sorting.
    Optionally return fronts (useful for visualization).

    Note: Assumes maximization of objective function

    Args:
        objective_values (numpy array of shape [num_individuals, num objectives]): Objective values
            of each individual.
        return_fronts (boolean): Whether to return the fronts or only ranks.

    Returns:
        numpy array of shape [num individuals, 1]: Rank in population of each individual
        numpy array of shape [num_individuals, 1]: Pareto front of each individual

    From: https://github.com/google/brain-tokyo-workshop/blob/master/WANNRelease/WANN/wann_src/nsga_sort.py
    """
    # Sort by dominance into fronts
    fronts = get_fronts(objective_values)

    # Rank each front by crowding distance
    for f in range(len(fronts)):
        x1 = objective_values[fronts[f], 0]
        x2 = objective_values[fronts[f], 1]
        crowdDist = get_crowding_dist(x1) + get_crowding_dist(x2)
        frontRank = np.argsort(-crowdDist)
        fronts[f] = [fronts[f][i] for i in frontRank]

    # Convert to ranking
    tmp = [ind for front in fronts for ind in front]
    rank = np.empty_like(tmp)
    rank[tmp] = np.arange(len(tmp))

    if return_fronts is True:
        return rank, fronts
    else:
        return rank


def get_fronts(objective_values):
    """
    Fast non-dominated sort.

    Args:
        objective_values (numpy array of shape [num individuals, num objectives]): Objective values
            of each individual.

    Returns:
        list: Each element is one list for each front with the indices of individuals in this front

    From: https://github.com/google/brain-tokyo-workshop/blob/master/WANNRelease/WANN/wann_src/nsga_sort.py
    ...which was adapted from: https://github.com/haris989/NSGA-II]
    """
    values1 = objective_values[:, 0]
    values2 = objective_values[:, 1]

    S = [[] for i in range(0, len(values1))]
    front = [[]]
    n = [0 for i in range(0, len(values1))]
    rank = [0 for i in range(0, len(values1))]

    # Get dominance relations
    for p in range(0, len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) \
                    or (values1[p] >= values1[q] and values2[p] > values2[q]) \
                    or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) \
                    or (values1[q] >= values1[p] and values2[q] > values2[p]) \
                    or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    # Assign fronts
    i = 0
    while (front[i] != []):
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if (n[q] == 0):
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)
    del front[len(front) - 1]
    return front


def get_crowding_dist(objective_vector):
    """
    Return crowding distance of a vector of values, used once on each front.

    Note: Crowding distance of individuals at each end of front is infinite, as they don't have a
        neighbor.

    Args:
        objective_vector (numpy array of shape [num individuals]): Objective value of each
            individual.

    Returns:
        numpy array of shape [num individuals, 1]: Crowding distance of each individual.
    """
    # Order by objective value
    key = np.argsort(objective_vector)
    sortedObj = objective_vector[key]

    # Distance from values on either side
    shiftVec = np.r_[np.inf, sortedObj, np.inf]  # Edges have infinite distance
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # inf on purpose
    prevDist = np.abs(sortedObj - shiftVec[:-2])
    nextDist = np.abs(sortedObj - shiftVec[2:])
    crowd = prevDist + nextDist
    if (sortedObj[-1] - sortedObj[0]) > 0:
        crowd *= abs((1 / sortedObj[-1] - sortedObj[0]))  # Normalize by fitness range

    # Restore original order
    dist = np.empty(len(key))
    dist[key] = crowd[:]

    return dist
