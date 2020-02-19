import numpy as np

from networks import WeightAgnosticNetwork
from evolution import get_fronts, reproduce_tournament, rank_by_dominance, rank_by_fitness_score


def test_get_fronts():
    """Test the get_fronts function on a simple example."""

    # Points look like this (numbers in the graph are indices of the points):
    #
    #             ^
    #             | 0 1
    # objective 2 |   2 3
    #             |     4
    #             |-------->
    #             objective 1
    #
    # Therefore, fronts should be [1, 3] and [0, 2, 4].

    objective1 = [1, 2, 2, 3, 3]
    objective2 = [3, 3, 2, 2, 1]
    objectives = np.vstack([objective1, objective2]).T

    fronts = get_fronts(objectives)
    assert fronts == [[1, 3], [0, 2, 4]]


def test_ranking():
    """Test rank_by_dominance and rank_by_fitness_score with a simple example."""

    # Set example values so that there's one element per front.
    mean_rewards = np.array([1, 2, 3])
    max_rewards = np.array([1, 2, 3])
    complexities = np.array([3, 2, 1])
    # --> ranks should be [2, 1, 0] (i.e. first element is worst, last one is best)

    # Rank a few times so that we rank at least once by mean_reward/max_reward and once by
    # mean_reward/complexity.
    for _ in range(50):
        ranks_dominance = rank_by_dominance(mean_rewards, max_rewards, complexities)
        assert np.all(ranks_dominance == np.array([2, 1, 0]))

    ranks_fitness_score = rank_by_fitness_score(mean_rewards, max_rewards, complexities)
    assert np.all(ranks_fitness_score == np.array([2, 1, 0]))


def test_reproduce_tournament():
    """Test reproduce_tournament with a specific population and seeding."""
    # TODO: Make this not rely on seeds.
    # Seeding network init is critical, otherwise seeding the evolution doesn't work because of
    # mutations.
    np.random.seed(0)
    population = [WeightAgnosticNetwork(10, 2, 0.5, 0.5, 0.25, 0.25) for _ in range(10)]
    for i, net in enumerate(population):
        net.index = i
    ranks = [5, 3, 8, 1, 4, 7, 9, 2, 0, 6]

    # Without culling and elitism.
    np.random.seed(0)
    new_population = reproduce_tournament(population, ranks, 3, cull_ratio=0, elite_ratio=0)
    assert np.all(np.array([net.index for net in new_population])
                  == np.array([7, 1, 7, 4, 3, 8, 9, 3, 1, 1]))

    # With culling and elitism.
    np.random.seed(0)
    new_population = reproduce_tournament(population, ranks, 3, cull_ratio=0.2, elite_ratio=0.2)
    assert np.all(np.array([net.index for net in new_population])
                  == np.array([8, 3, 3, 1, 7, 4, 0, 7, 7, 3]))
