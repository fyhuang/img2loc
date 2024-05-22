"""Weighted reservoir sampling with different algorithms."""

from typing import Any

import random
import heapq

class EfraimidisSpirakisSampler:
    """Efraimidis-Spirakis algorithm for weighted reservoir sampling."""
    def __init__(self, k: int):
        """Args:
            k: The number of samples to return.
        """
        self.k = k

        # A heapq, storing (rval, element), returning the min weight
        self.reservoir = []

    def observe(self, element: Any, weight: float):
        """Sample the element with the given weight."""
        if weight < 1e-10:
            # Effectively a zero weight, should never be sampled
            return

        rval = random.random() ** (1.0 / weight)
        if len(self.reservoir) < self.k:
            heapq.heappush(self.reservoir, (rval, element))
        else:
            # Keep the k items with largest rvals
            if rval > self.reservoir[0][0]:
                heapq.heappop(self.reservoir)
                heapq.heappush(self.reservoir, (rval, element))

    def get_samples(self):
        """Get the stored samples."""
        return [r[1] for r in self.reservoir]


def main():
    """Run tests to see if the sampling works."""
    import argparse
    import collections

    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("-n", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--unique", type=int, default=50)
    args = parser.parse_args()

    # Generate the unique elements with weights
    elements = [
        (e, random.randint(0, args.unique)) for e in range(args.unique)
    ]

    picks = collections.Counter()
    for r in range(args.rounds):
        sampler = EfraimidisSpirakisSampler(args.k)
        for i in range(args.n):
            e = elements[i % len(elements)]
            sampler.observe(e[0], e[1])

        for e in sampler.get_samples():
            picks[e] += 1

    # Compare normalized weights to sampled distribution
    total_weight = sum(e[1] for e in elements)
    expected = {e[0]: e[1] / total_weight for e in elements}
    actual = {e: times_picked / (args.k * args.rounds) for e, times_picked in picks.items()}
    print("Num\tExpected\tActual")
    for etuple in elements:
        e = etuple[0]
        print(f"{e}\t{expected.get(e, 0):.3f}\t{actual.get(e, 0):.3f}")


if __name__ == "__main__":
    main()