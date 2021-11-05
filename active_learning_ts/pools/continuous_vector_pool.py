from typing import List, Tuple

from active_learning_ts.pool import Pool

import tensorflow as tf

try:
    import operator
except ImportError:
    key_fun = lambda x: x.lower
else:
    key_fun = operator.attrgetter("lower")


class ContinuousVectorPool(Pool):
    class Range:
        def __init__(self, a: float, b: float):
            self.lower = min(a, b)
            self.upper = max(a, b)

            self.size = self.upper - self.lower

        def intersects(self, param):
            return self.upper > param.lower

        def get_at(self, interval: float):
            return self.lower + (self.size * interval)

    def __init__(self, dim, ranges: List[List[Tuple[float, float]]]):
        self.shape = (dim,)
        self.ranges = []

        for dimension in ranges:
            self.ranges.append([ContinuousVectorPool.Range(a, b) for a, b in dimension])

        # very fast in place sort
        [dim.sort(key=key_fun) for dim in self.ranges]

        self.total_sizes = []
        # sanity check
        for j in range(len(self.ranges)):
            dim = self.ranges[j]
            self.total_sizes.append(.0)
            for i in range(len(dim) - 1):
                if dim[i].intersects(dim[i + 1]):
                    raise ValueError("Intersection of intervals must be empty")
                self.total_sizes[j] += dim[i].size
            self.total_sizes[j] += dim[len(dim) - 1].size

    #@tf.function
    def get_element(self, element: tf.Tensor) -> tf.Tensor:
        indices = tf.unstack(element)

        out = []

        # avoids tf issues
        next_out = None

        for i in range(0, len(indices)):
            index = indices[i] * self.total_sizes[i]

            total_covered = 0.0
            next_total = 0.0
            correct_range = None
            for current_range in self.ranges[i]:
                next_total += current_range.size
                if next_total >= index:
                    correct_range = current_range
                    break
                total_covered = next_total

            # at this point, we have located the correct range, we just need to find the percentage of this range in
            # which the given point lies

            next_out = index - total_covered + correct_range.lower

            out.append(next_out)

        return tf.stack(out)
