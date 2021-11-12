from typing import List, Tuple

from active_learning_ts.pool import Pool

import tensorflow as tf

try:
    import operator
except ImportError:
    key_fun = lambda x: x.lower
else:
    key_fun = operator.attrgetter("lower")


def loop_body(iterator, body_index, body_total_covered, body_next_total, body_correct_offset, body_size_list,
              body_start_value):
    body_total_covered = body_next_total
    body_next_total += tf.gather(body_size_list, iterator)
    body_correct_offset = tf.gather(body_start_value, iterator)

    return iterator + 1, body_index, body_total_covered, body_next_total, body_correct_offset, body_size_list, body_start_value


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
        self.sizes = []
        self.start_values = []

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
                self.total_sizes[j] += float(dim[i].size)
            self.total_sizes[j] += float(dim[len(dim) - 1].size)

        # data from range objects allowed to sort and check the data. We now need to convert it back to ints, so we
        # can work with tf
        for i in range(len(ranges)):
            self.sizes.append([])
            self.start_values.append([])
            for j in range(len(ranges[i])):
                current_range = self.ranges[i][j]
                self.sizes[i].append(float(current_range.size))
                self.start_values[i].append(float(current_range.lower))

    @tf.function
    def _get_element_normalized(self, element: tf.Tensor) -> tf.Tensor:
        """
        Gets a value #TODO: this doc
        :param element:
        :return:
        """
        # please do not try to read this code

        indices = tf.unstack(element)

        out = []

        # declare all variables at the beginning in order to avoid tf issues
        next_out = None
        start_value_list_iterator = iter(self.start_values)
        size_list_iterator = iter(self.sizes)
        total_size_iterator = iter(self.total_sizes)
        total_covered = 0.0
        next_total = 0.0
        correct_offset = 0.0

        for index in indices:
            index = index * next(total_size_iterator)

            total_covered = 0.0
            next_total = 0.0
            correct_offset = 0.0
            start_value_list = next(start_value_list_iterator)
            size_list = next(size_list_iterator)

            j, i, total_covered, y, correct_offset, l1, l2 = tf.while_loop(
                lambda j, i, x, y, z, l1, l2: tf.math.less_equal(y, i),
                loop_body,
                [0, index, total_covered, next_total, correct_offset, size_list, start_value_list],
                parallel_iterations=1)

            # at this point, we have located the correct range, we just need to find the percentage of this range in
            # which the given point lies

            next_out = index - total_covered + correct_offset

            out.append(next_out)

        return tf.stack(out)

    def _get_element(self, element: tf.Tensor) -> tf.Tensor:
        return element

    def _normalize(self, query_candidate):
        indices = tf.unstack(query_candidate)

        out = []

        # declare all variables at the beginning in order to avoid tf issues
        next_out = None
        start_value_list_iterator = iter(self.start_values)
        size_list_iterator = iter(self.sizes)
        total_size_iterator = iter(self.total_sizes)
        total_covered = 0.0
        next_total = 0.0
        correct_offset = 0.0

        for index in indices:
            total_covered = 0.0
            next_total = 0.0
            start_value_list = next(start_value_list_iterator)
            size_list = next(size_list_iterator)

            j, i, y, total_covered, correct_offset, l1, l2 = tf.while_loop(
                lambda j, i, x, y, z, l1, l2: tf.math.less(tf.gather(l1, j) + tf.gather(l2, j), index),
                loop_body,
                [0, index, total_covered, next_total, correct_offset, size_list, start_value_list],
                parallel_iterations=1)

            next_out = (index - tf.gather(start_value_list, j) + total_covered) / next(total_size_iterator)

            out.append(next_out)
        return tf.stack(out)
