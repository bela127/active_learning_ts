from typing import List, Tuple

import numpy as np
import tensorflow as tf

from active_learning_ts.data_retrievement.pool import Pool

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

    def __init__(self, dim, ranges: Tuple[Tuple[Tuple[float, float],...],...]):
        """
        Constructor takes a list of multiple ranges in each dimension.
        The ranges are independent in each dimension, meaning the validity of a value in one dimension cannot
        depend on the value of one of the other elements.

        :param dim: The number of dimensions
        :param ranges: a list of tuples. These define the ranges within which valid queries can be made. An input of the
            form (a,b), translates to the range [a.b). The range does not include the second element.
            If b > a, both (b,a) and (a,b) will translate to the range [a,b). If there are multiple valid ranges per
            dimension, then the ranges need not be ordered.

        """
        self.shape = (dim,)
        self.ranges = ranges
        self.con_ranges = []
        self.sizes: List = []
        self.start_values: List = []

        for dimension in ranges:
            self.con_ranges.append([ContinuousVectorPool.Range(a, b) for a, b in dimension])

        # very fast in place sort
        [dim.sort(key=key_fun) for dim in self.con_ranges]

        self.total_sizes = []
        # sanity check
        for j in range(len(self.con_ranges)):
            dim = self.con_ranges[j]
            self.total_sizes.append(.0)
            for i in range(len(dim) - 1):
                if dim[i].intersects(dim[i + 1]):
                    raise ValueError("Intersection of intervals must be empty")
                self.total_sizes[j] += float(dim[i].size)
            self.total_sizes[j] += float(dim[len(dim) - 1].size)

        # data from range objects is used to sort and check the data. We now need to convert it back to ints, so we
        # can work with tf
        for i in range(len(ranges)):
            self.sizes.append([])
            self.start_values.append([])
            for j in range(len(ranges[i])):
                current_range = self.con_ranges[i][j]
                self.sizes[i].append(float(current_range.size))
                self.start_values[i].append(float(current_range.lower))

    @tf.function
    def get_element_normalized(self, element: tf.Tensor) -> tf.Tensor:
        """
        The ranges are normalised by removing any gaps between ranges and then mapping the values onto the interval
        [0,1).
        This function un-normalizes a given vector.

        :param element: a tensor with entries in the range [0,1)
        :return: The un-normalized vector
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

        return tf.reshape(tf.stack(out), (1, len(out)))

    def get_elements(self, element: tf.Tensor) -> tf.Tensor:
        """
        For efficiency reasons, and due to current use-cases, it is assumed that the input is valid.
        The output batch is then just this one element

        :param element: the element that should be checked
        :return: A batch of queries that consists solely of the input element
        """
        return tf.reshape(element, (1, element.shape[0], element.shape[1]))

    @tf.function
    def _normalize(self, query_candidate: tf.Tensor) -> tf.Tensor:
        """
        The ranges are normalised by removing any gaps between ranges and then mapping the values onto the interval
        [0,1).

        :param query_candidate: The element to be normalised
        :return: The normalised vector
        """
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

    # tf.function
    def is_valid(self, point) -> bool:
        start_value_list_iterator = iter(self.start_values)
        size_list_iterator = iter(self.sizes)
        indices = tf.unstack(point)
        size_tensor = []
        start_tensor = []

        for index in indices:
            size_iterator = reversed(next(size_list_iterator))
            start_value_list = reversed(next(start_value_list_iterator))

            i, st, si = tf.while_loop(lambda i, st, si: tf.reduce_all(tf.less_equal(i, st)),
                                      lambda i, st, si: (
                                          i, next(start_value_list, i - 1.0), next(size_iterator, -np.Inf)),
                                      [index, next(start_value_list), next(size_iterator)])
            start_tensor.append(st)
            size_tensor.append(si)
        start_tensor = tf.stack(start_tensor)
        size_tensor = tf.stack(size_tensor)

        return tf.reduce_all(point < (start_tensor + size_tensor))
