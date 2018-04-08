import logging
import sys
from unittest import TestCase

import numpy as np

from deepsnippet.data_generator import MinibatchGenerator

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


class TestMinibatchGenerator(TestCase):

    def test_wo_sort_batch(self):
        X = []

        batch_size = 3
        max_len = 9

        for _ in range(2):
            X.append(np.arange(10))

        for _ in range(4):
            X.append(np.arange(5))

        for _ in range(1):
            X.append(np.arange(7))

        for _ in range(3):
            X.append(np.arange(8))

        Y = np.repeat(np.arange(10).reshape((10, 1)), 3, axis=1)

        np.random.seed(123)
        sut = MinibatchGenerator(X, Y, batch_size=batch_size, max_len=max_len, choice_on_epoch=True,
                                 sort_batch=False)

        expected_X = [
            np.repeat(np.arange(5).reshape((1, 5)), 3, axis=0),
            np.vstack([np.hstack([np.zeros(1), np.arange(7)]), np.arange(8), np.arange(8)]),
            np.vstack([np.hstack([np.zeros(1), np.arange(8)]).reshape((1, 9)),
                       np.arange(1, 10).reshape((1, 9)), np.arange(1, 10).reshape((1, 9))]),

            np.repeat(np.arange(5).reshape((1, 5)), 3, axis=0),
            np.vstack([np.hstack([np.zeros(3), np.arange(5)]), np.hstack([np.zeros(1), np.arange(7)]),
                       np.arange(8).reshape((1, 8))]),
            np.vstack([np.hstack([np.zeros(1), np.arange(8)]).reshape((1, 9)),
                       np.arange(1, 10).reshape((1, 9)), np.arange(1, 10).reshape((1, 9))]),
        ]

        expected_Y = [
            np.vstack([np.ones((1, 3)) * 2, np.ones((1, 3)) * 3, np.ones((1, 3)) * 5]),
            np.vstack([np.ones((1, 3)) * 6, np.ones((1, 3)) * 7, np.ones((1, 3)) * 8]),
            np.vstack([np.ones((1, 3)) * 9, np.ones((1, 3)) * 0, np.ones((1, 3)) * 1]),
            np.vstack([np.ones((1, 3)) * 2, np.ones((1, 3)) * 3, np.ones((1, 3)) * 4]),
            np.vstack([np.ones((1, 3)) * 5, np.ones((1, 3)) * 6, np.ones((1, 3)) * 8]),
            np.vstack([np.ones((1, 3)) * 9, np.ones((1, 3)) * 0, np.ones((1, 3)) * 1])
        ]

        self.assertEqual(sut.steps_per_epoch, len(expected_X) // 2)

        for idx in range(sut.steps_per_epoch):
            if idx >= sut.steps_per_epoch:
                break
            batch = sut.__next__()
            np.testing.assert_array_equal(batch[0], expected_X[idx])
            np.testing.assert_array_equal(batch[1], expected_Y[idx])

            print(batch)

        np.random.seed(456)
        for idx in range(sut.steps_per_epoch):
            if idx >= sut.steps_per_epoch:
                break

            batch = sut.__next__()
            np.testing.assert_array_equal(batch[0], expected_X[idx + 3])
            np.testing.assert_array_equal(batch[1], expected_Y[idx + 3])

            print(batch)

    def test_w_sort_batch(self):
        X = []

        batch_size = 3
        max_len = 9

        for _ in range(2):
            X.append(np.arange(10))

        for _ in range(4):
            X.append(np.arange(5))

        for _ in range(1):
            X.append(np.arange(7))

        for _ in range(3):
            X.append(np.arange(8))

        Y = np.repeat(np.arange(10).reshape((10, 1)), 3, axis=1)

        np.random.seed(123)
        sut = MinibatchGenerator(X, Y, batch_size=batch_size, max_len=max_len, choice_on_epoch=True,
                                 sort_batch=True)

        expected_X = [
            np.repeat(np.arange(5).reshape((1, 5)), 3, axis=0),
            np.vstack([np.hstack([np.zeros(1), np.arange(7)]), np.arange(8), np.arange(8)]),
            np.vstack([np.hstack([np.zeros(1), np.arange(8)]).reshape((1, 9)),
                       np.arange(1, 10).reshape((1, 9)), np.arange(1, 10).reshape((1, 9))]),

            np.vstack([np.hstack([np.zeros(1), np.arange(8)]).reshape((1, 9)),
                       np.arange(1, 10).reshape((1, 9)), np.arange(1, 10).reshape((1, 9))]),
            np.vstack([
                np.hstack([np.zeros(1), np.arange(7)]),
                np.hstack([np.zeros(3), np.arange(5)]),
                np.arange(8).reshape((1, 8))
            ]),
            np.repeat(np.arange(5).reshape((1, 5)), 3, axis=0),
        ]

        expected_Y = [
            np.vstack([np.ones((1, 3)) * 2, np.ones((1, 3)) * 3, np.ones((1, 3)) * 5]),
            np.vstack([np.ones((1, 3)) * 6, np.ones((1, 3)) * 7, np.ones((1, 3)) * 8]),
            np.vstack([np.ones((1, 3)) * 9, np.ones((1, 3)) * 0, np.ones((1, 3)) * 1]),

            np.vstack([np.ones((1, 3)) * 9, np.ones((1, 3)) * 1, np.ones((1, 3)) * 0]),
            np.vstack([np.ones((1, 3)) * 6, np.ones((1, 3)) * 5, np.ones((1, 3)) * 8]),
            np.vstack([np.ones((1, 3)) * 3, np.ones((1, 3)) * 4, np.ones((1, 3)) * 2])
        ]

        self.assertEqual(sut.steps_per_epoch, len(expected_X) // 2)

        for idx in range(sut.steps_per_epoch):
            if idx >= sut.steps_per_epoch:
                break
            batch = sut.__next__()
            np.testing.assert_array_equal(batch[0], expected_X[idx])
            np.testing.assert_array_equal(batch[1], expected_Y[idx])

            print(batch)

        np.random.seed(456)
        for idx in range(sut.steps_per_epoch):
            if idx >= sut.steps_per_epoch:
                break

            batch = sut.__next__()
            np.testing.assert_array_equal(batch[0], expected_X[idx + 3])
            np.testing.assert_array_equal(batch[1], expected_Y[idx + 3])

            print(batch)
