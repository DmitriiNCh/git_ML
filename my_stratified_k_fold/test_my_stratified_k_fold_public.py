import numpy as np
import pytest
from sklearn.utils._testing import assert_array_equal
from collections import OrderedDict
from my_stratified_k_fold import MyStratifiedKFold


def test_folds_more_than_samples():
    with pytest.raises(ValueError):
        list(MyStratifiedKFold(n_splits=10).split(list(range(5)),
                                                  list(range(5))))


class Case(object):
    def __init__(self, n_splits, X, Y, expected):
        self.n_splits = n_splits
        self.X = X
        self.Y = Y
        self.expected = expected


TEST_CASES = OrderedDict([
    (
        "test_case_0",
        Case(
            n_splits=2,
            X=[1, 2],
            Y=[0, 0],
            expected=[
                ([0], [1]),
                ([1], [0]),
            ]
        )
    ),
    (
        "test_case_1",
        Case(
            n_splits=2,
            X=[1, 2, 3, 4],
            Y=[0, 0, 1, 1],
            expected=[
                ([0, 2], [1, 3]),
                ([1, 3], [0, 2]),
            ]
        )
    ),
    (
        "test_case_2",
        Case(
            n_splits=3,
            X=[1, 2, 3],
            Y=[0, 0, 0],
            expected=[([1, 2], [0]),
                      ([0, 2], [1]),
                      ([0, 1], [2])]
        )
    ),
    (
        "test_case_3",
        Case(
            n_splits=3,
            X=[1, 2, 3, 4],
            Y=[0, 0, 0, 0],
            expected=[([2, 3], [0, 1]),
                      ([0, 1, 3], [2]),
                      ([0, 1, 2], [3])]
        )
    )
])


@pytest.mark.parametrize(
    'test_case',
    TEST_CASES.values(),
    ids=list(TEST_CASES.keys())
)
def test_main(test_case):
    result = list(MyStratifiedKFold(n_splits=test_case.n_splits).split(test_case.X, test_case.Y))
    expected_copy = test_case.expected.copy()

    try:
        assert len(test_case.expected) == len(result), \
            'Expected len: {}, got: {}'.format(len(test_case.expected), len(result))

        for fold in result:
            x_train_fold, x_test_fold = fold
            was_matched: bool = False

            for ex in expected_copy:
                x_train_expected, x_test_expected = ex

                if x_train_fold == x_train_expected and x_test_fold == x_test_expected:
                    was_matched = True
                    expected_copy.remove(ex)
                    break

            assert was_matched, f'Your fold:\n {fold} was not matched'

        assert len(expected_copy) == 0, \
            f'{expected_copy} was/were not returned by MyStratifiedKFold'
    except:
        print('\nExpected:\n')
        for i in test_case.expected:
            print('{}'.format(i))

        print('\nReceived:\n')
        for i in result:
            print('{}'.format(i))

        raise


def test_stratified_kfold_2splits_shapes():
    X, y = np.ones(13).tolist(), np.ones(13).tolist()
    splits = MyStratifiedKFold(n_splits=2).split(X, y)
    train, test = next(splits)
    assert_array_equal((len(test), len(train)), (7, 6))
    train, test = next(splits)
    assert_array_equal((len(test), len(train)), (6, 7))


def test_stratified_kfold_3splits_shapes():
    X, y = np.ones(13).tolist(), np.ones(13).tolist()
    splits = MyStratifiedKFold(n_splits=3).split(X, y)
    train, test = next(splits)
    assert_array_equal((len(test), len(train)), (5, 8))
    train, test = next(splits)
    assert_array_equal((len(test), len(train)), (4, 9))
    train, test = next(splits)
    assert_array_equal((len(test), len(train)), (4, 9))


def test_stratified_kfold_4splits_shapes():
    X, y = np.ones(13).tolist(), np.ones(13).tolist()
    splits = MyStratifiedKFold(n_splits=4).split(X, y)
    train, test = next(splits)
    assert_array_equal((len(test), len(train)), (4, 9))
    train, test = next(splits)
    assert_array_equal((len(test), len(train)), (3, 10))
    train, test = next(splits)
    assert_array_equal((len(test), len(train)), (3, 10))
    train, test = next(splits)
    assert_array_equal((len(test), len(train)), (3, 10))


def test_stratified_kfold_5splits_shapes():
    X, y = np.ones(13).tolist(), np.ones(13).tolist()
    splits = MyStratifiedKFold(n_splits=5).split(X, y)
    train, test = next(splits)
    assert_array_equal((len(test), len(train)), (3, 10))
    train, test = next(splits)
    assert_array_equal((len(test), len(train)), (3, 10))
    train, test = next(splits)
    assert_array_equal((len(test), len(train)), (3, 10))
    train, test = next(splits)
    assert_array_equal((len(test), len(train)), (2, 11))
    train, test = next(splits)
    assert_array_equal((len(test), len(train)), (2, 11))


def test_stratified_kfold_2splits_binary():
    X, y = np.ones(13).tolist(), np.concatenate((np.zeros(9), np.ones(4))).tolist()
    splits = MyStratifiedKFold(2).split(X, y)
    train, test = next(splits)
    assert_array_equal(test, [0, 1, 2, 3, 4, 9, 10])
    assert_array_equal(train, [5, 6, 7, 8, 11, 12])


def test_stratified_kfold_3splits_binary():
    X, y = np.ones(13).tolist(), np.concatenate((np.zeros(9), np.ones(4))).tolist()
    splits = MyStratifiedKFold(3).split(X, y)
    train, test = next(splits)
    assert_array_equal(test, [0, 1, 2, 9, 10])
    assert_array_equal(train, [3, 4, 5, 6, 7, 8, 11, 12])
    train, test = next(splits)
    assert_array_equal(test, [3, 4, 5, 11])
    assert_array_equal(train, [0, 1, 2, 6, 7, 8, 9, 10, 12])
    train, test = next(splits)
    assert_array_equal(test, [6, 7, 8, 12])
    assert_array_equal(train, [0, 1, 2, 3, 4, 5, 9, 10, 11])


def test_stratified_kfold_4splits_binary():
    X, y = np.ones(13).tolist(), np.concatenate((np.zeros(9), np.ones(4))).tolist()
    splits = MyStratifiedKFold(4).split(X, y)
    train, test = next(splits)
    assert_array_equal(test, [0, 1, 2, 9])
    assert_array_equal(train, [3, 4, 5, 6, 7, 8, 10, 11, 12])
    train, test = next(splits)
    assert_array_equal(test, [3, 4, 10])
    assert_array_equal(train, [0, 1, 2, 5, 6, 7, 8, 9, 11, 12])
    train, test = next(splits)
    assert_array_equal(test, [5, 6, 11])
    assert_array_equal(train, [0, 1, 2, 3, 4, 7, 8, 9, 10, 12])
    train, test = next(splits)
    assert_array_equal(test, [7, 8, 12])
    assert_array_equal(train, [0, 1, 2, 3, 4, 5, 6, 9, 10, 11])


def test_stratified_kfold_5splits_binary():
    X, y = np.ones(13).tolist(), np.concatenate((np.zeros(9), np.ones(4))).tolist()
    splits = MyStratifiedKFold(5).split(X, y)
    train, test = next(splits)
    assert_array_equal(test, [0, 1, 9])
    assert_array_equal(train, [2, 3, 4, 5, 6, 7, 8, 10, 11, 12])
    train, test = next(splits)
    assert_array_equal(test, [2, 3, 10])
    assert_array_equal(train, [0, 1, 4, 5, 6, 7, 8, 9, 11, 12])
    train, test = next(splits)
    assert_array_equal(test, [4, 5, 11])
    assert_array_equal(train, [0, 1, 2, 3, 6, 7, 8, 9, 10, 12])
    train, test = next(splits)
    assert_array_equal(test, [6, 7])
    assert_array_equal(train, [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12])
    train, test = next(splits)
    assert_array_equal(test, [8, 12])
    assert_array_equal(train, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11])


def test_stratified_kfold_2splits_binary():
    X, y = np.ones(22).tolist(), [0] * 11 + [1] * 7 + [2] * 4
    splits = MyStratifiedKFold(2).split(X, y)

    train, test = next(splits)
    assert_array_equal(test, [
        0, 1, 2, 3, 4, 5,
        11, 12, 13,
        18, 19,
    ])
    assert_array_equal(train, [
        6, 7, 8, 9, 10,
        14, 15, 16, 17,
        20, 21,
    ])

    train, test = next(splits)
    assert_array_equal(test, [
        6, 7, 8, 9, 10,
        14, 15, 16, 17,
        20, 21,
    ])
    assert_array_equal(train, [
        0, 1, 2, 3, 4, 5,
        11, 12, 13,
        18, 19,
    ])
