from posixpath import split
from tkinter import Y
import typing as ty


class MyStratifiedKFold(object):
    def __init__(self,
                 n_splits: int = 5,
                 shuffle: bool = False,
                 random_state: int = None) -> ty.NoReturn:
        """
        Parameters
            ----------
            n_splits : Number of folds. Must be at least 2.
            shuffle & random_state - for compatibility, don't use it so far
        """
        self.n_splits: int = n_splits

    @property
    def n_splits(self):
        return self._n_splits

    @n_splits.setter
    def n_splits(self, value):
        self._n_splits = value

    def get_n_splits(self, X, y):
        return self._n_splits

    

    def split(self, X: ty.List[ty.Any], y: ty.List[ty.Any], _: ty.Any = None) -> \
            ty.Generator[ty.Tuple[ty.List[ty.Any], ty.List[ty.Any]], None, None]:
        """Generate indices to split data into training and test set.
            Parameters
            ----------
            X : List of shape (n_samples,), int/float
                Training data, where n_samples is the number of samples

            y : array-like of shape (n_samples,), int
                The target variable
            Yields
            ------
            train : List
                The training set indices for that split.
            test : List
                The testing set indices for that split.
        """
        class_to_indices: ty.Dict[int, ty.List[int]] #= #<YOUR CODE IS HERE>
        if self._n_splits > len(y):
            raise ValueError

        def listmerge(lstlst):
                all_=[]
                for lst in lstlst:
                    all_=all_+lst
                return all_

        for split_idx in range(self._n_splits):
            y_dict = {}
            for ind, el in enumerate(y):
                y_dict.setdefault(el, []).append(ind)
            #print(f'словарь меток и индексов (y_dict) {y_dict}')

            folds_y = []#[[] for i in range(splits)]
            for i in range(self._n_splits):
                folds_y.append(sorted(y)[i::self._n_splits])
            #print(f'фолды меток (folds_y) {folds_y}')

            #folds_index = []
            #for arr in folds_y:
            #    folds_index.append([])
            #    for x in arr:
            #        folds_index[-1].append(y_dict[x].pop(0))
            self.folds_index = [[y_dict[x].pop(0) for x in arr] for arr in folds_y]
            #print(f'фолды индексов (folds_index) {folds_index}')

        for el in self.folds_index:
            yield(
                    (sorted(listmerge([x for x in self.folds_index if x!=el])), el)
                )
                


#if __name__ == '__main__':
#    split(get_n_splits)