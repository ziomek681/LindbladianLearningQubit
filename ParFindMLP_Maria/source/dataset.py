import json
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

TAU_KEYS = ["tau_relax1", "tau_relax2", "tau_deph1", "tau_deph2", "tau_cordeph"] # ordered label vector


### Keep the full 16-entry (not just upper-triangle) encoding.
def rho_to_real_features(rhos: np.ndarray) -> np.ndarray:
    """
    Convert a complex density-matrix trajectory into a flat real feature vector.
    :param:
        rhos: complex array, shape (n_steps+1, 4, 4)
    :returns:
        float64 array, shape (n_steps+1 * 32,)  [(real,imag) of all 16 entries, per step, flattened]
    """

    real_part = rhos.real.astype(np.float64)
    imag_part = rhos.imag.astype(np.float64)

    stacked = np.stack([real_part, imag_part], axis=-1)
    return stacked.reshape(-1)

def traj_to_real_features(rhos: np.ndarray, times, featuretimes) -> np.ndarray:
    """
    Convert a complex density-matrix trajectory into a flat real feature vector,
    keeping only density matrices separated by at least `timestep`.

    Parameters
    ----------
    rhos : np.ndarray
        Complex array of shape (n_steps+1, 4, 4).
    times : array-like
        Time corresponding to each density matrix.
    timestep : float
        Minimum time separation between selected density matrices.

    Returns
    -------
    np.ndarray
        Flattened float64 feature vector containing the real and imaginary parts
        of the selected density matrices.
    """
    times = np.asarray(times)

    if len(rhos) != len(times):
        raise ValueError("rhos and times must have the same length.")

    # Always keep the first density matrix
    indices = [0]
    last_time = times[0]

    featuretimes_index = 0

    for i in range(1, len(times)):
        if featuretimes_index >= len(featuretimes): #jesli znajdzie juz wszystkie zamowione czasy to koniec
            break
        if times[i] >= featuretimes[featuretimes_index]:
            indices.append(i)
            featuretimes_index = featuretimes_index + 1

    rhos_selected = rhos[indices]

    #print(rhos_selected.shape)

    #real_part = rhos_selected.real.astype(np.float64)
    #imag_part = rhos_selected.imag.astype(np.float64)

    #stacked = np.stack([real_part, imag_part], axis=-1)
    #print(stacked.shape)
    #return stacked.reshape(-1)

    # lista elementów do zapisania
    compressed = np.empty((len(rhos_selected), 16), dtype=np.float64)

    for i, rho in enumerate(rhos_selected):
        k = 0

        # diagonalne elementy rzeczywiste
        for j in range(4):
            compressed[i, k] = rho[j, j].real
            k += 1

        # górny trójkąt - elementy zespolone
        for row in range(4):
            for col in range(row + 1, 4):
                compressed[i, k] = rho[row, col].real
                compressed[i, k + 1] = rho[row, col].imag
                k += 2

    #print(compressed.shape)
    return compressed.reshape(-1)

### Use only the 15 independent real parameters (see notes (3)).
# def rho_to_real_features(rhos: np.ndarray) -> np.ndarray:
#     """
#     Convert a complex density-matrix trajectory into a flat real feature vector,
#     using only the 15 independent real parameters (see notes (3)).
#
#     :param rhos: complex array, shape (n_steps+1, 4, 4)
#     :param n_points: if given, downsample the time axis to this many points
#         first (see downsample_time_steps). If None, use all time steps.
#     :returns: float64 array, shape (n_used_steps * 15,)
#     """
#
#     _UPPER_TRIANGLE_ROWS, _UPPER_TRIANGLE_COLS = np.triu_indices(4, k=1)  # get the upper triangle indices
#
#     diag = rhos[:, [0, 1, 2], [0, 1, 2]].real.astype(np.float64)  # (n_steps, 3) get the diagonal entries
#
#     # 6 strictly-upper-triangular entries, real + imag.
#
#     upper = rhos[:, _UPPER_TRIANGLE_ROWS, _UPPER_TRIANGLE_COLS]  # (n_steps, 6) complex
#     upper_real = upper.real.astype(np.float64)
#     upper_imag = upper.imag.astype(np.float64)
#
#     # concatenate per time step
#     per_step = np.concatenate([diag, upper_real, upper_imag], axis=1)  # (n_steps, 15)
#     return per_step.reshape(-1)


class TrajectoryDataset(Dataset):
    """
    one dataset directory = one TrajectoryDataset
    """

    def __init__(self, root_dir: str, preloaddata, featuretimes, label_transform: str = "log"):
        """
        :param:
            root_dir: path to a single dataset directory (contains metadata.json + traj_*.npz)
            label_transform: "log" (recommended, see notes (1)) or "none"
        """
        self.root_dir = root_dir
        self.label_transform = label_transform

        meta_path = os.path.join(root_dir, "metadata.json")
        with open(meta_path, "r") as f:
            self.metadata = json.load(f)

        self.files = sorted(glob.glob(os.path.join(root_dir, "traj_*.npz")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No traj_*.npz files found in {root_dir}")

        # Feature dimensionality is fixed for a given dataset (from Jagoda), but we don't want to assume it in the code
        self._feature_dim = None        # dimensionality of the features
        self._n_steps_plus_1 = None     # n_steps+1 (number of timesteps)

        self.preloaddata = preloaddata

        self.featuretimes = featuretimes

        if self.preloaddata == True:
            self.features = []
            self.labels = []
            self.rhos = []

            timestep = 2 * 1e-12  # 2ps

            for idx in range(len(self.files)):
                data = self._load_raw(idx)

                rhos = data["rhos"]
                times = data["times"]

                features = torch.from_numpy(
                    traj_to_real_features(rhos, times, self.featuretimes)
                ).float()

                labels = 1e12 * torch.tensor(
                    [float(data[k]) for k in TAU_KEYS],
                    dtype=torch.float32
                )

                if self._feature_dim is None:
                    self._feature_dim = features.shape[
                        0]  # get the dimensionality of the features from the first trajectory

                self.features.append(features)
                self.labels.append(labels)
                self.rhos.append(torch.from_numpy(rhos.astype(np.complex64)))


    def __len__(self):
        '''
        :return:
           length of the dataset
        '''
        return len(self.files)

    def _load_raw(self, idx):
        '''
        :param
            idx: numpy array index
        :return:
            loaded data from the file "traj_idx.npz"
        '''
        data = np.load(self.files[idx])
        return data

    def __getitem__(self, idx):
        '''
        :param
            idx: numpy array index
        :return:
            features  : (n_steps+1 * 32,) float tensor -- flattened trajectory encoding
            labels    : (5,) float tensor -- log(tau) (or raw tau if label_transform="none")
            rho_t     : (4,4) complex tensor -- density matrix at a randomly chosen time step t
            rho_tp1   : (4,4) complex tensor -- density matrix at time step t+1 (i.e. t + dt)

            The (rho_t, rho_tp1) pair is re-sampled at random on every call
            (every epoch/batch via the DataLoader), not fixed once per
            trajectory, so the physics loss gets evaluated at varying points
            along the dynamics rather than always at the same time.
        '''

        k = 10  # ilosc par wylosowanych z kazdej trajektorii

        if self.preloaddata == False:
            data = self._load_raw(idx)

            rhos = data["rhos"]  # complex64, (n_steps+1, 4, 4)
            times = data["times"]
            # features_np = rho_to_real_features(rhos)    # convert to real-valued features (n_steps+1 * 32,)
            features_np = traj_to_real_features(rhos, times, self.featuretimes)

            if self._feature_dim is None:
                self._feature_dim = features_np.shape[
                    0]  # get the dimensionality of the features from the first trajectory
                self._n_steps_plus_1 = rhos.shape[0]  # get the number of timesteps from the first trajectory

            labels_np = 1e12 * np.array([float(data[k]) for k in TAU_KEYS],
                                        dtype=np.float64)  # parameters as a float64 array

            if self.label_transform == "log":
                labels_np = np.log(labels_np)  # see notes (1) for the explanation of this
            elif self.label_transform == "none":
                pass
            else:
                raise ValueError(f"Unknown label_transform: {self.label_transform}")

            features = torch.from_numpy(features_np).float()  # convert from numpy to torch tensor
            labels = torch.from_numpy(labels_np).float()  # convert from numpy to torch tensor

            # physics-loss support: randomly chosen consecutive (rho_t, rho_t+dt) pair
            n_steps = rhos.shape[0] - 1  # number of (t, t+1) pairs available

            t_idx = np.random.choice(n_steps, size=k, replace=False)

            rho_t = torch.from_numpy(rhos[t_idx].astype(np.complex64))
            rho_tp1 = torch.from_numpy(rhos[t_idx + 1].astype(np.complex64))

        else:
            features = self.features[idx]
            labels = self.labels[idx]
            rhos = self.rhos[idx]

            if self.label_transform == "log":
                labels = torch.log(labels)  # see notes (1) for the explanation of this
            elif self.label_transform == "none":
                pass
            else:
                raise ValueError(f"Unknown label_transform: {self.label_transform}")

            n_steps = rhos.shape[0] - 1

            t_idx = torch.randperm(n_steps)[:k]

            rho_t = rhos[t_idx]
            rho_tp1 = rhos[t_idx + 1]

        return features, labels, rho_t, rho_tp1

    @property
    def feature_dim(self):
        """
        Infer feature dimensionality by peeking at file 0, without calling __getitem__
        """
        if self._feature_dim is None:
            _ = self[0]
        return self._feature_dim


def split_dataset(ds: Dataset, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0) -> tuple[Subset, Subset, Subset]:
    """
    Random split into train/val/test Subsets for training/validation/testing of the MLP.

    :param:
        ds: Dataset to split
        val_frac: fraction of the dataset to use for validation
        test_frac: fraction of the dataset to use for testing
        seed: random seed for reproducibility
    :return:
        subsets for train/val/test
    """
    n = len(ds)
    rng = np.random.default_rng(seed)   # a random number generator with a fixed seed
    indices = rng.permutation(n)        # random permutation of indices

    n_val = int(n * val_frac)           # valiation set size
    n_test = int(n * test_frac)         # test set size
    n_train = n - n_val - n_test        # training set size

    train_idx = indices[:n_train]                   # indices of the training set
    val_idx = indices[n_train:n_train + n_val]      # indices of the validation set
    test_idx = indices[n_train + n_val:]            # indices of the test set

    return Subset(ds, train_idx), Subset(ds, val_idx), Subset(ds, test_idx)


def compute_label_stats(ds: Dataset, indices=None):
    """
    Compute per-parameter mean/std over the (already log-transformed, if label_transform='log') labels of a dataset or subset of it.

    For more details, see the notes (2)

    :return:
        mean (5,), std (5,) as float32 tensors.
    """

    if indices is None:
        indices = range(len(ds))

    all_labels = torch.stack([ds[i][1] for i in indices])   # stack all labels as a single tensor
    mean = all_labels.mean(dim=0)
    std = all_labels.std(dim=0)

    return mean, std, all_labels




