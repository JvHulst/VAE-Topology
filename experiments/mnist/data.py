"""
The two MNIST datasets.

Both take a handful of MNIST exemplars and apply a group action to each one,
sweeping the whole orbit:

- ``make_rotated_mnist_data`` rotates by angles on S¹.
- ``make_shifted_mnist_data`` shifts by pixel offsets on T², wrapping at the
  image border so the shift really is periodic.

Two labels are recorded alongside every image. The true group coordinate is what
the figures colour by. The exemplar index says which base digit an image came
from, and images sharing one differ by nothing but the group element, so the
geodesic-stress metric can compare distances against a fully known manifold.

Digit choice matters. A rotated 0, 1 or 8 is close to its own rotation by π, so
those digits have a symmetry the latent circle cannot see and they are excluded
from the rotated dataset. The shifted dataset has no such problem — a shift acts
freely whatever the digit — so it uses all ten.

Author: Jilles van Hulst
"""

import os
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets as tv_datasets

#: Where the raw MNIST download is cached.  Anchored to the repository rather
#: than the working directory, which is how two copies of it came to exist.
MNIST_DIR = str(Path(__file__).resolve().parents[2] / 'data' / 'mnist')


def make_rotated_mnist_data(digits, n_exemplars_per_digit, rotations_per_exemplar,
                             data_dir=MNIST_DIR, seed=42):
    """
    Load MNIST and create rotationally augmented copies.

    For each digit in `digits`, samples `n_exemplars_per_digit` images and
    rotates each by `rotations_per_exemplar` uniformly sampled angles.

    Args:
        digits: List of digit classes (0–9) or a single int.
        n_exemplars_per_digit: Number of distinct MNIST images per digit.
        rotations_per_exemplar: Number of rotated copies per exemplar.
        data_dir: Directory to cache MNIST (default: MNIST_DIR).
        seed: Random seed.

    Returns:
        dict with:
            'x':        float32 [N, 784]
            'theta':    float32 [N] rotation angle in (−π, π]
            'labels':   int32   [N] digit class
            'exemplar': int32   [N] which base image, unique across digits
    """
    from scipy.ndimage import rotate as ndimage_rotate

    if isinstance(digits, int):
        digits = [digits]

    os.makedirs(data_dir, exist_ok=True)
    mnist = tv_datasets.MNIST(root=data_dir, train=True, download=True)
    all_images = mnist.data.numpy().astype(np.float32) / 255.0
    all_labels = mnist.targets.numpy()

    rng = np.random.RandomState(seed)
    xs, thetas, label_list, exemplar_list = [], [], [], []
    exemplar = -1

    for digit in digits:
        mask = all_labels == digit
        digit_images = all_images[mask]
        n_avail = len(digit_images)
        n_ex = min(n_exemplars_per_digit, n_avail)
        exemplar_idx = rng.choice(n_avail, size=n_ex, replace=False)
        exemplars = digit_images[exemplar_idx]

        for img in exemplars:
            exemplar += 1
            angles_deg = rng.uniform(0, 360, rotations_per_exemplar)
            for angle_deg in angles_deg:
                rotated = ndimage_rotate(img, angle_deg, reshape=False,
                                         order=1, mode='constant', cval=0.0)
                xs.append(rotated.ravel())
                theta = (np.deg2rad(angle_deg) + np.pi) % (2 * np.pi) - np.pi
                thetas.append(theta)
                label_list.append(digit)
                exemplar_list.append(exemplar)

    xs = np.array(xs, dtype=np.float32)
    thetas = np.array(thetas, dtype=np.float32)
    label_arr = np.array(label_list, dtype=np.int32)
    exemplar_arr = np.array(exemplar_list, dtype=np.int32)

    n = len(xs)
    n_digits = len(digits)
    print(f"  Loaded {n_digits} digit(s), {n_ex} exemplar(s) each "
          f"x {rotations_per_exemplar} rotations -> {n} images")
    return {'x': xs, 'theta': thetas, 'labels': label_arr,
            'exemplar': exemplar_arr}


def make_shifted_mnist_data(digits, n_exemplars_per_digit, shifts_per_exemplar,
                             data_dir=MNIST_DIR, seed=42):
    """
    Load MNIST and create periodically shifted copies.

    For each digit in `digits`, samples `n_exemplars_per_digit` distinct images
    and applies `shifts_per_exemplar` random cyclic shifts (np.roll) in both
    axes.  Because np.roll wraps around, the true generative factor lives on
    T² = S¹ × S¹.

    Shift-to-angle conversion:
        θ = (2π · d / 28 + π) % (2π) − π   ∈ (−π, π]
    so that d=0 ↦ 0, d=7 ↦ π/2, d=14 ↦ −π (±π boundary), d=21 ↦ −π/2.

    Args:
        digits: List of digit classes (0–9) or a single int.
        n_exemplars_per_digit: Number of distinct MNIST images per digit.
        shifts_per_exemplar: Number of shifted copies per exemplar.
        data_dir: Directory to cache the raw MNIST download (default: MNIST_DIR).
        seed: Random seed for reproducibility.

    Returns:
        dict with:
            'x':        float32 [N, 784] images in [0, 1]
            'theta1':   float32 [N] vertical-shift angles in (−π, π]
            'theta2':   float32 [N] horizontal-shift angles in (−π, π]
            'labels':   int32   [N] digit class (0–9)
            'exemplar': int32   [N] which base image, unique across digits
            'dx':       int32   [N] raw vertical pixel shift ∈ [0, 28)
            'dy':       int32   [N] raw horizontal pixel shift ∈ [0, 28)
    """

    if isinstance(digits, int):
        digits = [digits]

    os.makedirs(data_dir, exist_ok=True)
    mnist = tv_datasets.MNIST(root=data_dir, train=True, download=True)
    all_images = mnist.data.numpy().astype(np.float32) / 255.0  # [60000, 28, 28]
    all_labels = mnist.targets.numpy()

    rng = np.random.RandomState(seed)
    xs, theta1s, theta2s, label_list, dxs, dys = [], [], [], [], [], []
    exemplar_list = []
    exemplar = -1

    for digit in digits:
        mask = all_labels == digit
        digit_images = all_images[mask]
        n_avail = len(digit_images)
        n_ex = min(n_exemplars_per_digit, n_avail)
        exemplar_idx = rng.choice(n_avail, size=n_ex, replace=False)
        exemplars = digit_images[exemplar_idx]

        for img in exemplars:
            exemplar += 1
            for _ in range(shifts_per_exemplar):
                dx = rng.randint(0, 28)
                dy = rng.randint(0, 28)
                shifted = np.roll(np.roll(img, dx, axis=0), dy, axis=1)
                xs.append(shifted.ravel())
                theta1 = (2 * np.pi * dx / 28 + np.pi) % (2 * np.pi) - np.pi
                theta2 = (2 * np.pi * dy / 28 + np.pi) % (2 * np.pi) - np.pi
                theta1s.append(theta1)
                theta2s.append(theta2)
                label_list.append(digit)
                exemplar_list.append(exemplar)
                dxs.append(dx)
                dys.append(dy)

    xs = np.array(xs, dtype=np.float32)
    theta1s = np.array(theta1s, dtype=np.float32)
    theta2s = np.array(theta2s, dtype=np.float32)
    label_arr = np.array(label_list, dtype=np.int32)
    dxs = np.array(dxs, dtype=np.int32)
    dys = np.array(dys, dtype=np.int32)

    n = len(xs)
    n_digits = len(digits)
    print(f"  Loaded {n_digits} digit(s), "
          f"{n_ex} exemplar(s) each x {shifts_per_exemplar} shifts -> {n} images")
    return {
        'x': xs,
        'theta1': theta1s,
        'theta2': theta2s,
        'labels': label_arr,
        'exemplar': np.array(exemplar_list, dtype=np.int32),
        'dx': dxs,
        'dy': dys,
    }


class MNISTDataset(torch.utils.data.Dataset):
    """
    Thin PyTorch Dataset wrapper around pre-computed numpy MNIST data.

    Each item is returned as a tuple whose first element is the flat image
    tensor [784]; remaining elements carry whatever metadata arrays were
    passed (e.g., theta1, theta2, label).
    """

    def __init__(self, arrays):
        """
        Args:
            arrays: dict containing at minimum 'x' (float32 [N, 784]).
                    Any additional keys are stored as extra metadata tensors.
        """
        self.x = torch.tensor(arrays['x'], dtype=torch.float32)
        self.extras = {}
        for k, v in arrays.items():
            if k == 'x':
                continue
            if v.dtype in (np.float32, np.float64):
                self.extras[k] = torch.tensor(v, dtype=torch.float32)
            else:
                self.extras[k] = torch.tensor(v, dtype=torch.long)
        self._extra_keys = list(self.extras.keys())

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        items = [self.x[idx]]
        for k in self._extra_keys:
            items.append(self.extras[k][idx])
        return tuple(items)


def make_mnist_loaders(data, batch_size=64, train_ratio=0.8, seed=0):
    """
    Create train and validation DataLoaders from a data dict.

    Args:
        data: dict from make_shifted_mnist_data or make_rotated_mnist_data.
        batch_size: Mini-batch size.
        train_ratio: Fraction of data used for training.
        seed: Random seed for the split.

    Returns:
        train_loader, val_loader, train_indices, val_indices
    """
    import torch.utils.data as tud

    N = len(data['x'])
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)
    n_train = int(train_ratio * N)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    train_arrays = {k: v[train_idx] for k, v in data.items()}
    val_arrays = {k: v[val_idx] for k, v in data.items()}

    train_ds = MNISTDataset(train_arrays)
    val_ds = MNISTDataset(val_arrays)

    train_loader = tud.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=False)
    val_loader = tud.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=False)

    return train_loader, val_loader, train_arrays, val_arrays
