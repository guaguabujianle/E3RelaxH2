import os
import pickle
import torch
import itertools
import numpy as np

def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path,i)  
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)

def write_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj

def warp_relaxed_structure_batched(coords_u, coords_r, cell_r, batch):
    """
    coords_u: (N, 3)   unrelaxed/reference positions (all structures concatenated)
    coords_r: (N, 3)   relaxed positions (all structures concatenated)
    cell_r:   (B, 3, 3) cell for each structure
    batch:    (N,)      long tensor, batch[i] = structure index of atom i

    Returns:
    coords_r_wrapped: (N, 3)   wrapped relaxed positions (all structures concatenated)
    """

    device = coords_u.device
    supercells = torch.FloatTensor(
        list(itertools.product((-1, 0, 1), repeat=3))
    ).to(device)  # (27, 3)

    # Loop over each structure index in the batch
    coords_r_wrapped_list = []
    for b in batch.unique(sorted=True):
        b = int(b.item())
        mask = (batch == b)             # atoms belonging to structure b
        cu_b = coords_u[mask]           # (Nb, 3)
        cr_b = coords_r[mask]           # (Nb, 3)
        cell_b = cell_r[b]              # (3, 3)

        if cu_b.numel() == 0:
            continue  # just in case

        # --- Same logic as your original function, but per-structure ---

        # 27 periodic images of coords_r
        # supercells @ cell_b: (27, 3)
        super_coords_r = cr_b.unsqueeze(1) + (supercells @ cell_b).unsqueeze(0)  # (Nb, 27, 3)

        # Compute distances from each cu_b[i] to its 27 images of cr_b[i]
        # cu_b.unsqueeze(1): (Nb, 1, 3)
        # torch.cdist with shapes (Nb,1,3) and (Nb,27,3) → (Nb,1,27)
        dists = torch.cdist(cu_b.unsqueeze(1), super_coords_r)  # (Nb, 1, 27)

        # For each atom, choose the closest image index in [-1,0,1]^3
        image = dists.argmin(dim=-1).squeeze(-1)   # (Nb,)
        cell_offsets = supercells[image]           # (Nb, 3)

        # Apply periodic shift
        coords_r_wrapped = cr_b + cell_offsets @ cell_b  # (Nb, 3)
        coords_r_wrapped_list.append(coords_r_wrapped)

    coords_r_wrapped = torch.cat(coords_r_wrapped_list, dim=0)  # (N, 3)

    return coords_r_wrapped.detach()


def compute_cart_mean_absolute_displacement(coords_u, coords_r, cell_r):
    supercells = torch.FloatTensor(list(itertools.product((-1, 0, 1), repeat=3))).to(coords_u.device)
    super_coords_r = coords_r.unsqueeze(1) + (supercells @ cell_r).unsqueeze(0)
    dists = torch.cdist(coords_u.unsqueeze(1), super_coords_r)
    image = dists.argmin(dim=-1).squeeze()
    cell_offsets = supercells[image]
    coords_r = coords_r + cell_offsets @ cell_r
    mad = (coords_r - coords_u).abs().mean()

    return mad


def compute_cart_mean_absolute_displacement_wrap_pred(coords_pred, coords_true, cell, batch):
    """
    Compute mean absolute Cartesian displacement under PBC, wrapping *predictions*.

    Args:
        coords_pred: (N, 3)  predicted positions (all structures concatenated)
        coords_true: (N, 3)  reference / target positions (all structures concatenated)
        cell:        (B, 3, 3) cell matrix for each structure in the batch
        batch:       (N,)  long tensor, batch[i] = structure index of atom i

    Returns:
        mad: scalar, mean absolute displacement over all atoms and Cartesian components
    """

    device = coords_pred.device
    dtype = coords_pred.dtype

    # Translation vectors for 27 images: (-1,0,1)^3
    supercells = torch.tensor(
        list(itertools.product((-1, 0, 1), repeat=3)),
        device=device,
        dtype=dtype,
    )  # (27, 3)

    total_abs = coords_pred.new_tensor(0.0)
    total_count = 0

    for b in batch.unique(sorted=True):
        b = int(b.item())
        mask = (batch == b)          # atoms for structure b

        cp_b = coords_pred[mask]     # (Nb, 3)
        ct_b = coords_true[mask]     # (Nb, 3)
        cell_b = cell[b]             # (3, 3)

        if cp_b.numel() == 0:
            continue

        # 27 periodic images of the PREDICTIONS
        translations = supercells @ cell_b                        # (27, 3)
        super_coords_pred = cp_b.unsqueeze(1) + translations.unsqueeze(0)  # (Nb, 27, 3)

        # Distances from each TRUE coord to its 27 images of the PRED coord
        # ct_b.unsqueeze(1): (Nb, 1, 3)
        # super_coords_pred: (Nb, 27, 3)
        # => dists: (Nb, 1, 27)
        dists = torch.cdist(ct_b.unsqueeze(1), super_coords_pred)  # (Nb, 1, 27)

        # Index of closest periodic image for each atom
        image = dists.argmin(dim=-1).squeeze(-1)  # (Nb,)
        cell_offsets = supercells[image]          # (Nb, 3)

        # Wrap predictions using the selected cell offsets
        coords_pred_wrapped = cp_b + cell_offsets @ cell_b   # (Nb, 3)

        # L1 error per Cartesian component
        diff = (coords_pred_wrapped - ct_b).abs()            # (Nb, 3)
        total_abs += diff.sum()
        total_count += diff.numel()

    mad = total_abs / total_count
    return mad



def _cell_abc(cell: np.ndarray):
    """Return lattice lengths (a,b,c) from a 3x3 cell (rows are lattice vectors)."""
    cell = np.asarray(cell, dtype=float).reshape(3, 3)
    a, b, c = np.linalg.norm(cell, axis=1)
    return a, b, c

def _rdf_g(pos: np.ndarray, cell: np.ndarray, rmax: float, nbins: int):
    """
    Compute RDF g(r) from positions (N,3) in Cartesian with PBC using minimum-image convention.
    Returns (r_centers, g_r).
    """
    pos = np.asarray(pos, dtype=float)
    cell = np.asarray(cell, dtype=float).reshape(3, 3)
    N = pos.shape[0]
    if N < 2:
        r = np.linspace(0.0, rmax, nbins, endpoint=False) + (rmax / nbins) * 0.5
        return r, np.zeros_like(r)

    V = abs(np.linalg.det(cell))
    if V <= 1e-12:
        raise ValueError("Cell volume is non-positive or near zero.")

    inv_cell = np.linalg.inv(cell)

    # Collect pair distances with MIC
    dists = []
    for i in range(N - 1):
        disp = pos[i + 1:] - pos[i]          # (N-i-1, 3) Cartesian
        df = disp @ inv_cell                 # fractional displacement
        df -= np.round(df)                   # wrap to [-0.5, 0.5) (minimum-image)
        disp_mic = df @ cell                 # back to Cartesian
        d = np.linalg.norm(disp_mic, axis=1)
        if rmax is not None:
            d = d[d < rmax]
        dists.append(d)

    if len(dists) == 0:
        d_all = np.array([], dtype=float)
    else:
        d_all = np.concatenate(dists, axis=0)

    # Histogram
    edges = np.linspace(0.0, rmax, nbins + 1)
    hist, _ = np.histogram(d_all, bins=edges)

    # Normalize to make g(r) ~ 1 for uniform distribution
    r_in = edges[:-1]
    r_out = edges[1:]
    shell_vol = (4.0 / 3.0) * np.pi * (r_out**3 - r_in**3)  # volume of spherical shell

    total_pairs = N * (N - 1) / 2.0
    expected = total_pairs * (shell_vol / V)               # expected counts per bin for ideal gas
    g = np.divide(hist, expected, out=np.zeros_like(expected, dtype=float), where=expected > 0)

    r_centers = 0.5 * (r_in + r_out)
    return r_centers, g

def get_rdf_mae(pos_p, cell_p, pos_r, cell_r, nbins=200, rmax=None):
    """
    MAE between RDF curves of predicted (p) and reference (r).
    """
    # convert cell_* to abc and pick a safe common rmax
    ap, bp, cp = _cell_abc(cell_p)
    ar, br, cr = _cell_abc(cell_r)

    rmax_safe = 0.5 * min(ap, bp, cp, ar, br, cr)
    if rmax is None:
        rmax = rmax_safe
    else:
        rmax = min(float(rmax), rmax_safe)  # enforce MIC safety

    r, g_p = _rdf_g(pos_p, cell_p, rmax=rmax, nbins=nbins)
    _, g_r = _rdf_g(pos_r, cell_r, rmax=rmax, nbins=nbins)

    mae = float(np.mean(np.abs(g_p - g_r)))
    return mae, r, g_p, g_r



def create_dir(dir_list):
    assert  isinstance(dir_list, list) == True
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)

def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))

def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))

class BestMeter(object):
    """Computes and stores the best value"""

    def __init__(self, best_type):
        self.best_type = best_type  
        self.count = 0      
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg