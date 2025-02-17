import torch


def greedy_match(X: torch.Tensor):
    m, n = X.shape

    X_flat = X.T.reshape(-1)
    indices = torch.argsort(X_flat, descending=True)

    min_size = min(m, n)
    used_rows = torch.zeros(m, dtype=torch.bool)
    used_cols = torch.zeros(n, dtype=torch.bool)

    row = torch.zeros(min_size, dtype=torch.long)
    col = torch.zeros(min_size, dtype=torch.long)

    matched = 0
    index = 0

    while matched < min_size:
        ipos = indices[index].item()  # Convert to Python integer
        jc = ipos // m  # Column index
        ic = ipos % m  # Row index

        if not used_rows[ic] and not used_cols[jc]:
            row[matched] = ic
            col[matched] = jc
            used_rows[ic] = True
            used_cols[jc] = True
            matched += 1

        index += 1

    M = torch.zeros((m, n), dtype=torch.float32)
    M[row, col] = 1.0

    return M


def direct_match(s):
    matched = torch.argsort(-s, dim=1)[:, 0]
    m = torch.zeros_like(s)
    m[torch.arange(s.shape[0]), matched] = 1
    return m


def compute_accuracy(M, gnd):
    row, col = torch.where(M == 1)
    matched_pairs = torch.stack((col, row), dim=1)

    matched_pairs = matched_pairs.to(torch.int64)
    gnd = gnd.to(torch.int64)

    intersection = torch.sum((matched_pairs[:, None] == gnd).all(dim=2).any(dim=1))
    acc = intersection.item() / gnd.shape[0]

    return acc
