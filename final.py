import torch
import torch.nn.functional as F
import time


def final(data, alpha, maxiter, tol):
    n1, n2 = data['n1'], data['n2']
    adj1, adj2 = data['adj1'], data['adj2']
    node_attr1, node_attr2 = data['node_attr1'], data['node_attr2']
    edge_attr1, edge_attr2 = data['edge_attr1'], data['edge_attr2']
    H = data['H']

    adj1 = adj1.to(torch.float64)
    adj2 = adj2.to(torch.float64)

    if node_attr1 is None and node_attr2 is None:
        node_attr1 = torch.ones(n1, 1).to(torch.float64)
        node_attr2 = torch.ones(n2, 1).to(torch.float64)
    else:
        node_attr1 = node_attr1.to(torch.float64)
        node_attr2 = node_attr2.to(torch.float64)

    if edge_attr1 is None and edge_attr2 is None:
        edge_attr1 = (adj1 > 0).to(torch.float64).unsqueeze(0)
        edge_attr2 = (adj2 > 0).to(torch.float64).unsqueeze(0)
    else:
        edge_attr1 = edge_attr1.to(torch.float64)
        edge_attr2 = edge_attr2.to(torch.float64)

    K, L = node_attr1.shape[1], edge_attr1.shape[0]

    # normalize node feature vectors
    node_attr1 = F.normalize(node_attr1, p=2, dim=1)
    node_attr2 = F.normalize(node_attr2, p=2, dim=1)

    # normalize edge feature vectors
    edge_attr1 = _normalize_edge_attr(edge_attr1)
    edge_attr2 = _normalize_edge_attr(edge_attr2)

    # compute node feature cosine cross-similarity
    N = torch.zeros(n2, n1).to(torch.float64)
    for k in range(K):
        N += torch.outer(node_attr2[:, k], node_attr1[:, k])

    # Compute the Kronecker degree vector
    d = torch.zeros(n2, n1).to(torch.float64)
    start_time = time.time()
    for l in range(L):
        for k in range(K):
            d += torch.outer((edge_attr2[l] * adj2 @ node_attr2[:, k]),
                             (edge_attr1[l] * adj1 @ node_attr1[:, k]))
    print('Time for degree: {:.2f} seconds'.format(time.time() - start_time))

    D = N * d
    maskD = D > 0
    D[maskD] = torch.reciprocal(torch.sqrt(D[maskD]))

    # Fixed-point solution
    q = D * N
    h = H.to(torch.float64)
    s = h.clone()

    for i in range(maxiter):
        print(f"Iteration {i + 1}")
        start_time = time.time()
        prev_s = s.clone()

        M = q * s
        S = torch.zeros(n2, n1)

        for l in range(L):
            S += (edge_attr2[l] * adj2) @ M @ (edge_attr1[l] * adj1)

        s = (1 - alpha) * h + alpha * q * S
        diff = torch.norm(s - prev_s)

        print(f"Time for iteration {i + 1}: {time.time() - start_time:.2f} sec, diff = {100 * diff:.5f}")

        if diff < tol:
            break

    return s


def _normalize_edge_attr(edge_attr):
    T = torch.sum(edge_attr ** 2, dim=0)
    T[T > 0] = torch.reciprocal(torch.sqrt(T[T > 0]))
    edge_attr_normalized = edge_attr * T.unsqueeze(0)
    return edge_attr_normalized
