import json

from args import make_args
from utils.data import load_dataset_to_torch
from utils.match import greedy_match, compute_accuracy
from final import final

if __name__ == "__main__":
    args = make_args()
    with open(f"settings/{args.dataset}.json", "r") as f:
        settings = json.load(f)

    g1, g2 = settings['g1'], settings['g2']
    print(f'Loading dataset {args.dataset}...', end=' ')
    data = load_dataset_to_torch(args.dataset, g1, g2)
    print('Done')

    alpha = settings['alpha']
    maxiter = settings['maxiter']
    tol = settings['tol']
    s = final(data, alpha, maxiter, tol)
    m = greedy_match(s)
    acc = compute_accuracy(m, data['gnd'])
    print('Accuracy: {:.2f}%'.format(acc * 100))
