# FINAL-Python

Python (Pytorch) implementation of "[FINAL: Fast Attributed Network Alignment](https://dl.acm.org/doi/abs/10.1145/2939672.2939766)". The official MATLAB implementation is [here](https://github.com/sizhang92/FINAL-KDD16).

### Prerequisites

- numpy
- scipy
- pytorch

### Datasets
You can run `main.py` using one of the following datasets

- foursquare-twitter
- ACM-DBLP
- Douban
- flickr-lastfm
- flickr-myspace

### Usage

1. Clone the repository to your local machine:

```sh
git clone https://github.com/yq-leo/FINAL-Python.git
```

2. Navigate to the project directory:

```sh
cd FINAL-Python
```

3. Install the required dependencies:
```sh
pip install -r requirements.txt
```

4. To run FINAL, execute the following command in the terminal:
```sh
python main.py --dataset={dataset}
```

## Reference
### Official Code
[FINAL-KDD16](https://github.com/zhichenz98/PARROT-WWW23)

### Paper
Zhang, S., & Tong, H. (2016, August). Final: Fast attributed network alignment. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 1345-1354). [DOI](http://dx.doi.org/10.1145/2939672.2939766).
