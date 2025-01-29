from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='foursquare-twitter',
                        choices=['foursquare-twitter', 'ACM-DBLP', 'Douban', 'flickr-lastfm', 'flickr-myspace'],
                        help='Datasets: foursquare-twitter; ACM-DBLP; Douban; flickr-lastfm; flickr-myspace')
    return parser.parse_args()
