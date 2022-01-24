import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--frame_num', type=int, default=100)
    parser.add_argument('--method', type=str, default='sibefull')
    parser.add_argument('--Y', type=float, default=1e7)
    parser.add_argument('--P', type=float, default=0.4)
    args = parser.parse_args()
    return args