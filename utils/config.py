import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', default="/home/tjc/pycharmprojects/newdata/dataset1/train_val_seg",
                        help='dataset root path')
    parser.add_argument('--lr', default=0.001, type=float, metavar='N',
                        help='learning rate')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=24, type=int, metavar='N',
                        help='mini-batch size')

    args = parser.parse_args()

    return args
