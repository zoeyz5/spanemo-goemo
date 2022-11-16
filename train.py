import os

from trainer import Trainer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # training config
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--batch_size_eval', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=50, help='# of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')
    parser.add_argument('--n_workers', type=int, default=4, help='# of workers in dataloader.')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold to decide 0 or 1')
    parser.add_argument('--mode', type=str, choices=('original', 'grouping', 'ekman'), default='original')
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    print(args)
    solver = Trainer(args)
    solver.train()


