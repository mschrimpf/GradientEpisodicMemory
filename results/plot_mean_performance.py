import argparse

import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot

seaborn.set()
seaborn.set_context('paper')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, default='ewc')
  parser.add_argument('--dataset', type=str, default='mnist_permutations')
  parser.add_argument('--identifier', type=str, default='2018_10_09_14_35_10_11b1023e748f45ec8e0b994bd3fac8f6')
  args = parser.parse_args()
  print("Running with args", args)

  basepath = f"{args.model}_{args.dataset}.pt_{args.identifier}"
  log = pd.read_csv(f"{basepath}.txt", sep=" ", header=None, skiprows=2, nrows=20).values
  mean_task_performances = [log[i, :i + 1].mean() for i in range(len(log))]

  pyplot.plot(list(range(1, len(mean_task_performances) + 1)), mean_task_performances)
  pyplot.xlabel("Task Num")
  pyplot.ylabel("Mean Performance on Observed Tasks")
  pyplot.ylim([0, 1])
  pyplot.xticks(np.arange(1, 20 + 1))
  pyplot.savefig(f"{basepath}.png")


if __name__ == '__main__':
  main()
