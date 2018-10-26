import os
from glob import glob

import fire
import numpy as np
import pandas as pd
import seaborn
import torch
from matplotlib import pyplot
from matplotlib.ticker import ScalarFormatter

seaborn.set()
seaborn.set_context('paper')


def build_basepath(model, dataset, identifier):
  return f"{model}_{dataset}.pt_{identifier}"


def get_task_performances(basepath):
  log = pd.read_csv(f"{basepath}.txt", sep=" ", header=None, skiprows=2, nrows=20).values
  mean_task_performances = [log[i, :i + 1].mean() for i in range(len(log))]
  return mean_task_performances


def get_args(basepath):
  args = eval(torch.load(basepath + '.pt')[4])
  return args


def collect_data():
  savepath = os.path.join(os.path.dirname(__file__), 'results.csv')
  if os.path.isfile(savepath):
    data = pd.read_csv(savepath, index_col=0)
    data['performances'] = data['performances'].apply(lambda strarray: np.fromstring(strarray[1:-1], sep=', '))
  else:
    files = glob(os.path.join(os.path.dirname(__file__), '*.txt'))
    files = [os.path.splitext(file)[0] for file in files]
    performances = [get_task_performances(file) for file in files]
    args = [get_args(file) for file in files]
    args = {k: [dic[k] for dic in args] for k in args[0]}
    data = pd.DataFrame({**{'basepath': files, 'performances': performances}, **args})
    data.to_csv(savepath)

  data = data[(data['n_hiddens'] == 3200) & (data['n_layers'] == 1) & (data['samples_per_task'] == 60000)]
  return data


def memory_performances(dataset='mnist_permutations'):
  data = collect_data()
  data = data[data['data_file'] == f"{dataset}.pt"]
  data['mean-performance'] = data['performances'].apply(lambda performances: performances[-1])
  models = np.unique(data['model'])

  fig = pyplot.figure()
  lines = []
  for i, model in enumerate(models):
    ax = fig.add_subplot(111, label=f"{i}", frame_on=i == 0)
    if i > 0:
      ax.xaxis.tick_top()
      ax.xaxis.set_label_position('top')
    model_data = data[data['model'] == model]
    memories, performances = model_data['n_memories'], model_data['mean-performance']
    memories, performances = list(zip(*sorted(zip(memories, performances))))
    if model == 'gem':
      # gem uses n_memories per task, so multiply by number of tasks
      memories = [m * 20 for m in memories]
    line = ax.semilogx(memories, performances, color=f"C{i}")
    lines += line
    ax.set_xticks(memories)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_ylim([0, 1])
  pyplot.ylabel('Mean performance after last task')
  fig.legend(lines, models)
  pyplot.savefig(f'memory-{dataset}.png')


def mean_performance(model='ewc', dataset='mnist_permutations', n_memories=2):
  data = collect_data()
  data = data[(data['data_file'] == f"{dataset}.pt") & (data['model'] == model) & (data['n_memories'] == n_memories)]

  pyplot.plot(list(range(1, len(mean_task_performances) + 1)), mean_task_performances)
  pyplot.xlabel("Task Num")
  pyplot.ylabel("Mean Performance on Observed Tasks")
  pyplot.ylim([0, 1])
  pyplot.xticks(np.arange(1, 20 + 1))
  pyplot.savefig(f"{basepath}.png")


if __name__ == '__main__':
  fire.Fire()
