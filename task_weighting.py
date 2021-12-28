# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Code for weighting examples from different tasks based on dataset sizes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def map_values(f, d):
    return {k: f(v) for k, v in d.items()}


def map_kv(f, d):
    return {k: f(k, v) for k, v in d.items()}


def multiples_and_weights(dataset_sizes,task_weight_exponent=0.75):

  def normalize(d):
    total = float(sum(d.values()))
    return map_values(lambda v: v / total, d)

  dataset_weights = map_values(lambda s: s **  task_weight_exponent,
                               dataset_sizes)
  dataset_weights = normalize(dataset_weights)
  task_large = max(dataset_sizes,key=dataset_sizes.get)
  correction = (dataset_sizes[task_large] / dataset_weights[task_large]) if len(dataset_sizes)>1 else dataset_sizes[task_large]
  dataset_tgts =  map_values(lambda v: v * correction, dataset_weights)
  dataset_multiples = map_kv(
      lambda task, tgt: round((tgt + 0.01) / dataset_sizes[task]), dataset_tgts)
      # lambda task, tgt: (tgt / dataset_sizes[task]), dataset_tgts)
  new_dataset_sizes = map_kv(
      lambda task, multiple: dataset_sizes[task] * multiple, dataset_multiples)
  weights_after_multiples = map_values(
      lambda v: v * len(dataset_sizes),
      normalize({task: dataset_weights[task] / new_dataset_sizes[task]
                 for task in new_dataset_sizes}))

  return dataset_multiples, weights_after_multiples


def get_task_multiple(task):
    multiples, _ = multiples_and_weights(task.config)
    return int(multiples[task.name] + 1e-5)

def get_task_weights(config, sizes):
  """Get task weights according to dataset sizes."""

  if config.dataset_multiples:
    _, weights = multiples_and_weights(config)
    return weights


if __name__ == '__main__':
    multiples_and_weights({'ner': 30673, 're': 6750})
