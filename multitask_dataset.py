import torch
from torch.utils.data import Dataset
import random
from task_weighting import *
from logger import Logger

class MultiTaskDataset(Dataset):
    def __init__(self):
        self.__datasets = {}
        self.__iter =None
        self.__total_samples = []

    def add_task(self, name, dataset):
        self.__datasets[name] = dataset

    def build(self,batch, randomize=True, multiple=True):

        dataset_sizes =  map_values(lambda v: len(v),self.__datasets)

        Logger.debug(dataset_sizes)
        multiples, self.__weights = multiples_and_weights(dataset_sizes)


        self.__total_samples=[]
        for d in  self.__datasets :
            indices_final=[]

            if randomize:
                if multiple:
                    for _ in range(multiples[d]):
                        Logger.debug('multiple for {} is {}.  Weight is {}'.format(d,multiples[d], self.__weights[d] ))
                        indices = torch.randperm(len(self.__datasets[d])).tolist()
                        indices_final+=indices
                else:
                    indices_final=torch.randperm(len(self.__datasets[d])).tolist()

            else:
                if multiple:
                    for _ in range(multiples[d]):
                        indices = list(range(0,len(self.__datasets[d])))
                        indices_final+=indices
                else:
                    indices_final=list(range(0,len(self.__datasets[d])))

            samples = [indices_final[i:i + batch] for i in range(0, len(indices_final), batch)]

            for s in samples:
                if len(s) == batch:
                    self.__total_samples.append({'name': d, 'samples':s })

        Logger.debug('total samples {}'.format(len(self.__total_samples)))

        if randomize:
            random.shuffle(self.__total_samples)


        return self

    def get_weights(self,name):
        return self.__weights[name] if self.__weights.__len__()>1 else 1

    def __len__(self):
        return len(self.__total_samples)

    def __getitem__(self, index):
        batch_name =  self.__total_samples[index]['name']
        batch_samples =self.__total_samples[index]['samples']
        batch = [self.__datasets[batch_name][i] for i in batch_samples]
        data = self.collate_fn(batch)

        return (data, batch_name)

    def collate_fn(self, data):
        data = list(zip(*data))
        input_ids =torch.stack(data[0], 0)
        token_type_ids = torch.stack(data[1], 0)
        attention_mask = torch.stack(data[2], 0)
        position_ids = torch.stack(data[3], 0)
        labels = torch.stack(data[4],0)
        masks = torch.stack(data[5], 0)
        index=torch.tensor(data[6])

        return [input_ids,token_type_ids,attention_mask,position_ids,labels,masks,index]
