import os
import torch
import torch.nn as nn

import utils.io as io


class Model(io.WritableToFile):
    def __init__(self):
        super(Model,self).__init__()

    def to_txt(self,dir_name,single_file=False):
        if single_file:
            print('Writing model.txt ...')
            model_txt = os.path.join(dir_name,f'model.txt')
            self.to_file(model_txt)
            return

        for model_name, model_instance in self.__dict__.items():
            print(f'Writing {model_name}.txt ...')
            model_txt = os.path.join(dir_name,f'{model_name}.txt')
            model_instance.to_file(model_txt)

    def __str__(self):
        serialized = ''
        for model_name, model_instance in self.__dict__.items():
            if not isinstance(model_instance,nn.Module):
                continue
            serialized += '-'*80
            serialized += '\n'
            serialized += '\n'
            serialized += model_instance.__str__()
            serialized += '\n'
            serialized += '\n'
        serialized += '-'*80
        return serialized