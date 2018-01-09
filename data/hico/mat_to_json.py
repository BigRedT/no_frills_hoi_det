import scipy.io as scio

import utils.io as io
from data.hico.hico_constants import HicoConstants


class ConvertMat2Json():
    def __init__(self,const):
        self.const = const
        self.anno = scio.loadmat(hico_const.anno_mat)
        
    def create_hoi_list(self):
        num_hoi = self.anno['list_action'].shape[0]
        hoi_list = [None]*num_hoi
        for i in range(num_hoi):
            hoi_list[i] = {
                'id': str(i+1).zfill(3),
                'object': self.anno['list_action'][i,0][0][0],
                'verb': self.anno['list_action'][i,0][1][0],
            }
        
        return hoi_list

    def convert(self):
        print('Creating hoi list ...')
        hoi_list = self.create_hoi_list()
        io.dump_json_object(hoi_list,self.const.hoi_list_json)

        print('Creating object list ...')
        object_list = sorted(list(set([hoi['object'] for hoi in hoi_list])))
        for i,obj in enumerate(object_list):
            object_list[i] = {
                'id': str(i+1).zfill(3),
                'name': obj
            }
        
        io.dump_json_object(object_list,self.const.object_list_json)
        
        print('Creating verb list ...')
        verb_list = sorted(list(set([hoi['verb'] for hoi in hoi_list])))
        for i,verb in enumerate(verb_list):
            verb_list[i] = {
                'id': str(i+1).zfill(3),
                'name': verb
            }
        
        io.dump_json_object(verb_list,self.const.verb_list_json)
        import pdb; pdb.set_trace()
        


if __name__=='__main__':
    hico_const = HicoConstants()
    converter = ConvertMat2Json(hico_const)
    converter.convert()