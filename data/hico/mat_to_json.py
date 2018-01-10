import numpy as np
from tqdm import tqdm
import scipy.io as scio

import utils.io as io
from data.hico.hico_constants import HicoConstants


class ConvertMat2Json():
    def __init__(self,const):
        self.const = const
        self.anno = scio.loadmat(self.const.anno_mat)
        self.anno_bbox = scio.loadmat(self.const.anno_bbox_mat)
        
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

    def get_image_size(self,i,subset):
        W = self.anno_bbox[f'bbox_{subset}'][0,i][1][0,0][0][0,0]
        H = self.anno_bbox[f'bbox_{subset}'][0,i][1][0,0][1][0,0]
        C = self.anno_bbox[f'bbox_{subset}'][0,i][1][0,0][2][0,0]
        image_size = [int(v) for v in [H,W,C]]
        return image_size

    def get_hoi_bboxes(self,i,subset):
        num_hois = self.anno_bbox[f'bbox_{subset}'][0,i][2].shape[1]
        hois = [None]*num_hois
        for j in range(num_hois):
            hoi_data = self.anno_bbox[f'bbox_{subset}'][0,i][2][0,j]
            
            hoi_id = str(hoi_data[0][0,0]).zfill(3)    
        
            num_boxes = hoi_data[1].shape[1]
            human_bboxes = [None]*num_boxes
            for b in range(num_boxes):
                human_bboxes[b] = \
                    [int(hoi_data[1][0,b][k][0,0]-1) for k in [0,2,1,3]]
            
            num_boxes = hoi_data[2].shape[1]
            object_bboxes = [None]*num_boxes
            for b in range(num_boxes):
                object_bboxes[b] = \
                    [int(hoi_data[2][0,b][k][0,0]-1) for k in [0,2,1,3]]

            connections = (hoi_data[3]-1).tolist()

            invis = int(hoi_data[4][0,0])

            hois[j] = {
                'id': hoi_id,
                'human_bboxes': human_bboxes,
                'object_bboxes': object_bboxes,
                'connections': connections,
                'invis': invis,
            }
        
        return hois

    def create_anno_list(self):
        anno_list = []
        for subset in ['train','test']:
            print(f'Adding {subset} data to anno list ...')
            num_samples = self.anno[f'anno_{subset}'].shape[1]
            for i in tqdm(range(num_samples)):
                image_jpg = self.anno[f'list_{subset}'][i][0][0]
                
                if image_jpg.endswith('.jpg'):
                    global_id = image_jpg[:-4]
                else:
                    assert(False), 'Image extension is not .jpg'

                anno = {
                    'global_id': global_id,
                    'image_path_postfix': f'{subset}2015/{image_jpg}',
                    'image_size': self.get_image_size(i,subset),
                    'hois': self.get_hoi_bboxes(i,subset)
                }

                anno['pos_hoi_ids'] = [str(k[0]+1).zfill(3) for k in \
                    np.argwhere(self.anno[f'anno_{subset}'][:,i]==1).tolist()]
                anno['neg_hoi_ids'] = [str(k[0]+1).zfill(3) for k in \
                    np.argwhere(self.anno[f'anno_{subset}'][:,i]==-1).tolist()]

                anno_list.append(anno)

        return anno_list

    def convert(self):
        print('Creating anno list ...')
        anno_list = self.create_anno_list()
        io.dump_json_object(anno_list,self.const.anno_list_json)

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


if __name__=='__main__':
    hico_const = HicoConstants()
    converter = ConvertMat2Json(hico_const)
    converter.convert()