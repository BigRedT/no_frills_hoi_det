import os

import utils.io as io


class HicoConstants(io.JsonSerializableClass):
    def __init__(
            self,
            clean_dir='/home/ssd/hico_det_clean_20160224',
            proc_dir='/home/ssd/hico_det_processed_20160224'):
        self.clean_dir = clean_dir
        self.proc_dir = proc_dir

        # Clean constants
        self.anno_bbox_mat = os.path.join(self.clean_dir,'anno_bbox.mat')
        self.anno_mat = os.path.join(self.clean_dir,'anno.mat')
        self.hico_list_hoi_txt = os.path.join(
            self.clean_dir,
            'hico_list_hoi.txt')
        self.hico_list_obj_txt = os.path.join(
            self.clean_dir,
            'hico_list_obj.txt')
        self.hico_list_vb_txt = os.path.join(
            self.clean_dir,
            'hico_list_vb.txt')
        self.images_dir = os.path.join(self.clean_dir,'images')

        # Processed constants
        self.anno_list_json = os.path.join(self.proc_dir,'anno_list.json')
        self.hoi_list_json = os.path.join(self.proc_dir,'hoi_list.json')
        self.object_list_json = os.path.join(self.proc_dir,'object_list.json')
        self.verb_list_json = os.path.join(self.proc_dir,'verb_list.json')
