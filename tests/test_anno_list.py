import os
import skimage.io as skio

import utils.io as io
from tests.tester import *
import utils.bbox_utils as bbox_utils
from data.hico.hico_constants import HicoConstants


def test_anno_list():
    outdir = '/tmp/test_anno_list'
    io.mkdir_if_not_exists(outdir,recursive=True)
    data_const = HicoConstants()
    anno_list = io.load_json_object(data_const.anno_list_json)
    anno = anno_list[1]
    global_id = anno['global_id']
    img_path = os.path.join(
        data_const.images_dir,
        anno['image_path_postfix'])
    img = skio.imread(img_path)
    for i, hoi_info in enumerate(anno['hois']):
        img_ =  img.copy()
        human_bboxes = hoi_info['human_bboxes']
        object_bboxes = hoi_info['object_bboxes']
        connections = hoi_info['connections']
        for connection in connections:
            human_bbox_id, object_bbox_id = connection
            human_bbox = human_bboxes[human_bbox_id]
            object_bbox = object_bboxes[object_bbox_id]
            bbox_utils.add_bbox(img_,human_bbox,color=[0,0,255])
            bbox_utils.add_bbox(img_,object_bbox,color=[0,255,0])

        out_img_path = os.path.join(
            outdir,
            f'{global_id}_{i}.jpg')
        skio.imsave(out_img_path,img_)


if __name__=='__main__':
    list_tests(globals())