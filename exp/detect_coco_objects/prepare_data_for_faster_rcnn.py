import os

import utils.io as io


def prepare_hico(exp_const,data_const):
    io.mkdir_if_not_exists(exp_const.exp_dir)
    
    print('Writing constants to exp dir ...')
    data_const_json = os.path.join(exp_const.exp_dir,'data_const.json')
    data_const.to_json(data_const_json)

    exp_const_json = os.path.join(exp_const.exp_dir,'exp_const.json')
    exp_const.to_json(exp_const_json)

    print('Loading anno_list.json ...')
    anno_list = io.load_json_object(data_const.anno_list_json)
    
    print('Creating input json for faster rcnn ...')
    images_in_out = [None]*len(anno_list)
    for i, anno in enumerate(anno_list):
        global_id = anno['global_id']
        image_in_out = dict()
        image_in_out['in_path'] = os.path.join(
            data_const.images_dir,
            anno['image_path_postfix'])
        image_in_out['out_dir'] = os.path.join(
            data_const.proc_dir,
            'faster_rcnn_boxes')
        image_in_out['prefix'] = f'{global_id}_'
        images_in_out[i] = image_in_out

    images_in_out_json = os.path.join(
        exp_const.exp_dir,
        'faster_rcnn_im_in_out.json')
    io.dump_json_object(images_in_out,images_in_out_json)


