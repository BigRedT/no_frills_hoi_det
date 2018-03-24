import numpy as np
import skimage.draw as skdraw


def add_bbox(img,bbox,color=[0,0,0],fill=False,alpha=1):
    x1,y1,x2,y2 = bbox
    
    # Clockwise starting from top left
    r = [y1,y1,y2,y2]
    c = [x1,x2,x2,x1]
    
    if fill:
        coords = skdraw.polygon(r,c,shape=img.shape[0:2])
        skdraw.set_color(img,coords,color,alpha=alpha)
        return

    peri_coords = skdraw.polygon_perimeter(r,c,shape=img.shape[0:2])
    skdraw.set_color(img,peri_coords,color,alpha=alpha)


def compute_area(bbox,invalid=None):
    x1,y1,x2,y2 = bbox

    if (x2 <= x1) or (y2 <= y1):
        area = invalid
    else:
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

    return area


def compute_iou(bbox1,bbox2):
    x1,y1,x2,y2 = bbox1
    x1_,y1_,x2_,y2_ = bbox2
    
    x1_in = max(x1,x1_)
    y1_in = max(y1,y1_)
    x2_in = min(x2,x2_)
    y2_in = min(y2,y2_)

    intersection = compute_area(bbox=[x1_in,y1_in,x2_in,y2_in],invalid=0.0)
    area1 = compute_area(bbox1)
    area2 = compute_area(bbox2)
    union = area1 + area2 - intersection
    iou = intersection / union

    return iou 
    
