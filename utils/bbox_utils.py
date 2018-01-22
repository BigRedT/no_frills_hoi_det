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


    


