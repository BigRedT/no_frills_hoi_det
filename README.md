# Setup

We will be executing all commands inside the root directory (`.../no_frills_hoi_det/`) that was created when you cloned the repository. 

To begin, we will create a directory in the root directory called `data_symlinks` that would contain symlinks to any data to be used or produced by our code. Specifically we will create 3 symlinks:
- `hico_clean` -> directory where you downloaded HICO-Det dataset
- `hico_processed` -> directory where you want to store processed data required for training/evaluating models
- `hico_exp` -> directory where you want to store outputs of model training and evaluation

Creating these symlinks is useful if your hardware setup constrains where you keep your data. For example, if you want to store the dataset on the local drives, and code, processed files, and experiment data on the NFS to be shared across multiple servers (or risk getting kicked-off the servers by your Admin for hogging network I/O by putting everything on the NFS or hogging local disk space by putting everything on local drives! Or maybe your Admin knows how to setup the NFS and you don't have these problems)


```
mkdir data_symlinks
cd data_symlinks
ln -s <path to hico_clean> ./hico_clean
ln -s <path to hico_processed> ./hico_processed
ln -s <path to hico_exp> ./hico_exp
```

If executed correctly, `ls -l data_symlinks` in the root directory should show something like:
```
hico_clean -> /data/tanmay/hico/hico_det_clean_20160224
hico_exp -> /home/nfs/tgupta6/Code/hoi_det_data/hico_exp
hico_processed -> /home/nfs/tgupta6/Code/hoi_det_data/hico_processed
```

# Download the HICO-Det dataset
We will now download the required data from the [HICO-Det website](http://www-personal.umich.edu/~ywchao/hico/) to `hico_clean`. Here are the links to all the files (version 0160224) you would need to download
- [Images and Annotations](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk) (.tar.gz)
- [List of HOIs](https://drive.google.com/open?id=1ipvRTUF2zpOlHHqzbEb29iwscizoM1CK) (.txt)
- [List of Verbs](https://drive.google.com/open?id=1EeHNHuYyJI-qqDk_-5nay7Mb07tzZLsl) (.txt)
- [List of Objects](https://drive.google.com/open?id=1geCHW-yukOnEPjkiD9n9N5rWGczpzX4p) (.txt)

Extract the images and annotations file which will be download as a tar.gz file using
```
tar xvzf <path to tar.gz file> -C <path to hico_clean directory>
```
Here `-C` flag specifies the target location where the files will be extracted.

After this step output of `ls -l data_symlinks/hico_clean` should look like
```
anno_bbox.mat
anno.mat
hico_list_hoi.txt
hico_list_obj.txt
hico_list_vb.txt
images
README
tools
```
# Process HICO-Det files
The HICO-Det dataset consists of images and annotations stored in the form of .mat and .txt files. Run the following command to quickly convert this data into easy to understand json files which will be written to `hico_processed` directory
```
bash data/hico/process.sh
```
In addition, the `process.sh` performs the following functions:
- It calls `data/hico/split_ids.py` which separates sample ids into train, val, train_val (union of train and val), and test sets.
- It executes `data/hico/hoi_cls_count.py` which counts number of training samples for each HOI category

The splits are needed for both training and evaluation. Class counts are needed only for evaluation to compute mAP of group of HOI classes created based on number of available training examples.

# Run Object Detector (or download the detections we provide)

## Download

## Create your own

### Step 1: Prepare data for running faster-rcnn
```
python -m exp.detect_coco_objects.run --exp exp_detect_coco_objects_in_hico
```
This creates `faster_rcnn_im_in_out.json` file in `hico_exp/detect_coco_objects_in_hico` which specifies paths to the images on which to run the detector and the paths to the target location where the results would be saved. 

For each image with unique <global_id> the object detector writes the following to `hico_processed/faster_rcnn_boxes`:
- <global_id>_scores.npy
- <global_id>_boxes.npy
- <global_id>_fc7.npy
- <global_id>_nms_keep_indices.npy

### Step 2: Run faster-rcnn


### Step 3: Select candidate boxes for each object category from all predictions

For each image, Faster-RCNN predicts class scores (for 80 COCO classes) and box regression offsets (per class) for upto 300 regions. In this step, for each COCO class, we select upto 10 high confidence predictions per class after non-max suppression by running the following:
```
python -m exp.detect_coco_objects.run --exp exp_select_and_evaluate_confident_boxes_in_hico
```

This will create an hdf5 file called `selected_coco_cls_dets.h5py` in `hico_exp/select_confident_boxes_in_hico` directory. More details about the structure of this file can be found [here](exp/detect_coco_objects/data_description.md).

The above command also performs a recall based evaluation of the object detections to see if what fraction of ground truth human and object annotations are recalled by these predictions. These stats are written to the following files in the same directory:
- `eval_stats_boxes.json`: All selected detections irrespective of the predicted class are used for computing recall numbers.
- `eval_stats_boxes_labels.json`: Only selected detections for the corresponding class are used for computing recall. 

As shown in the table below, our run resulted on average 6 human and 94 object (all categories except human) candidate detections per image. `eval_stats_boxes_labels.json` shows recall of the *labelled* detections (for example only the 6 human candidates would be used as predictions to compute human recall), where as `eval_stats_boxes.json` shows recall values if all 100 (6+94) detections were considered as predictions for every human/object category irrespective of the labels predicted by faster-rcnn. Let us refer to this case as *unlabelled*. The idea is that using labels predicted by the detector helps prune out candidates to be examined for any HOI category by limiting to candidates for human and the object category involved in the HOI category. The significant reduction in candidates (and hence reduced computation for our model) is at the expense of a reasonable and expected drop in object recall due to detections missed by the object detector and our selection scheme. 

||Labelled|Unlabelled|
|:--|:------|:--------|
| Avg. Human Candidates | 6 | 100 (=6+94) |
| Avg. Object Candidates | 1.2 (=94/79) | 100 |
| Avg. Connection / Pair Candidates | 0.9 (=568/600) | 16.9 =(10122/600) |
| Human Recall | 86% | 89% |
| Object Recall | 70% | 86% |
| Connection / Pair Recall | 59% | 77% | 
