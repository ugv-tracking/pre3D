python example/env/train_end2end.py \
--pretrained model/final --epoch 1 \
--prefix model/3dbox/3dbox --begin_epoch 0 --end_epoch 500 \
--lr 0.0002 --lr_step 30000 --gpus 6 --root_path 'data' \
--dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'val' \
--bbox \
--frequent 20 2>&1 | tee -a train_rcnn.log
#--resume \
