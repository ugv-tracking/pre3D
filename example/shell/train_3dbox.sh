python example/env/train_end2end.py \
--pretrained model/vgg --epoch 2 \
--prefix model/3dbox/3dbox --begin_epoch 1 --end_epoch 500 \
--lr 0.0001 --lr_step 30000 --gpus 0 --root_path 'data' \
--dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'val' \
--bbox \
--frequent 20 2>&1 | tee -a train_rcnn.log
#--resume \
