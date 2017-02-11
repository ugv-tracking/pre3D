python example/env/train_end2end.py \
--pretrained model/vgg21/vgg --epoch 4 \
--prefix model/3dbox/3dbox --begin_epoch 1 --end_epoch 500 \
--lr 0.0001 --lr_step 30000 --gpus 1 --root_path 'data' \
--dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'val' \
--bbox \
--num_class 21 \
--frequent 20 2>&1 | tee -a train_rcnn.log
#--resume \
