python example/env/train_end2end.py \
--pretrained model/final --epoch 1 \
--prefix model/vgg4/vgg --begin_epoch 1 --end_epoch 20 \
--lr 0.0001 --lr_step 30000 --gpus 5 --root_path 'data' \
--dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'val' \
--frequent 20 2>&1 | tee -a train_rcnn.log
#--resume \
