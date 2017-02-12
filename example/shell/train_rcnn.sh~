python example/env/train_end2end.py \
--pretrained model/final --epoch 1 \
--prefix model/vgg4/vgg --begin_epoch 0 --end_epoch 500 \
--lr 0.00001 --lr_step 30000 --gpus 1 --root_path 'data' \
--num_class 4 \
--dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'val' \
--frequent 20 2>&1 | tee -a train_rcnn.log
#--resume \
