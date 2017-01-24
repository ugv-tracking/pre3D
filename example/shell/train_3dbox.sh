python example/env/train_end2end.py \
--pretrained model/3dbox --epoch 1 \
--prefix model/3dbox/3dbox --begin_epoch 2 --end_epoch 20 \
--lr 0.00001 --lr_step 30000 --gpus 6 --root_path 'data' \
--dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'val' \
--bbox \
--frequent 20 2>&1 | tee -a train_3dbox.log
#--resume
