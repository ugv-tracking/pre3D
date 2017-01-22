python example/env/train_end2end.py \
--network vgg \
--pretrained /data01/hustxly/model/faster_rcnn/kitti_ry_cls_input_up_2/ry_alpha_car_only_reg \
--prefix model/basic \
--epoch 20 --begin_epoch 1 --end_epoch 20 \
--lr 0.00001 --lr_step 30000 --gpus 4 --root_path 'data' \
--dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'val' \
--frequent 20 2>&1 | tee -a train_ry_alpha_car_only_reg.log

