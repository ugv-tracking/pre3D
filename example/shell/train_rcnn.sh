python example/env/train_end2end.py \
--pretrained model/vgg21/vgg --epoch 4 \
--prefix model/vgg21/vgg --begin_epoch 4 --end_epoch 20 \
--lr 0.00001 --lr_step 30000 --gpus 0 --root_path 'data' \
--num_class 21 \
--resume \
--dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'val' \
--frequent 20 2>&1 | tee -a train_rcnn.log

'''
python example/env/train_end2end.py \
--pretrained model/vgg4/vgg --epoch 4 \
--prefix model/vgg4/vgg --begin_epoch 4 --end_epoch 20 \
--lr 0.00001 --lr_step 30000 --gpus 0 --root_path 'data' \
--num_class 21 \
--dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'val' \
--frequent 20 2>&1 | tee -a train_rcnn.log
'''
