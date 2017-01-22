rm data/cache/*
rm data/kitti/imglists/val_image.list
ls data/kitti/images/ >> data/kitti/imglists/val_image.list
python tools/generate_trainval_list.py
