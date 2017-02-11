import os
import numpy as np
from rcnn.config import config

def generate(list_path, src_path, save_path) :
    name2id = dict()
    # Kitti version
    name2id["DontCare"] = 0
    name2id["Tram"] = 0
    name2id["Misc"] = 0  
    
    if config.NUM_CLASSES == 4:  # In Kitti version
        name2id["Car"] = 1
        name2id["Van"] = 1
        name2id["Truck"] = 1
        name2id["Person_sitting"] = 2
        name2id["Pedestrian"] = 2
        name2id["Cyclist"] = 3
    else:                       # In VOC-21 version
        name2id["Car"] = 7
        name2id["Van"] = 7
        name2id["Truck"] = 7
        name2id["Person_sitting"] = 15
        name2id["Pedestrian"] = 15
        name2id["Cyclist"] = 2


    save_file = open(save_path, 'w')

    CLASSES = config.CLASSES
    assert os.path.exists(list_path), 'Path does not exist: {}'.format(list_path)
    with open(list_path, 'r') as f:
        
        gt_value = [[]]
        for i in range (config.NUM_CLASSES-1):
            gt_value.append([])

        for image_name in f:
            image_name = image_name.strip()
            image_split = image_name.strip().split('.')
            annotation_file = os.path.join(src_path, image_split[0] + '.txt')
            assert os.path.exists(annotation_file), 'Path does not exist: {}'.format(annotation_file)
            print image_name

            box_list = [[]]
            for i in range (config.NUM_CLASSES-1):
                box_list.append([])

            with open(annotation_file, 'r') as af:
                for line in af:

                    label = line.strip().split(' ')
                    class_id = name2id[label[0]]
                    if class_id == 0:
                        continue

                    bbox = label[2:14]
                    bbox[0] = label[-1]

                    # rotation_y(1), alpha(1), bbox(4), dims(3), location(3), length is 12
                    if class_id < config.NUM_CLASSES and class_id > 0:
                        box_list[class_id].append(bbox)
                        gt_value[class_id].append(bbox)

                    print CLASSES[class_id], bbox

            save_file.write(image_split[0] + '.png')
            for class_id in xrange(len(box_list)):
                class_obj = box_list[class_id]
                save_file.write(':')
                for obj in class_obj:
                    for elem in obj:
                        save_file.write(elem + ' ')
            save_file.write('\n')

        # estimate mean value
        if config.NUM_CLASSES == 4:
            mean_car     = np.array(gt_value[1]).astype(float)
            mean_people  = np.array(gt_value[2]).astype(float)
            mean_bicycle = np.array(gt_value[3]).astype(float)
        else:
            mean_car     = np.array(gt_value[7]).astype(float)
            mean_people  = np.array(gt_value[15]).astype(float)
            mean_bicycle = np.array(gt_value[2]).astype(float)

        print 'Save the Ground Truth in Version of ', config.NUM_CLASSES
        print 'car mean is ',     mean_car.mean(0)
        print 'people mean is ',  mean_people.mean(0)
        print 'bicycle mean is ', mean_bicycle.mean(0)

if __name__ == '__main__':

    list_path = "data/kitti/imglists/val_image.list"
    src_path  = "/rawdata/liulingbo/3d_detection/kitti/train_val_dataset/left_eye/training_label/training/label_2/"
    save_path = "data/kitti/imglists/val.lst"
    
    config.NUM_CLASSES = 4
    config.CLASSES = ('__background__', 'car', 'pedestrian', 'cyclist')

    generate(list_path, src_path, save_path)
