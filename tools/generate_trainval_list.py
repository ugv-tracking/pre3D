import os
from rcnn.config import config

def generate(list_path, src_path, save_path) :
    name2id = dict()
    name2id["DontCare"] = 0
    name2id["Tram"] = 0
    name2id["Misc"] = 0   
    name2id["Car"] = 1
    name2id["Van"] = 1
    name2id["Truck"] = 1
    name2id["Person_sitting"] = 2
    name2id["Pedestrian"] = 2
    name2id["Cyclist"] = 3

    save_file = open(save_path, 'w')

    CLASSES = config.CLASSES
    assert os.path.exists(list_path), 'Path does not exist: {}'.format(list_path)
    with open(list_path, 'r') as f:
        for image_name in f:
            image_name = image_name.strip()
            image_split = image_name.strip().split('.')
            annotation_file = os.path.join(src_path, image_split[0] + '.txt')
            assert os.path.exists(annotation_file), 'Path does not exist: {}'.format(annotation_file)
            print image_name

            box_list = [[], [], [], []]
            is_has_car = False
            with open(annotation_file, 'r') as af:
                for line in af:

                    label = line.strip().split(' ')
                    class_id = name2id[label[0]]
                    if class_id == 0:
                        continue

                    bbox = label[2:14]
                    bbox[0] = label[-1]

                    # rotation_y(1), alpha(1), bbox(4), dims(3), location(3), length is 12
                    '''
                    if class_id == 1 :
                           is_has_car = True
                           box_list[class_id].append(bbox)
                    '''
                    if class_id < 4 and class_id > 0:
                        box_list[class_id].append(bbox)

                    print CLASSES[class_id], bbox

            save_file.write(image_split[0] + '.png')
            for class_id in xrange(len(box_list)):
                class_obj = box_list[class_id]
                save_file.write(':')
                for obj in class_obj:
                    for elem in obj:
                        save_file.write(elem + ' ')
            save_file.write('\n')

if __name__ == '__main__':

    list_path = "data/kitti/imglists/val_image.list"
    src_path  = "/rawdata/liulingbo/3d_detection/kitti/train_val_dataset/left_eye/training_label/training/label_2/"
    save_path = "data/kitti/imglists/val.lst"
    
    generate(list_path, src_path, save_path)
