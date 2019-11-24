from configuration import TXT_DIR, MAX_TRUE_BOX_NUM_PER_IMG, PASCAL_VOC_DIR
import numpy as np
import os


# Return :
# image_name_list : list, length is N (N is the total number of pictures.)
# boxes_array : numpy.ndarrray, shape is (N, MAX_TRUE_BOX_NUM_PER_IMG, 5)
def read_txt():
    print("Start reading txt...")
    image_name_list = []
    boxes_list = []
    index = 0
    with open(TXT_DIR, mode="r") as f:
        for line_info in f.readlines():
            index += 1
            if index > 2000:
                break
            line_info = line_info.strip('\n')
            split_line = line_info.split(" ")
            box_num = (len(split_line) - 1) / 5
            image_name = split_line[0]
            print("Reading {}".format(image_name))
            split_line = split_line[1:]
            boxes = []
            for i in range(MAX_TRUE_BOX_NUM_PER_IMG):
                if i < box_num:
                    box_xmin = int(float(split_line[i * 5]))
                    box_ymin = int(float(split_line[i * 5 + 1]))
                    box_xmax = int(float(split_line[i * 5 + 2]))
                    box_ymax = int(float(split_line[i * 5 + 3]))
                    class_id = int(split_line[i * 5 + 4])
                    boxes.append([box_xmin, box_ymin, box_xmax, box_ymax, class_id])
                else:
                    box_xmin = 0
                    box_ymin = 0
                    box_xmax = 0
                    box_ymax = 0
                    class_id = 0
                    boxes.append([box_xmin, box_ymin, box_xmax, box_ymax, class_id])

            image_name_list.append(os.path.join(PASCAL_VOC_DIR + "JPEGImages", image_name))
            boxes_list.append(boxes)
            boxes_array = np.array(boxes_list, dtype=np.float32)

    return image_name_list, boxes_array
