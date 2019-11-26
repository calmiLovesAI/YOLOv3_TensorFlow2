from configuration import PASCAL_VOC_DIR, PASCAL_VOC_CLASSES, \
    custom_dataset_classes, custom_dataset_dir, using_dataset


class ParseCfg():

    def get_images_dir(self):
        if using_dataset == "custom":
            return custom_dataset_dir
        elif using_dataset == "pascal_voc":
            return PASCAL_VOC_DIR + "JPEGImages"

    def get_classes(self):
        if using_dataset == "custom":
            return custom_dataset_classes
        elif using_dataset == "pascal_voc":
            return PASCAL_VOC_CLASSES


