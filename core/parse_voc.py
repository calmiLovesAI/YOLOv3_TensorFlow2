import xml.dom.minidom as xdom
from configuration import PASCAL_VOC_DIR, PASCAL_VOC_CLASSES, TXT_DIR
import os


class ParsePascalVOC():
    def __init__(self):
        super(ParsePascalVOC, self).__init__()
        self.all_xml_dir = PASCAL_VOC_DIR + "Annotations"
        self.all_image_dir = PASCAL_VOC_DIR + "JPEGImages"

    # parse one xml file
    def __parse_xml(self, xml):
        obj_and_box_list = []
        DOMTree = xdom.parse(os.path.join(self.all_xml_dir, xml))
        annotation = DOMTree.documentElement
        image_name = annotation.getElementsByTagName("filename")[0].childNodes[0].data

        obj = annotation.getElementsByTagName("object")
        for o in obj:
            o_list = []
            obj_name = o.getElementsByTagName("name")[0].childNodes[0].data
            bndbox = o.getElementsByTagName("bndbox")
            for box in bndbox:
                xmin = box.getElementsByTagName("xmin")[0].childNodes[0].data
                ymin = box.getElementsByTagName("ymin")[0].childNodes[0].data
                xmax = box.getElementsByTagName("xmax")[0].childNodes[0].data
                ymax = box.getElementsByTagName("ymax")[0].childNodes[0].data
                o_list.append(xmin)
                o_list.append(ymin)
                o_list.append(xmax)
                o_list.append(ymax)
                break
            o_list.append(PASCAL_VOC_CLASSES[obj_name])
            obj_and_box_list.append(o_list)
        return image_name, obj_and_box_list

    def __combine_info(self, image_name, box_list):
        line_str = image_name
        line_str += " "
        for box in box_list:
            for item in box:
                item_str = str(item)
                line_str += item_str
                line_str += " "
        line_str = line_str.strip()
        return line_str

    def write_data_to_txt(self, txt_dir):
        for item in os.listdir(self.all_xml_dir):
            image_name, box_list = self.__parse_xml(xml=item)
            print("Writing information of picture {} to {}".format(image_name, txt_dir))
            # Combine the information into one line.
            line_info = self.__combine_info(image_name, box_list)
            line_info += "\n"
            with open(txt_dir, mode="a+") as f:
                f.write(line_info)


if __name__ == '__main__':
    ParsePascalVOC().write_data_to_txt(txt_dir=TXT_DIR)