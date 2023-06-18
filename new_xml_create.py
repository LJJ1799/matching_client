import xml.etree.ElementTree as ET
import os
import json

def pretty_xml(element, indent, newline, level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
    if element:  # 判断element是否有子元素
        if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
            # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将element转成list

    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作

file_path='./metrics_withoutR'
files=os.listdir(file_path)
model_path='data/train/xml2/Reisch2'
xml_list=os.listdir(model_path)
target_path='data/train/xml2/Reisch3'
os.mkdir(target_path)
for file in files:
    print(file)
    list1 = []
    with open(file_path+'/'+file,encoding='utf-8') as a:
        result=json.load(a)
        for key,value in result.items():
            if value<float(0.16):
                list1.append(key)
    FRAME_DUMP=ET.Element('FRAME_DUMP', {'VERSION':'1.0','Baugruppe':'Reisch'})
    for i in list1:
        print(i)
        xml_path=os.path.join(model_path,i+'.xml')
        tree_src = ET.parse(xml_path)
        root = tree_src.getroot()
        SNahts = root.findall("SNaht")
        for SNaht in SNahts:
            SNaht_node = ET.SubElement(FRAME_DUMP, SNaht.tag, SNaht.attrib)
            Konturs = SNaht.findall("Kontur")
            for Kontur in Konturs:
                Konturs_node = ET.SubElement(SNaht_node, Kontur.tag)
                Punkts = Kontur.findall("Punkt")
                for Punkt in Punkts:
                    Konturs_node.append(Punkt)
    tree = ET.ElementTree(FRAME_DUMP)
    pretty_xml(FRAME_DUMP, '    ', '\n')
    xml_name = os.path.join(target_path, file.split('.')[0]+'.xml')
    tree.write(xml_name, encoding="utf-8", xml_declaration=True, short_empty_elements=True)

