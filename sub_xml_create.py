import xml.etree.ElementTree as ET
import os

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


# files=os.listdir(model_path)
# for file in files:

xml_path = '../Reisch/Reisch.xml'
target_path='Reisch'
# os.makedirs(target_path)
tree_src = ET.parse(xml_path)
root = tree_src.getroot()
SNahts=root.findall("SNaht")
# print(SNahts)
i=0
similar='aaa'+','+'bbb'
for SNaht in SNahts:
#     SNaht.set('Naht_IDs',similar)
#
    print(SNaht.attrib)
    dict={}
    for key,value in SNaht.attrib.items():
        dict[key] = value
        if key=='ID':
            dict['Naht_IDs']=similar
    # print(dict)
    SNaht.attrib.clear()
    for key,value in dict.items():
        SNaht.set(key,value)
tree_src.write('tesing_xml.xml')
    # Konturs=SNaht.findall("Kontur")
    # for Kontur in Konturs:
    #     FRAME_DUMP = ET.Element(root.tag, root.attrib)
    #     SNaht_node = ET.SubElement(FRAME_DUMP, SNaht.tag, SNaht.attrib)
    #     Konturs_node = ET.SubElement(SNaht_node, Kontur.tag)
    #     Punkts=Kontur.findall("Punkt")
    #     for Punkt in Punkts:
    #         print(Punkt)
    #         Konturs_node.append(Punkt)
    #         tree = ET.ElementTree(FRAME_DUMP)
    #         pretty_xml(FRAME_DUMP, '    ', '\n')
    #         xml_name = os.path.join(target_path, 'Reisch_' + str(i) + '.xml')
    #         tree.write('xml_name', encoding="utf-8", xml_declaration=True, short_empty_elements=True)
    #         i += 1
# print(i)
# FRAME_DUMP=ET.Element('FRAME-DUMP',{"VERSION":"1.0","Baugruppe":"20102"})
# SNaht=ET.SubElement(FRAME_DUMP,'SNaht',{"Name":"AutoDetect_2_0","ZRotLock":"0","WkzWkl":"45","WkzName":"MRW510_CDD_10GH"})
# Kontur=ET.SubElement(SNaht,'Kontur')
# Punkt=ET.SubElement(Kontur,'Punkt',{"X":"-14.9999999","Y":"2.749999999999972","Z":"-404.5"})
# Fl_Norm1=ET.SubElement(Punkt,'Fl_Norm',{"X":"-1","Y":"0","Z":"0"})
# Fl_Norm2=ET.SubElement(Punkt,'Fl_Norm',{"X":"0","Y":"1","Z":"0"})
# ROT1=ET.SubElement(Punkt,'Rot',{"X":"90","Y":"0","Z":"0"})
# Ext_Achswerte=ET.SubElement(Punkt,'Ext-Achswerte',{"EA3":"-90"})
# Frames=ET.SubElement(SNaht,'Frames')
# Frame=ET.SubElement(Frames,'Frame')
# POS=ET.SubElement(Frame,'Pos',{"X":"-0.1686861956042889","Y":"-4.701284168618442e-16","Z":"-404.4999999999995" })
# XVek=ET.SubElement(Frame,'XVek',{"X":"-14.99999989999999","Y":"2.749999999999954","Z":"-0.9856698064831607" })
# YVek=ET.SubElement(Frame,'YVek',{"X":"-0.691925861227301","Y":"-0.7121912875222437","Z":"0.118415254685646" })
# ZVek=ET.SubElement(Frame,'ZVek',{"X":"-0.7019854485510432","Y":"0.7019854485510427","Z":"0.1201368388346472" })
# ROT2=ET.SubElement(Frame,'Rot',{"X":"90","Y":"0","Z":"0"})
# Ext_Achswerte2=ET.SubElement(Frame,'Ext-Achswerte',{"EA3":"-90"})
#
# tree=ET.ElementTree(FRAME_DUMP)
# pretty_xml(FRAME_DUMP,'    ','\n')
# tree.write("new_tree.xml",encoding="utf-8",xml_declaration=True,short_empty_elements=True)



# list=[]
# for i in range(5):
#     locals()['tree'+str(i)]=i
#     list.append(locals()['tree'+str(i)])
# print(tree0)
