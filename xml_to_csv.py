"""
Usage:

Convert XML annotated files into csv
python xml_to_csv.py --xml_path=<PATH TO ANNOTATIONS> --csv_output=<OUTPUT FILE NAME>

Defaults:
--xml_path: annotations/
--csv_output: annotations.csv
"""

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--xml_path',
                    help='Annotation Directory containing all xml files',
                    default="annotations/")
parser.add_argument('--csv_output',
                    help='Name of output csv file',
                    default="annotations.csv")

args = parser.parse_args()


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (
                root.find('filename').text,
                int(root.find('size')[0].text),
                int(root.find('size')[1].text),
                member[0].text,
                int(member[5][0].text),
                int(member[5][1].text),
                int(member[5][2].text),
                int(member[5][3].text)
            )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class',
                   'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = os.path.join(os.getcwd(), args.xml_path)
    print("Reading xml from folder {}".format(args.xml_path))
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv(args.csv_output, index=False)
    print('Successfully converted xml to csv.')


if __name__ == "__main__":
    main()