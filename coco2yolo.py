import os
import json
from tqdm import tqdm
import argparse

def convert_bbox(img_width, img_height, bbox):
    """
    Convert bbox from COCO format to YOLO format

    Params:
    ----------
    img_width : int
        width of image
    img_height : int
        height of image
    bbox : list[float]
        bounding box annotation in COCO format: 
        [top left x position, top left y position, bbox_width, bbox_height]

    Returns
    -------
    list[float]
        bounding box annotation in YOLO format: 
        [x_center_rel, y_center_rel, bbox_width_rel, bbox_height_rel]
        (float values relative to width and height of image)
    """

    # Assign coco bbox values to variables
    x_tl, y_tl, w, h = bbox

    # Compute relative scaling factors
    delta_w = 1.0 / img_width
    delta_h = 1.0 / img_height

    # Calculate center coords.
    x_center = x_tl + w / 2.0
    y_center = y_tl + h / 2.0

    # Scale center, w, h coords. -> normalized coords. [0, 1]
    x = x_center * delta_w
    y = y_center * delta_h
    w = w * delta_w
    h = h * delta_h

    return [x, y, w, h]


def convert_coco_json_to_yolo_txt(output_path, json_file):

    with open(json_file) as f:
        json_data = json.load(f)

    # Write labels.txt, which holds names of all classes (one class per line)
    label_file = os.path.join(output_path, "labels.txt")
    with open(label_file, "w") as f:
        for category in tqdm(json_data["categories"], desc="Categories"):
            category_name = category["name"]
            f.write(f"{category_name}\n")

    # Iterate over all images in json_data and retrieve certain properties
    for image in tqdm(json_data["images"], desc="Images"):
        img_id = image["id"]
        img_name = image["file_name"]
        img_width = image["width"]
        img_height = image["height"]

        # Generate YOLO annotations for each image
        anno_in_image = [anno for anno in json_data["annotations"] if anno["image_id"] == img_id]
        anno_txt = os.path.join(output_path, img_name.split(".")[0] + ".txt")
        with open(anno_txt, "w") as f:
            for anno in anno_in_image:
                category = anno["category_id"]
                bbox_COCO = anno["bbox"]
                x, y, w, h = convert_bbox(img_width, img_height, bbox_COCO)
                f.write(f"{category} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    print("COCO Json to YOLO txt successfully converted!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to input COCO JSON file', required=True)
    parser.add_argument('-o', '--output', help='Path to output folder', required=True)
    args = parser.parse_args()

    convert_coco_json_to_yolo_txt(args.output, args.input)