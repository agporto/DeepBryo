import cv2
import argparse
import os
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.core import encode_mask_results
import pandas as pd
import pycocotools.mask as mask_util
import math
import xgboost
import pickle


def initialization():
    """Loads configuration and model for the prediction.

    Returns:
        model: The constructed detector (Initialized from config file).

    """
    config_file = "./configs/swin/mask_config_serve_tiny.py"
    checkpoint_file = "./inference/deepbryo_tiny.pth"
    model = init_detector(config_file, checkpoint_file, device="cuda:0")
    return model


def init_filter():
    autofilter = pickle.load(open("./app/automated_filtering.dat", "rb"))
    return autofilter


def inference(_model, img):
    return inference_detector(_model, img)


def filter_border(img_size, outputs, classes, pad, confidence):
    (boxes, masks) = outputs
    for num, structure in enumerate(boxes):
        if classes == 0:
            idx = [
                idx
                for idx, element in enumerate(structure)
                if condition(element, img_size, pad, confidence)
            ]
            if idx:
                outputs[0][num] = np.delete(outputs[0][num], idx, 0)
                outputs[1][num] = np.delete(outputs[1][num], idx, 0)
        else:
            if num == (classes - 1):
                idx = [
                    idx
                    for idx, element in enumerate(structure)
                    if condition(element, img_size, pad, confidence)
                ]
                if idx:
                    outputs[0][num] = np.delete(outputs[0][num], idx, 0)
                    outputs[1][num] = np.delete(outputs[1][num], idx, 0)
            else:
                idx = [idx for idx, element in enumerate(structure)]
                outputs[0][num] = np.delete(outputs[0][num], idx, 0)
                outputs[1][num] = np.delete(outputs[1][num], idx, 0)
    return outputs


def filter_by_occlusion(_model, outputs, classes, strictness):
    (boxes, masks) = outputs
    encoded_masks = encode_mask_results(masks)
    for num, structure in enumerate(masks):
        if classes == 0 or num == (classes - 1):
            idx = [
                idx
                for idx, element in enumerate(structure)
                if autocondition(
                    encoded_masks[num][idx],
                    element.shape[0],
                    element.shape[1],
                    _model,
                    num,
                    strictness,
                )
            ]
            if idx:
                outputs[0][num] = np.delete(outputs[0][num], idx, 0)
                outputs[1][num] = np.delete(outputs[1][num], idx, 0)
    return outputs


def add_number_id(ouputs, img):
    (boxes, masks) = ouputs
    count = 0
    for num, structure in enumerate(boxes):
        for idx, element in enumerate(structure):
            x = int((element[0] + element[2]) / 2)
            y = int((element[1] + element[3]) / 2)
            cv2.rectangle(img, (x - 5, y - 30), (x + 45, y + 5), (0, 0, 0), -1)
            cv2.putText(
                img,
                str(count),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )
            count += 1
    return img


def condition(x, img, pad, confidence):
    return (
        x[0] < pad[0]
        or x[1] < pad[1]
        or x[2] > (img[1] - pad[2])
        or x[3] > (img[0] - pad[3])
        or x[4] < confidence
    )


def autocondition(element, height, width, model, class_id, strict):
    bbox = mask_util.toBbox(element)
    area, perimeter, solidity, circularity, xc, yc, d1, d2, angle, hu = mask_stats(
        element
    )
    array = np.array(
        [
            [
                class_id,
                area,
                perimeter,
                solidity,
                circularity,
                min(d1, d2) / max(d1, d2),
                height,
                width,
                angle,
                xc,
                yc,
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3]),
            ]
        ],
        dtype=float,
    )
    prediction = model.predict_proba(array)
    return True if prediction[0][0] > strict else False


def mask_stats(mask):
    maskedArr = mask_util.decode(mask)
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    hull = cv2.convexHull(cnt)
    solidity = float(area) / cv2.contourArea(hull)
    rect = cv2.minAreaRect(cnt)
    (xc, yc), (d1, d2), angle = rect
    circularity = 4 * math.pi * area / (perimeter * perimeter)
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments)
    return area, perimeter, solidity, circularity, cX, cY, d1, d2, angle, hu


def summarize(predictions, class_id, classes, filename, scale=None):
    (bboxes, masks) = predictions
    dict_out = []
    numerator = 1 if scale is None else scale * 1000
    for num, structure in enumerate(bboxes):
        if class_id == 0 or num == (class_id - 1):
            for idx, element in enumerate(structure):
                (
                    area,
                    perimeter,
                    solidity,
                    circularity,
                    xc,
                    yc,
                    d1,
                    d2,
                    angle,
                    hu,
                ) = mask_stats(masks[num][idx])
                annotation_info = {
                    "image_id": filename,
                    "category": classes[num],
                    "width(bbox)": (element[2] - element[0]) / numerator,
                    "height(bbox)": (element[3] - element[1]) / numerator,
                    "area": area / numerator**2,
                    "perimeter": perimeter / numerator,
                    "solidity": solidity,
                    "circularity": circularity,
                    "eccentricity": min(d1, d2) / max(d1, d2),
                    "majorAxis": max(d1, d2) / numerator,
                    "minorAxis": min(d1, d2) / numerator,
                    "angle": angle,
                    "center_x": int(xc),
                    "center_y": int(yc),
                    "hu1": hu[0][0],
                    "hu2": hu[1][0],
                    "hu3": hu[2][0],
                    "hu4": hu[3][0],
                    "hu5": hu[4][0],
                    "hu6": hu[5][0],
                    "hu7": hu[6][0],
                    "confidence": element[4],
                    "unit": "pixels" if scale is None else "mm",
                }
                dict_out.append(annotation_info)
        else: 
            pass
    if dict_out:
        df = pd.DataFrame(dict_out)
        return df


def main(args):
    model = initialization()
    autofilter = init_filter()
    classes = [
        "all",
        "autozooid",
        "orifice",
        "avicularium",
        "spiramen",
        "ovicell",
        "ascopore",
        "opesia",
    ]
    model.CLASSES = tuple(classes[1:])

    filter = args["autofilter"]

    for filename in os.listdir(args["input_dir"]):
        out_file = os.path.join(args["out_dir"], filename[:-4] + ".csv")
        if os.path.exists(out_file) == False:
            print(out_file)

            img = cv2.imread(os.path.join(args["input_dir"], filename))
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

            height = img.shape[0]
            width = img.shape[1]

            idx = classes.index(args["class"])
            # Detection
            out = inference(model, img)

            # Filtering
            out_filtered = filter_border(
                [height, width], out, idx, args["padding"], args["confidence"]
            )

            if filter:
                out_filtered = filter_by_occlusion(
                    autofilter, out_filtered, idx, args["strictness"]
                )

            # Encode and summarize results
            encode_result = (out_filtered[0], encode_mask_results(out_filtered[1]))
            df = summarize(
                encode_result, idx, model.CLASSES, filename, scale=args["scale"]
            )
            
            if df is not None:
                df.to_csv(out_file)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--input_dir",
        type=str,
        help="folder containing images to be predicted",
        required=True,
    )
    ap.add_argument(
        "-o",
        "--out-dir",
        type=str,
        help="output folder. if not specified, defaults to current directory",
        required=True,
    )
    ap.add_argument(
        "-c",
        "--class",
        type=str,
        default='all',
        help="output folder. if not specified, defaults to current directory",
    )
    ap.add_argument(
        "-p",
        "--padding",
        nargs = '+',
        type = float,
        default=[0,0,0,0],
        help="remove objects falling within a certain distance from the image border. please provide it as a list in the following order: left, top, right, bottom ",
    )
    ap.add_argument(
        "-t",
        "--confidence",
        type=float,
        default=0.5,
        help="model's confidence threshold (default = 0.5)",
    )
    ap.add_argument(
        "-a",
        "--autofilter",
        action="store_true",
        help="enable autofilter of model predictions",
    )
    ap.add_argument(
        "-s",
        "--strictness",
        type=float,
        default=0.5,
        help="regulated the strictness of the automated filtering algorithm",
    )
    ap.add_argument(
        "-sc",
        "--scale",
        type=float,
        default=None,
        help="pixel-to-mm scaling parameter (default = None)",
    )

    args = vars(ap.parse_args())
    main(args)
