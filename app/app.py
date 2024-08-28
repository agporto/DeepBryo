import cv2
import numpy as np
import streamlit as st
from mmdet.apis import init_detector, inference_detector
from mmdet.core import encode_mask_results
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import PIL
import pycocotools.mask as mask_util
import math
import xgboost
import pickle

PAGE_CONFIG = {"page_title": "DeepBryo", "page_icon": ":o", "layout": "wide"}
st.set_page_config(**PAGE_CONFIG)


@st.cache_resource
def initialization():
    config_file = "../configs/swin/mask_config_serve_tiny.py"
    checkpoint_file = "../inference/deepbryo_tiny.pth"
    model = init_detector(config_file, checkpoint_file, device="cuda:0")
    return model


@st.cache_resource
def init_filter():
    autofilter = pickle.load(open("./automated_filtering.dat", "rb"))
    return autofilter


@st.cache_data
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


def filter_by_occlusion(model, outputs, classes, strictness):
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
                    model,
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
    # Create a canvas component


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


def convert_df(df):
    return df.to_csv().encode("utf-8")


def main():
    model = initialization()
    autofilter = init_filter()

    cola, colb, colc = st.sidebar.columns([0.2, 1.4, 0.2])
    colb.image("../resources/logo_transparent.png", use_column_width=True)

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

    st.sidebar.markdown("## Object Class")
    object_type = st.sidebar.selectbox("Search for which objects?", classes, 0)
    hide_label = st.sidebar.checkbox("Hide Label", value=True)
    hide_bbox = st.sidebar.checkbox("Hide Bounding Box", value=True)
    add_number = st.sidebar.checkbox("Add structure ID", value=False)

    st.sidebar.markdown("## Set Confidence Threshold")
    confidence = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5)

    st.sidebar.markdown("## Filter objects")
    pad_filter = st.sidebar.checkbox(
        "Distance from image border (in pixels)", value=False
    )
    if pad_filter:
        left = st.sidebar.number_input("Left", 0, 2000, 0)
        top = st.sidebar.number_input("Top", 0, 2000, 0)
        right = st.sidebar.number_input("Right", 0, 2000, 0)
        bottom = st.sidebar.number_input("Bottom", 0, 2000, 0)
        padding = [left, top, right, bottom]
    else:
        padding = [0, 0, 0, 0]

    auto_filter = st.sidebar.checkbox("Automated filtering", value=False)
    strictness = st.sidebar.slider("Strictness", 0.0, 1.0, 0.5)

    st.sidebar.markdown("## Output table")
    table = st.sidebar.checkbox("Generate table output", value=False)

    filter_id = st.sidebar.checkbox("Remove object from output by ID", value=True)
    selected_ids = st.sidebar.multiselect("Please select ID numbers", range(200))

    st.sidebar.markdown("## Set scale")
    scale = st.sidebar.checkbox("Set image scale using two points", value=False)

    uploaded_img = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "tif", "bmp"],
        accept_multiple_files=False,
    )

    if uploaded_img:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        height = img.shape[0]
        width = img.shape[1]
        label_font = width / 100 if hide_label is False else 1
        bbox_thickness = 2 if hide_bbox is False else 0
        objects = pd.DataFrame()
        out = inference(model, img)
        out_filtered = filter_border(
            [height, width], out, classes.index(object_type), padding, confidence
        )

        if auto_filter:
            out_filtered = filter_by_occlusion(
                autofilter, out_filtered, classes.index(object_type), strictness
            )

        out_image = model.show_result(
            img, out_filtered, font_size=label_font, thickness=bbox_thickness
        )

        if add_number:
            out_image = add_number_id(out_filtered, out_image)

        encode_result = (out_filtered[0], encode_mask_results(out_filtered[1]))

        if scale:
            number = st.sidebar.number_input("Length (μm)", 0, 2000, 100, 25)
            drawing_mode = st.sidebar.selectbox("Drawing tool:", ["point"])
            stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#FF0000")

            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=3,
                stroke_color=stroke_color,
                background_image=PIL.Image.fromarray(out_image),
                update_streamlit=False,
                width=1500 if width > 1500 else width,
                height=(height / width) * 1500 if width > 1500 else width,
                drawing_mode=drawing_mode,
                key="canvas",
            )
            if canvas_result.json_data is not None:
                objects = pd.json_normalize(canvas_result.json_data["objects"])
                for col in objects.select_dtypes(include=["object"]).columns:
                    objects[col] = objects[col].astype("str")
                if objects.empty is False:
                    if len(objects.left) == 2:
                        ratio = width / 1500 if width > 1500 else 1
                        length = (
                            (
                                (objects.left[0] - objects.left[1]) ** 2
                                + (objects.top[0] - objects.top[1]) ** 2
                            )
                            ** 0.5
                        ) * ratio
                        st.write(
                            f"Success! Your scale is {length/number} pixels per μm."
                        )
                    else:
                        st.write(
                            f"Please select only two points. Use backward arrow to delete points."
                        )
                else:
                    st.write(
                        f"Please select two points: one at the beginning and one at the end of the scale. Then press the leftmost button to submit to the app."
                    )

        else:
            st.image(out_image, caption="Detected objects", use_column_width="auto")

        if table:
            if out_filtered:
                if scale is True and not objects.empty:
                    df = summarize(
                        encode_result,
                        classes.index(object_type),
                        model.CLASSES,
                        uploaded_img.name,
                        scale=length / number,
                    )
                    if df is not None:
                        if filter_id is True and selected_ids is not None:
                            df = df.drop(index=selected_ids)
                        st.write("The following table reports values in mm or mm².")
                        st.dataframe(df)
                        csv = convert_df(df)
                        st.download_button(
                            "Press to Download",
                            csv,
                            f"{uploaded_img.name[:-4]}.csv",
                            "text/csv",
                            key="download-csv",
                        )
                    else:
                        st.write("No objects detected.")
                else:
                    df = summarize(
                        encode_result,
                        classes.index(object_type),
                        model.CLASSES,
                        uploaded_img.name,
                    )
                    if df is not None:
                        if filter_id is True and selected_ids is not None:
                            df = df.drop(index=selected_ids)
                        st.write(
                            "You have not set the scale yet. The current table reports values in pixels."
                        )
                        st.dataframe(df)
                        csv = convert_df(df)
                        st.download_button(
                            "Press to Download",
                            csv,
                            f"{uploaded_img.name[:-4]}.csv",
                            "text/csv",
                            key="download-csv",
                        )
                    else:
                        st.write("No objects detected.")


if __name__ == "__main__":
    main()
