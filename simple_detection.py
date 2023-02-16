#!/usr/bin/env python3

import cv2
import numpy as np
from openvino.runtime import Core


def convert_result_to_image(bgr_image, resized_image, boxes, threshold=0.3, conf_labels=True):
    # Define colors for boxes and descriptions.
    colors = {"red": (255, 0, 0), "green": (0, 255, 0)}

    # Fetch the image shapes to calculate a ratio.
    (real_y, real_x), (resized_y, resized_x) = bgr_image.shape[:2], resized_image.shape[:2]
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

    # Convert the base image from BGR to RGB format.
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Iterate through non-zero boxes.
    for box in boxes:
        # Pick a confidence factor from the last place in an array.
        conf = box[-1]
        if conf > threshold:
            # Convert float to int and multiply corner position of each box by x and y ratio.
            # If the bounding box is found at the top of the image,
            # position the upper box bar little lower to make it visible on the image.
            (x_min, y_min, x_max, y_max) = [
                int(max(corner_position * ratio_y, 10)) if idx % 2
                else int(corner_position * ratio_x)
                for idx, corner_position in enumerate(box[:-1])
            ]

            # Draw a box based on the position, parameters in rectangle function are: image, start_point, end_point, color, thickness.
            rgb_image = cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), colors["green"], 3)

            # Add text to the image based on position and confidence.
            # Parameters in text function are: image, text, bottom-left_corner_textfield, font, font_scale, color, thickness, line_type.
            if conf_labels:
                rgb_image = cv2.putText(
                    rgb_image,
                    f"{conf:.2f}",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    colors["red"],
                    1,
                    cv2.LINE_AA,
                )

    return rgb_image


ie = Core()
model = ie.read_model(model="model/model_det_ssd.xml")
compiled_model = ie.compile_model(model=model, device_name="CPU")

input_layer_ir = compiled_model.input(0)
output_layer_ir = compiled_model.output("boxes")
# N,C,H,W = batch size, number of channels, height, width.
N, C, H, W = input_layer_ir.shape

# Load input
image = cv2.imread(filename="images/img_ok.png")

# Resize the image to meet network expected input sizes.
resized_image = cv2.resize(image, (W, H))

# Reshape to the network input shape.
input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)

# Create an inference request.
boxes = compiled_model([input_image])[output_layer_ir]

# Remove zero only boxes.
boxes = boxes[~np.all(boxes == 0, axis=1)]

# Fetch the image shapes to calculate a ratio.
(real_y, real_x), (resized_y, resized_x) = image.shape[:2], resized_image.shape[:2]
ratio_x, ratio_y = real_x / resized_x, real_y / resized_y
threshold = 0.5

detected_imgs = []

# Iterate through non-zero boxes.
for box in boxes:
    # Pick a confidence factor from the last place in an array.
    conf = box[-1]
    if conf > threshold:
        # Convert float to int and multiply corner position of each box by x and y ratio.
        # If the bounding box is found at the top of the image,
        # position the upper box bar little lower to make it visible on the image.
        (x_min, y_min, x_max, y_max) = [
            int(max(corner_position * ratio_y, 10)) if idx % 2
            else int(corner_position * ratio_x)
            for idx, corner_position in enumerate(box[:-1])
        ]

        # Draw a box based on the position, parameters in rectangle function are: image, start_point, end_point, color, thickness.
        detected_imgs.append(image[y_min:y_max, x_min:x_max])

out_img = convert_result_to_image(image, resized_image, boxes, threshold, conf_labels=False)

# Display
idx = 0
for detected_img in detected_imgs:
    cv2.imshow("Detected - {}".format(idx), detected_img)
    idx += 1
cv2.imshow("ret", out_img)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()

