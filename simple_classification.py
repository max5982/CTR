#!/usr/bin/env python3

import cv2
import numpy as np
from openvino.runtime import Core

ie = Core()
model = ie.read_model(model="model/model_cls_eff_v2_s.xml")
compiled_model = ie.compile_model(model=model, device_name="CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
N, C, H, W = input_layer.shape

# Load input
#image = cv2.imread(filename="images/cropped_ok.jpg")
image = cv2.imread(filename="images/cropped_bad.jpg")

# Pre-processing - resize & reshape to the network input shape
resized_image = cv2.resize(src=image, dsize=(W, H))
input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)

# Inference
result_infer = compiled_model([input_image])[output_layer]

# Post-processing
result_index = np.argmax(result_infer)
imagenet_classes = open("label.txt").read().splitlines()

# Result
score = result_infer[0][result_index]
ret = imagenet_classes[result_index]

# Text
font      = cv2.FONT_HERSHEY_SIMPLEX
position  = (10, 200)
fontScale = 0.8
fontColor = (255,0,255)
thickness = 2
lineType  = 1

print("{} - score: {}".format(ret, score))
cv2.putText(resized_image,'{} ({:.4f})'.format(ret, score),
    position,
    font,
    fontScale,
    fontColor,
    thickness,
    lineType)

# Display
cv2.imshow("ret", resized_image)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()
