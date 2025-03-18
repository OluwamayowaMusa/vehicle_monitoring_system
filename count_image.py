import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from helpers import load_labels, prepare_interpreter, process_image
from helpers import generate_colors, add_label, get_tensor_output


# Declare Constants
TEST_IMAGE = "image.jpg"
LABEL_PATH = "labelmap.txt"
MODEL = "efficientdet_lite0.tflite"
LABELS_TO_TRACK = ["car", "motorcycle", "bus", "truck"]
NUM_THREADS = 4
THRESHOLD = 0.3
vehicle_count = {}

# Get interpreter info
interpreter_info = prepare_interpreter(MODEL, NUM_THREADS)

# Process image
image = cv2.imread(TEST_IMAGE)
input_data, image = process_image(image, interpreter_info["input_dim"])

# Allocate tensors
interpreter = interpreter_info["Interpreter"]
interpreter.set_tensor(interpreter_info["tensors_index"]["input_tensor_index"],input_data)
interpreter.invoke()

# Get boxes, classes, scores
boxes, classes, scores = get_tensor_output(interpreter_info)

labels = load_labels(LABEL_PATH)
colors = generate_colors(len(labels))

for box, class_, score in zip(boxes, classes, scores):
	class_name = labels[int(class_)]
	if score < THRESHOLD or not (class_name in LABELS_TO_TRACK):
		continue

	count = vehicle_count.get(class_name, 0)
	if count == 0:
		vehicle_count[class_name] = 1
	else:
		vehicle_count[class_name] += 1
	color = [int(i) for i in colors[int(class_)]]
	add_label(image, box, labels[int(class_)], score, color)
	cv2.imshow("Input", image)
	print(f"{score} {labels[int(class_)]}")

cv2.imwrite("./result.jpg", image)
print(vehicle_count)
cv2.waitKey(0)
cv2.destroyAllWindows()
