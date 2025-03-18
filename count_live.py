import cv2
import time
import numpy as np
from picamera2 import Picamera2
from helpers import prepare_interpreter, process_image, load_labels, get_tensor_output
from helpers import generate_colors, add_label, set_camera, add_text_to_image

# Declare Parameters
LABEL_PATH = "labelmap.txt"
MODEL = "efficientdet_lite0.tflite"
LABELS_TO_TRACK = ["car", "motorcycle", "bus", "truck"]
NUM_THREADS = 4
THRESHOLD = 0.3
DISPLAY_HEIGHT = 1280
DISPLAY_WIDTH = 720
FRAME_RATE = 25.0
vehicle_count = {}

# Text Properties
TEXT_LOCATION = (20, 60)
FONT_STYLE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 1
FONT_THICKNESS = 2
FONT_COLOR = [int(i) for i in generate_colors(1)[0]]

# Get interpreter info
interpreter_info = prepare_interpreter(MODEL, NUM_THREADS)
interpreter = interpreter_info["Interpreter"]

labels = load_labels(LABEL_PATH)
colors = generate_colors(len(labels))

# Set Camera
picam = set_camera(DISPLAY_WIDTH, DISPLAY_HEIGHT, FRAME_RATE)
picam.start()

time_start = time.time()
fps = 0
while True:
	frame = picam.capture_array()
	frame = cv2.flip(frame, -1) # Flip Image by 180 degrees
	
	input_data, image = process_image(frame, interpreter_info["input_dim"])
	
	# Allocate Tensors
	interpreter.set_tensor(interpreter_info["tensors_index"]["input_tensor_index"], input_data)
	interpreter.invoke()
	
	# Get boxes, classes, scores
	boxes, classes, scores = get_tensor_output(interpreter_info)
	
	for box, class_, score in zip(boxes, classes, scores):
		class_name = labels[int(class_)]
		if score < THRESHOLD or not (class_name in LABELS_TO_TRACK):
			continue
		
		color = [int(i) for i in colors[int(class_)]]
		add_label(image, box, labels[int(class_)], score, color)
		text =  f"{round(fps)} FPS "
		add_text_to_image(image, text, TEXT_LOCATION, FONT_STYLE, FONT_SIZE,
					  FONT_COLOR, FONT_THICKNESS)
		cv2.imshow("Input", image)
		print(f"{score} {labels[int(class_)]}")
	
	if cv2.waitKey(1) == ord('q'):
		break
		
	print(vehicle_count)
	# Calculate Frames Per Second	
	time_end = time.time()
	loop_time = time_end - time_start
	fps = 0.9*fps + 0.1*(1/loop_time)
	time_start = time.time()
	
cv2.destroyAllWindows()
