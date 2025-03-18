import cv2
import time
import pymysql
import numpy as np
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
from collections import defaultdict, deque


def load_labels(label_path):
	"""Loads the labels.
	
	Args:
		label_path: path to the label file
		
	Returns:
		list of the labels
	"""
	
	with open(label_path, 'r') as f:
		return [line.strip() for line in f.readlines()]
		
		
def prepare_interpreter(model_path, num_threads):
	"""Prepare the tensorflow model runtime
	
	Args:
		model_path: path to tflite model
		num_threds: Number of threads used by interpreter
		
	Returns:
		A dictionary containing interpreter interface, tensors index and 
		input dimension i.e (width, height)
	"""
	interpreter_info = {}
	
	interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
	interpreter.allocate_tensors()
	
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	
	input_height = input_details[0]["shape"][1]
	input_width = input_details[0]["shape"][2]
	input_dimension = (input_width, input_height)
	
	tensors_index = {}
	tensors_index["input_tensor_index"] = input_details[0]["index"]
	tensors_index["box_tensor_index"] = output_details[0]["index"]
	tensors_index["class_tensor_index"] = output_details[1]["index"]
	tensors_index["score_tensor_index"] = output_details[2]["index"]
	
	interpreter_info["Interpreter"] = interpreter
	interpreter_info["tensors_index"] = tensors_index
	interpreter_info["input_dim"] = input_dimension
	
	return interpreter_info
	
	
def get_tensor_output(interpreter_info):
	"""	Get the Tensor ouput based on the the tensor index
	
	Args:
		interpreter_info: Dictionary contains information about interpreter
		
	Returns:
		An array of box dimensions, classes and scores
	"""
	interpreter = interpreter_info["Interpreter"]
	boxes = interpreter.get_tensor(interpreter_info["tensors_index"]["box_tensor_index"])
	classes = interpreter.get_tensor(interpreter_info["tensors_index"]["class_tensor_index"])
	scores = interpreter.get_tensor(interpreter_info["tensors_index"]["score_tensor_index"])
	boxes, classes, scores = np.squeeze(boxes), np.squeeze(classes), np.squeeze(scores)
	
	return boxes, classes, scores
	

def add_pad(image):
	""" Add pad and create border
	
	Args:
		image: Loaded image

		
	Returns:
		Image with borders
	"""
	if image is None:
		print("Provide an Image")
		exit(1)
	img_height, img_width = image.shape[:2]
	pad = abs(img_height - img_width) // 2
	x_pad = pad if img_height > img_width else 0
	y_pad = pad if img_width > img_height else 0
	img_padded = cv2.copyMakeBorder(image, top=y_pad, bottom=y_pad,
									left=x_pad, right=x_pad,
									borderType=cv2.BORDER_CONSTANT,
									value=(0, 0, 0, 0))
									
	return img_padded
	
	
def process_image(image, input_dimension):
	""" Process image for inference
	
	Args:
		image: Loaded Image 
		input_dimension: Input dimension to model
		
	Returns:
		An array of the image pixels, padded_image
	"""
	img_padded = add_pad(image)
	
	imgRGB = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
	img_resized = cv2.resize(imgRGB, input_dimension, interpolation=cv2.INTER_AREA)
	
	input_data = np.expand_dims(img_resized, axis=0)
	
	return input_data, img_padded
	
	
def generate_colors(num_of_colors):
	""" Generate an  array of RGB color
		
	Args:
		num_of_colors: Number of Colors
		
	Returns:
		An array of RGB colors of shape (num_of_colors, 3)		
	"""	
	
	return np.random.randint(255, size=(num_of_colors, 3))


def add_label(image, box, class_name, score, color):
	""" Add labels to the image
	
	Args:
		image: Loaded image
		box: Array of the coordinates of the box
		class_name: Label 
		score: Model Score
		color: Color of the label
	"""
	font_color = (255, 255, 255) if sum(color) < 144 * 3 else (0, 0, 0)
	FONT_STYLE = cv2.FONT_HERSHEY_SIMPLEX
	FONT_THICKNESS = 2
	FONT_SIZE = 1
	
	img_height, img_width = image.shape[:2]
	min_y = round(box[0] * img_height)
	min_x = round(box[1] * img_width)
	max_y = round(box[2] * img_height)
	max_x = round(box[3] * img_width)
	
	# Add Bounding Box to object
	cv2.rectangle(image, (min_x, min_y), (max_x, max_y), color, 2)
	
	# Add Label
	label = f"{class_name} {score * 100:.2f}%"
	labelsize, baseline = cv2.getTextSize(label, FONT_STYLE, FONT_SIZE, FONT_THICKNESS)
	cv2.rectangle(image,
				(min_x, min_y + labelsize[1]),
				(min_x + labelsize[0], min_y - baseline),
				color, cv2.FILLED)
	text_location = (min_x, min_y + labelsize[1])
	cv2.putText(image, label, text_location, FONT_STYLE, FONT_SIZE,
				font_color, FONT_THICKNESS)
	 
	
def add_id(image, box, class_name, score, color, tracker):
	""" Assign Id to label
	
	Args:
		image: Loaded image
		box: Array of the coordinates of the box
		class_name: Label 
		score: Model Score
		color: Color of the label
		tracker: assigns Id to label
	"""
	font_color = (255, 255, 255) if sum(color) < 144 * 3 else (0, 0, 0)
	FONT_STYLE = cv2.FONT_HERSHEY_SIMPLEX
	FONT_THICKNESS = 2
	FONT_SIZE = 1
	CIRCLE_RADIUS = 10
	
	img_height, img_width = image.shape[:2]
	min_y = round(box[0] * img_height)
	min_x = round(box[1] * img_width)
	max_y = round(box[2] * img_height)
	max_x = round(box[3] * img_width)
	mid_x, mid_y = ((min_x + max_x)//2, (min_y + max_y)//2)
	first_line = int(img_height//1.6)
	second_line = int(img_height//1.5)
	
	id_ = tracker.assign_id((min_x, min_y), (max_x, max_y))
	

	cv2.line(image, (0, first_line), (img_width, first_line), color, 4)
	cv2.line(image, (0, second_line), (img_width, second_line), color, 4)

	if first_line < mid_y + CIRCLE_RADIUS and first_line > mid_y - CIRCLE_RADIUS:
		tracker.populate_going_down(id_)
	
	if id_ in tracker.going_down:	
		if second_line < mid_y + CIRCLE_RADIUS and second_line > mid_y - CIRCLE_RADIUS:
			tracker.populate_gone_down(id_)
			cv2.circle(image, (mid_x, mid_y), CIRCLE_RADIUS, color, cv2.FILLED)
		
			# Add Label
			label = f"{class_name} {id_}"
			labelsize, baseline = cv2.getTextSize(label, FONT_STYLE, FONT_SIZE, FONT_THICKNESS)
			cv2.rectangle(image,
						(min_x, min_y + labelsize[1]),
						(min_x + labelsize[0], min_y - baseline),
						color, cv2.FILLED)
			text_location = (min_x, min_y + labelsize[1])
			cv2.putText(image, label, text_location, FONT_STYLE, FONT_SIZE,
						font_color, FONT_THICKNESS)
			
				
def calculate_speed_fixed_distance_measure_time(image, box, class_name, score, color, tracker):
	""" Calculate speed of the moving object based on 
		measuring the travelling time in a given unit
		of distance.
	
	Args:
		image: Loaded image
		box: Array of the coordinates of the box
		class_name: Label 
		score: Model Score
		color: Color of the label
		tracker: assigns Id to label
	"""
	speed = -1
	font_color = (255, 255, 255) if sum(color) < 144 * 3 else (0, 0, 0)
	FONT_STYLE = cv2.FONT_HERSHEY_SIMPLEX
	FONT_THICKNESS = 2
	FONT_SIZE = 1
	CIRCLE_RADIUS = 10
	
	img_height, img_width = image.shape[:2]
	min_y = round(box[0] * img_height)
	min_x = round(box[1] * img_width)
	max_y = round(box[2] * img_height)
	max_x = round(box[3] * img_width)
	mid_x, mid_y = ((min_x + max_x)//2, (min_y + max_y)//2)
	first_line = int(img_height//1.6)
	second_line = int(img_height//1.5)
	
	id_ = tracker.assign_id((min_x, min_y), (max_x, max_y))
	

	cv2.line(image, (0, first_line), (img_width, first_line), color, 4)
	cv2.line(image, (0, second_line), (img_width, second_line), color, 4)

	if first_line < mid_y + CIRCLE_RADIUS and first_line > mid_y - CIRCLE_RADIUS:
		tracker.populate_going_down(id_)
		tracker.populate_object_time_stamp(time.time())
	
	if id_ in tracker.going_down:	
		if second_line < mid_y + CIRCLE_RADIUS and second_line > mid_y - CIRCLE_RADIUS:
			tracker.populate_gone_down(id_)
			tracker.populate_object_time_stamp(time.time(), before=False)
			
			distance = 30 # Meters
			speed = tracker.calculate_object_speed_in_km_per_hr_model_one(distance)
			cv2.circle(image, (mid_x, mid_y), CIRCLE_RADIUS, color, cv2.FILLED)
		
			# Add Label
			label = f"{class_name} {id_} {speed = } km/hr"
			labelsize, baseline = cv2.getTextSize(label, FONT_STYLE, FONT_SIZE, FONT_THICKNESS)
			cv2.rectangle(image,
						(min_x, min_y + labelsize[1]),
						(min_x + labelsize[0], min_y - baseline),
						color, cv2.FILLED)
			text_location = (min_x, min_y + labelsize[1])
			cv2.putText(image, label, text_location, FONT_STYLE, FONT_SIZE,
						font_color, FONT_THICKNESS)
						
	return speed

def calculate_speed_fixed_time_measure_distance(image, box, class_name, score, color, tracker):
	""" Calculate speed of the moving object based on 
		measuring the moving distance in a given unit
		of time.
	
	Args:
		image: Loaded image
		box: Array of the coordinates of the box
		class_name: Label 
		score: Model Score
		color: Color of the label
		tracker: assigns Id to label
	"""
	speed = -1
	font_color = (255, 255, 255) if sum(color) < 144 * 3 else (0, 0, 0)
	FONT_STYLE = cv2.FONT_HERSHEY_SIMPLEX
	FONT_THICKNESS = 2
	FONT_SIZE = 1
	CIRCLE_RADIUS = 10
	
	img_height, img_width = image.shape[:2]
	min_y = round(box[0] * img_height)
	min_x = round(box[1] * img_width)
	max_y = round(box[2] * img_height)
	max_x = round(box[3] * img_width)
	mid_x, mid_y = ((min_x + max_x)//2, (min_y + max_y)//2)
	
	# Add Bounding Box
	cv2.rectangle(image, (min_x, min_y), (max_x, max_y), color, 2)
	
	id_ = tracker.assign_id((min_x, min_y), (max_x, max_y))
	tracker.populate_coordinates_y(id_, mid_y)
	
	
	if len(tracker.coordinates_y[id_]) < tracker.coordinates_y[id_].maxlen / 2:
		label = f"{class_name} {id_}"
	else:
		speed = tracker.calculate_object_speed_in_km_per_hr_model_two(id_)
		label = f"{class_name} {id_} {speed} km/hr"
			
		
	# Add Label
	labelsize, baseline = cv2.getTextSize(label, FONT_STYLE, FONT_SIZE, FONT_THICKNESS)
	cv2.rectangle(image, (min_x, min_y + labelsize[1]), 
				  (min_x + labelsize[0], min_y - baseline),
				  color, cv2.FILLED)
	text_location = (min_x, min_y + labelsize[1])
	cv2.putText(image, label, text_location, FONT_STYLE, FONT_SIZE, font_color, FONT_THICKNESS)
						
	return speed
	
def set_camera(display_width, display_height, fps):
	""" Set Camera with configuration
	
	Args:
		display_width: Display width
		display_height: Display height
		fps: frames per second
		
	Returns:
		Camera Object
	"""
	picam = Picamera2()
	picam.preview_configuration.main.size = (display_width, display_height)
	picam.preview_configuration.main.format="RGB888"
	picam.preview_configuration.controls.FrameRate = fps
	picam.preview_configuration.align()
	picam.configure("preview")
	
	return picam

	
def add_text_to_image(image, text, text_location, style, size, color, thickness):
	""" Add text to loaded Image
	
	Args:
		image: Loaded Image
		text: Text to be added
		text_location: Location to place text:
		style: Font Style
		size: Font Size
		color: Font Color
		thickness: Font Thickness
	"""
	cv2.putText(image, text, text_location, style, size, color, thickness)


def connect_to_database(host, user, password, database=None):
	""" Connect to mysql Database
	
	Args:
		host: Host where the database is hosted
		user: Username log in as
		password: User's Password
		database: Database to use, None to not use a specific one
		
	Return:
		Connection Object
	"""
	return pymysql.connect(host=host, user=user, password=password,
						   database=database)
						   

def populate_database(cursor, table, speed, date_time):
	""" Populate the database with speed and date
	
	Args:
		Cursor: Database Object to execute SQL statements
		table: Table in database
		speed: Speed of trespasser
		date_time: Date_time Trespassed
	"""
	statement = f"INSERT INTO {table}(speed, trespassed_at) VALUES ({speed}, '{date_time}');"
	print(cursor.execute(statement))
