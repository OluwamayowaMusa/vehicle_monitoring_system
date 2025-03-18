# Vehicle Monitoring System

This project is designed to monitor vehicles using machine learning models. It utilizes TensorFlow Lite for object detection and provides functionalities to process both live video feeds and static images.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Counting Vehicles in Images](#counting-vehicles-in-images)
  - [Counting Vehicles in Live Video](#counting-vehicles-in-live-video)
- [Files in the Repository](#files-in-the-repository)
- [License](#license)

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/OluwamayowaMusa/vehicle_monitoring_system.git
   cd vehicle_monitoring_system
   ```

2. **Set Up the Python Environment:**

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
   ```

3. **Install Dependencies with uv:**

   Ensure you have `uv` installed, then install the dependencies:

   ```bash
   pip install uv
   uv pip install -r requirements.txt
   ```

   If a `requirements.txt` file is not present, you can manually install the dependencies listed in `pyproject.toml`.

## Usage

### Counting Vehicles in Images

To count vehicles in a static image:

```bash
python count_image.py --image_path path/to/your/image.jpg
```

Replace `path/to/your/image.jpg` with the actual path to your image file.

### Counting Vehicles in Live Video

To count vehicles using a live video feed (e.g., from a webcam):

```bash
python count_live.py
```

This will start the video stream and display the detected vehicles in real-time.

## Files in the Repository

- `.gitignore`: Specifies files and directories to be ignored by Git.
- `.python-version`: Indicates the Python version used for the project.
- `README.md`: This file, providing an overview of the project.
- `TF_Lite_Object_Detection.py`: Contains functions and classes related to TensorFlow Lite object detection.
- `count_image.py`: Script to detect and count vehicles in a static image.
- `count_live.py`: Script to detect and count vehicles in a live video feed.
- `efficientdet_lite0.tflite`: TensorFlow Lite model used for object detection.
- `helpers.py`: Helper functions used across the project.
- `labelmap.txt`: Maps class labels to their corresponding indices in the model.
- `main.py`: Main script that might serve as an entry point for the project.
- `pyproject.toml`: Contains project metadata and dependencies.
- `uv.lock`: Lock file related to dependency management.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

*Note: Ensure that your environment meets all the requirements specified in the `pyproject.toml` or `requirements.txt` file before running the scripts.*


