# Arobotxvision Project

This project aims to provide a computer vision solution for Arobot using Python. It utilizes a Makefile to automate the setup and execution process.

## Prerequisites

- Python 3.9 or later installed on your system.
- Git (optional) if you're cloning the repository.

## Getting Started

1. Clone this repository to your local machine (or download the source code):
   ```bash
   git clone https://github.com/ARoBotX/ARoBotXVision.git
   cd arobotxvision
   ```
2. Set up a virtual environment:
   ```
   make venv
   ```
3. Activate the virtual environment:
   ```
   make activate
   ```

## Installing Dependencies

This project has two sets of dependencies: PyTorch with CUDA support, and other Python packages.

1. Install PyTorch with CUDA support:

```
make install_torch_with_cuda
```

2. Install other dependencies:

```
make install_other_dependencies
```

## Fix pykinect2 package

1. Replace all of your pykinect2 package .py files with the https://github.com/Kinect/PyKinect2/tree/master/pykinect2 repository .py files
2. Change

```
from comtypes import _check_version; _check_version('')
```

to

```
from comtypes import _check_version; _check_version('1.3.1')
```

3. Replace all of the `time.clock()` to `time.time()` in `PyKinectRuntime.py` file

## Running the Project

1. Execute the main.py script:

```
make run
```

## Running All Steps at Once

Alternatively, you can run all the setup steps at once:

```
make all
```

This will create the virtual environment, install dependencies, and execute the main.py script.

## Additional Notes

Make sure to customize paths and configurations in the Makefile and source code as needed for your environment.
For any issues or inquiries, please refer to the project's documentation or contact the project maintainers.

## License

This project is licensed under the MIT License.
