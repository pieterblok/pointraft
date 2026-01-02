## Installation
Tested on a computer with Ubuntu 24.04 and a NVIDIA GeForce RTX 4090 Laptop GPU. 

Software: Python 3.12.3, CUDA 12.8, Pytorch 2.8, torchvision 0.23.0. 

Test date: 17-12-2025. <br/> <br/>

**Install Python virtual environment for handling of custom software dependencies**
- sudo apt install python3-venv -y
- python3 -m venv ~/prft <br/> <br/>

**Install the custom software packages (in python virtual environment):**
- source ~/prft/bin/activate
- pip3 install open3d==0.19.0
- pip3 install torch==2.8
- pip3 install torchvision==0.23.0 
- pip3 install torch_cluster -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
- pip3 install torch_geometric==2.7.0
- pip3 install opencv-python==4.12.0.88
- pip3 install optuna==4.6.0
- pip3 install wheel
- git clone https://github.com/leonardodalinky/pytorch_fpsample.git
- cd pytorch_fpsample/
- replace the text in **pyproject.toml** to this text:
```python
[build-system]
requires = ["setuptools>=45.0.0", "torch==2.8"]
build-backend = "setuptools.build_meta"
```
- save and close
- replace the install_requires line in **setup.py** to this line:
```python
install_requires=["torch==2.8"],
```
- save and close
- pip3 install --no-deps --verbose .
- cd to root directory (otherwise it throws a NoneType object origin error)
- python3 -c "import torch_fpsample; print('torch_fpsample import OK')"
