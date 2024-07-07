# mora-bprs/ SAM-model

## Getting Started

Currently, `fastSAM.ipynb` is partially ready for public consumption.

### 1. Google Colab

- Open the notebook in Colab
  
  <a target="_blank" href="https://colab.research.google.com/github/mora-bprs/sam-model/blob/main/fast-sam.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Google Colab"/>
  </a>

- Modify the `COLAB` and `INITIALIZED` variables accordingly.
- Menu -> Runtime -> Run All -> Sit back and relax.

### 2. Local Environment for Real-time Inference

- Make sure you have Python version >= 3.10:
  ```shell
    python3 --version
  ```

- Create a Python virtual environment (preferably in the working directory):
  ```shell
    python3 -m venv .venv
  ```
- Activate the Python environment according to your shell.
- Install pip dependencies:
  ```shell
    pip install -r requirements.txt
  ```
- Run the notebook in VSCode with Python and Jupyter extensions or in the Jupyter environment.
- Set the `INITIALIZED` variable accordingly.

## Model Checkpoints

Source: [https://pypi.org/project/segment-anything-fast/](https://pypi.org/project/segment-anything-fast/)

Click the links below to download the checkpoint for the corresponding model type.

- `default` or `FastSAM`: [YOLOv8x based Segment Anything Model](https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view?usp=sharing) | [Baidu Cloud (pwd: 0000)](https://pan.baidu.com/s/18KzBmOTENjByoWWR17zdiQ?pwd=0000).