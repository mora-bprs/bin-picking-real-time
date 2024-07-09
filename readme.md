# Bin-Picking-Real-Time

Here's the real-time video demonstration using the Fast-SAM model.

<iframe width="560" height="315" src="https://www.youtube.com/embed/pogcyD64Qgk?si=ghniwcxx3rYkDeWN" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Getting Started

There are two ways that you can use the model:

1. Using Google Colab.

- You can have a look at the notebook and run it in Google Colab.
- Not recommended for real-time inference.

2. Using a local environment for real-time inference.

- You can clone the repository and run the python script for real-time inference.

### 1. Google Colab

- Open the notebook in Colab <a target="_blank" href="https://colab.research.google.com/github/mora-bprs/sam-model/blob/main/fast-sam.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Google Colab"/>
</a>

- Modify the `COLAB` and `INITIALIZED` variables accordingly.
- Menu -> Runtime -> Run All -> Sit back and relax.

### 2. Local Environment for Real-time Inference (For Windows)

- Clone the repository

```bash
git clone https://github.com/mora-bprs/bin-picking-real-time.git
cd bin-picking-real-time
```

- Create a python environment
  
```bash
python -m venv bin-venv
bin-venv\Scripts\Activate.ps1
```

- Install the required packages. About 300MB of data will be downloaded.

```bash
pip install -r requirements.txt
```

- Run the python script for real-time inference

```bash
python smooth_main_rt.py
```

- Press 'q' to exit the real-time inference.

- Run the following command to deactivate the virtual environment

```bash
deactivate
```

## Possible Errors & Solutions

- You have to install python and venv if not installed
- Python version used:  `3.10.7`
- Install all the requirements in a virtual environment and then run the scripts
- Change camera index if there are multiple cameras connected to the system (`default` is  `0`)

## Model Checkpoints

Source: [https://pypi.org/project/segment-anything-fast/](https://pypi.org/project/segment-anything-fast/)

Click the links below to download the checkpoint for the corresponding model type.

- `default` or `FastSAM`: [YOLOv8x based Segment Anything Model](https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view?usp=sharing) | [Baidu Cloud (pwd: 0000)](https://pan.baidu.com/s/18KzBmOTENjByoWWR17zdiQ?pwd=0000).
