<p align="center">
  
  <h3 align="center"><strong>GEAL: Generalizable 3D Affordance Learning with Cross-Modal Consistency</strong></h3>

  <p align="center">
      <a href="https://dylanorange.github.io" target='_blank'>Dongyue Lu</a>&nbsp;&nbsp;&nbsp;
      <a href="https://ldkong.com" target='_blank'>Lingdong Kong</a>&nbsp;&nbsp;&nbsp;
      <a href="https://tianxinhuang.github.io/" target='_blank'>Tianxin Huang</a>&nbsp;&nbsp;&nbsp;
      <a href="https://www.comp.nus.edu.sg/~leegh/">Gim Hee Lee</a>&nbsp;&nbsp;&nbsp;
    </br>
  National University of Singapore&nbsp;&nbsp;&nbsp;
  </p>

</p>

<p align="center">
  <a href="https://dylanorange.github.io/files/geal.pdf" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-%F0%9F%93%83-lightblue">
  </a>
  <a href="https://dylanorange.github.io/projects/geal" target='_blank'>
    <img src="https://img.shields.io/badge/Project-%F0%9F%94%97-blue">
  </a>
  <a href="https://huggingface.co/datasets/dylanorange/geal" target="_blank">
    <img src="https://img.shields.io/badge/Dataset-%20Hugging%20Face-yellow">
  </a>
</p>



## 🛠️ About

GEAL is a novel framework designed to enhance the generalization and robustness of 3D affordance learning by leveraging pre-trained 2D models. We employ a dual-branch architecture with Gaussian splatting to map 3D point clouds to 2D representations, enabling realistic renderings. Granularity-adaptive fusion and 2D-3D consistency alignment modules further strengthen cross-modal alignment and knowledge transfer, allowing the 3D branch to benefit from the rich semantics and generalization capacity of 2D models.


<div style="text-align: center;">
    <img src="docus/webpage.gif" alt="GEAL Performance GIF" style="max-width: 100%; height: auto; width: 1000px;">
</div>


## Table of Contents

- [🛠️ About](#️-about)
- [Table of Contents](#table-of-contents)
- [:gear: Installation](#gear-installation)
- [:hotsprings: Data Preparation](#hotsprings-data-preparation)
  - [LASO](#laso)
  - [PIAD](#piad)
- [:rocket: Getting Started](#rocket-getting-started)
  - [Stage 1: 2D Branch Traing](#stage-1-2d-branch-traing)
  - [Stage 2: 3D Branch Traing](#stage-2-3d-branch-traing)
- [:test\_tube: Evaluation](#test_tube-evaluation)
- [:framed\_picture: Visualization](#framed_picture-visualization)
  - [Inference \& Point Cloud Export](#inference--point-cloud-export)
  - [Mitsuba Image Rendering](#mitsuba-image-rendering)
  - [Mitsuba Video Rendering](#mitsuba-video-rendering)
- [:file\_folder: Corrupted Dataset \& Robustness Benchmark](#file_folder-corrupted-dataset--robustness-benchmark)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)


## :gear: Installation

Our code is tested under Python 3.10 and CUDA 11.8.

1️⃣ Create a new environment
```
conda create -n geal python==3.10
conda activate geal
```
2️⃣ Install PyTorch 2.1.0 (CUDA 11.8)

Please refer to the official [PyTorch installation guide](https://pytorch.org/get-started/previous-versions/)
.
Example command:

```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118
```

3️⃣ Install Gaussian Rasterization dependencies

We use a modified version of diff-gaussian-rasterization, which supports:

- feature projection

- Gaussian–pixel correspondence output

Its dependencies include:

**(a) simple-knn**

```
pip install git+https://github.com/dreamgaussian/dreamgaussian.git#subdirectory=simple-knn
```

**(b) kiuikit**

```
pip install git+https://github.com/ashawkey/kiuikit
```

You can also refer to the official instructions in the [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian) repository
.

4️⃣ Build our modified diff-gaussian-rasterization

```
cd thirdparty/diff-gaussian-rasterization
pip install .
cd ../../
```

5️⃣ Install remaining Python dependencies

```
pip install -r requirements.txt
```

## :hotsprings: Data Preparation

### LASO

Please refer to the official [LASO repository](https://github.com/yl3800/LASO) for dataset download instructions.
After downloading, organize the files into a directory (denoted as LASO_root) with the following structure:

```
LASO_root
  ├── Affordance-Question.csv
  ├── anno_test.pkl
  ├── anno_train.pkl
  ├── anno_val.pkl
  ├── objects_test.pkl
  ├── objects_train.pkl
  └── objects_val.pkl         
```

### PIAD

Please refer to the official [PIAD repository](https://github.com/yyvhang/IAGNet) for dataset download.

We apply an additional preprocessing step to make the data format compatible with our pipeline:

```
python dataset/piad_process.py
```
This script will generate four .pkl files corresponding to different training settings:

- `seen_train.pkl`

- `seen_test.pkl`

- `unseen_train.pkl`

- `unseen_test.pkl`

We reuse the text annotations from LASO, so you need to copy `Affordance-Question.csv` from `LASO_root` into your `PIAD_root`.

The final directory structure should look like:
```
PIAD_root
  ├── Affordance-Question.csv
  ├── seen_train.pkl
  ├── seen_test.pkl
  ├── unseen_train.pkl
  └── unseen_test.pkl         
```


## :rocket: Getting Started

### Stage 1: 2D Branch Traing

As described in our paper, the training is divided into two stages.
In Stage 1, we train the 2D branch (Branch2D) to learn the correspondence between visual appearance and affordance semantics.
This stage aims to obtain stable 2D visual representations, which will later serve as initialization for the 3D branch in Stage 2.

All configurations are provided in `config/train_stage1.yaml`

Please pay attention to the following key parameters:

- `category`: dataset type, choose between laso or piad.

- `setting`: experiment setting, either seen or unseen, corresponding to the splits defined in the paper.

- `data_root`: root directory of your dataset.


After updating the YAML file, simply run:

```
python scripts/train_stage1.py --config config/train_stage1.yaml
```

The script will automatically load configurations, initialize the model and optimizer, and start supervised training for the 2D branch. Training logs and model checkpoints will be saved under `runs/train/geal_stage1`.

The trained 2D weights will be used as initialization in Stage 2 (3D Branch Training).


### Stage 2: 3D Branch Traing

In Stage 2, we train the 3D branch (Branch3D) to learn geometry-aware affordance representations by aligning 3D point features with the 2D semantic embeddings obtained from Stage 1.
This stage focuses on transferring visual-affordance knowledge from the 2D branch into the 3D domain for full affordance prediction on point clouds.

All configurations are provided in `config/train_stage2.yaml`.

Make sure the following points are consistent with Stage 1:

- `category` and `setting` should match the dataset and split used in Stage 1.

- Specify the trained 2D weights path under the field `pretrained_2d` in the YAML file to correctly load the 2D branch checkpoint.

Then start Stage 2 training with:
```
python scripts/train_stage2.py --config config/train_stage2.yaml
```
The script will automatically load the pretrained 2D branch, initialize the 3D network, and train it for cross-modal affordance prediction.


## :test_tube: Evaluation

We provide pretrained checkpoints for both datasets (PIAD and LASO) under the seen and unseen settings, as reported in the paper:

- [PIAD (Seen)](https://huggingface.co/datasets/dylanorange/geal/blob/main/piad_seen.pt)

- [PIAD (Unseen)](https://huggingface.co/datasets/dylanorange/geal/blob/main/piad_unseen.pt)

- [LASO (Seen)](https://huggingface.co/datasets/dylanorange/geal/blob/main/laso_seen.pt)

- [LASO (Unseen)](https://huggingface.co/datasets/dylanorange/geal/blob/main/laso_unseen.pt)

Download the pretrained weights and place them in the `ckpt` directory.

All evaluation configurations are provided in config/evaluation.yaml.

Please make sure the following fields are correctly set before running the script:

- `dataset`: choose between piad or laso

- `setting`: choose between seen or unseen

- `ckpt`: path to the pretrained model checkpoint

- `data_root`: path to the dataset root directory

Then simply run:

```
python scripts/evaluation.py --config config/evaluation.yaml
```
The evaluation script will compute per-category, per-affordance, and overall metrics, including IoU, AUC, SIM, and MAE. Results will be automatically saved under `runs/result/`.

## :framed_picture: Visualization

We provide point cloud exporting, visualization, and rendering tools under the `visualization` directory.

### Inference & Point Cloud Export

`export_point_cloud.py` is to export 3D affordance predictions as colored .ply files.

The script reuses the same config as evaluation, loads the pretrained model, computes IoU, and selects the top-N samples per (affordance, class) for export.


```
python visualization/export_point_cloud.py --config config/evaluation.yaml --top_n 10
```

Outputs:
- GT and Pred .ply files under `runs/ply/`
- A summary file `ply_paths.txt` listing all exported paths. 

Each .ply visualizes affordance strength (red = high, gray = low) and can be viewed in Open3D, Meshlab, or Blender for qualitative analysis.


### Mitsuba Image Rendering
The script `render_image.py` provides a full Mitsuba-based rendering pipeline for converting exported .ply point clouds into high-quality rendered images.

It supports four modes:

- 1️⃣ Generate Mitsuba .xml scene files

- 2️⃣ Render them to .exr using Mitsuba

- 3️⃣ Convert .exr to .jpg

- 4️⃣ Or run the full pipeline automatically

```
# Full pipeline
python visualization/render_image.py --mode full \
    --input_txt runs/ply/ply_paths.txt \
    --xml_dir runs/xml_file \
    --exr_dir runs/exr_file \
    --jpg_dir runs/jpg_file
```

Outputs: 
- `.xml` scene files for Mitsuba rendering
- `.exr` high dynamic range renders
- `.jpg` images for visualization


### Mitsuba Video Rendering

```render_video.py```provides an end-to-end pipeline for creating rotating GIFs of 3D affordance point clouds rendered in Mitsuba.
It generates sequential .xml scenes, renders them to .exr, converts to .jpg, and assembles the frames into an animated GIF.

```
python visualization/render_video.py \
    --input runs/ply/ply_paths.txt \
    --out_dir runs/video \
    --frames 200 --radius 3.5 --fps 24
```
Output: 
- Sequential .xml, .exr, and .jpg frames
- A rotating .gif stored in `runs/video/`

Each GIF shows a smooth camera rotation around the predicted affordance visualization, useful for presentation or qualitative analysis.

## :file_folder: Corrupted Dataset & Robustness Benchmark

We introduce a Corrupted 3D Affordance Dataset and the corresponding Robustness Benchmark, designed to evaluate model performance under controlled geometric and structural perturbations.
The dataset is publicly available on [Hugging Face](https://huggingface.co/datasets/dylanorange/geal)
, and its construction follows the framework in [PointCloud-C](https://github.com/ldkong1205/PointCloud-C).

We provide an updated dataloader `dataset/corrupt.py`, and the evaluation script `evaluation_corrupt.py`, which test model robustness across seven corruption types and five severity levels. The script reuses pretrained checkpoints and automatically evaluates all corruptions, reporting averaged IoU, AUC, SIM, and MAE metrics.

```
python scripts/evaluation_corrupt.py --config config/evaluation_corrupt.yaml
```
Output:

- Per-corruption averaged metrics saved as `.txt` in `runs/result/`

- Summary table printed with mean performance across all corruption types

This benchmark measures how well the model generalizes to geometric and structural distortions, following the robustness evaluation protocol described in our paper.

## Citation
If you find this work helpful, please kindly consider citing our paper:
```bibtex
@InProceedings{Lu_2025_CVPR,
    author    = {Lu, Dongyue and Kong, Lingdong and Huang, Tianxin and Lee, Gim Hee},
    title     = {GEAL: Generalizable 3D Affordance Learning with Cross-Modal Consistency},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {1680-1690}
}
```

## Acknowledgements

This work builds upon the generous efforts of the open-source community, especially [LASO](https://github.com/yl3800/LASO), [IAGNet](https://github.com/yyvhang/IAGNet), [OOAL](https://github.com/Reagan1311/OOAL), [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian), and [PointCloud-C](https://github.com/ldkong1205/PointCloud-C).
We are also grateful to our colleagues and collaborators for their encouragement and insightful discussions.
