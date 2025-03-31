PGD
==== 
1.Introduction  
------- 
This project is for paper "Physics-Guided Detector for SAR Airplanes"

### 1.1 Features
![本地路径](data/images/fig_pgdet.png)

The detailed implementation of the proposed physics-guided detector for SAR airplanes.

![本地路径](data/images/fig_pgdlite_exp.png)

The explanation results based on Grad-CAM for PGD and PGD-Lite models.

![本地路径](data/images/fig_IPexp.png)



### 1.2 Contribution
* A general physics-guided detector (PGD) learning paradigm is proposed for SAR airplane detection and fine-grained classification, aiming to address the challenges of discreteness and variability.
* PGD is consist of physics-guided self-supervised learning (PGSSL), feature enhancement (PGFE) and instance perception (PGIP).
* Extensive experiments are conducted on SAR-AIRcraft-1.0 dataset. Based on existing detectors, we construct nine PGD models for evaluation.

2.Previously on PGD
------- 
In our precious works, we propose a physics guided learning method for SAR airplane target feature rep resentation, where the airplane scattering characteristics are extracted to guide the model training.

```
@inproceedings{wang2023new,
  title={A new Perspective on Physics Guided Learning for SAR Image Interpretation},
  author={Wang, Zishi and Huang, Zhongling and Datcu, Mihai},
  booktitle={IGARSS 2023-2023 IEEE International Geoscience and Remote Sensing Symposium},
  pages={1926--1929},
  year={2023},
  organization={IEEE}
}
```

3.Getting Started
------- 
### 3.1 Requirements
Code is based on an  object detection YOLOv5. Please refer to [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) for installation and dataset preparation.

### 3.2 Preparation process
The code are proposed here. We will complete the pivotal code after the paper is accepted.

The file directory tree is as below:
```
├── data
│   ├── SAR_AIRcraft
│   │   ├── test
|   |   |   |——images
|   |   |   |——labels
│   │   ├── train
|   |   |   |——images
|   |   |   |——labels
│   │   ├── val
|   |   |   |——images
|   |   |   |——labels
├──models
│   ├── common.py
│   ├── PGD_yolo.py
│   ├── ...
├──tools
│   ├── main_cnn.py
│   ├── utils.py
│   ├── ...
```

### 3.3 Data Preparation
SAR-AIRcraft-1.0:[https://radars.ac.cn/cn/article/doi/10.12000/JR23043](https://radars.ac.cn/cn/article/doi/10.12000/JR23043)

### 3.4 Train for PGD
```python
python train_PGD.py --data data/SAR_PLANE.yaml  --cfg models/hub/resnet18_RepPAN.yaml  --hyp data/hyps/hyp_SARPLANE.yaml 
```

### 3.5 Train for PGD_Lite
```python
python train_PGD_Lite.py --data data/SAR_PLANE.yaml  --cfg models/hub/resnet18_RepPAN.yaml  --hyp data/hyps/hyp_SARPLANE.yaml 
```

### 3.6 Pre-training weights
PGD:download(https://pan.baidu.com/s/13OMggVOiwatqLR4hWIMMvA?pwd=zjs9)
PGD_Lite:download(https://pan.baidu.com/s/1Hmt5lFrfHbJRfddJWL2dUg?pwd=ithq)

4.Contributions
------- 
Based on the preliminary findings in our previous work, we released this PGD project as a continuous study with the following extensions:
* A general physics-guided detector (PGD) learning paradigm is proposed for SAR airplane detection and fine-grained classification, aiming to address the challenges of discreteness and variability. It can be extended to different deep learning-based detectors and applicable for various types of backbone models and detection heads. Several implementations, denoted as PGD and PGD-Lite, are provided to illustrate its flexibility and effectiveness.
* PGD is consist of physics-guided self-supervised learning (PGSSL), feature enhancement (PGFE) and instance perception (PGIP). PGSSL and PGFE aim to learn from airplane scattering structure distributions and further facilitate more discriminative feature representations. PGIP is designed to concern the refined scattering characteristics of each instance adaptively at the detection head. The discrete and variable scattering information of SAR airplanes are comprehensively investigated.
* Extensive experiments are conducted on SAR-AIRcraft-1.0 dataset. Based on existing detectors, we construct nine PGD models for evaluation. The results show that the proposed PGD can achieve the state-of-the-art performance on SAR-AIRcraft-1.0 dataset (90.7\% mAP) compared with other methods, and improve the existing detectors by 3.1\% mAP most. The effectiveness also proves the model-agnostic ability of PGD learning paradigm, showing significant potentials of its utility.

5.Citation
------- 
If you find this repository useful for your publications, please consider citing our paper.
