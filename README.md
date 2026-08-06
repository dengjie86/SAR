# 🌠 Towards Stable Test-Time Adaptation in Dynamic Wild World

This is the official project repository for [Towards Stable Test-Time Adaptation in Dynamic Wild World 🔗](https://openreview.net/pdf?id=g2YraF75Tj) by
Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Zhiquan Wen, Yaofo Chen, Peilin Zhao and Mingkui Tan **(ICLR 2023 Oral, Notable-Top-5%)**.

- 1️⃣ SAR conducts model learning at test time to adapt a pre-trained model to test data that has distributional shifts ☀️ 🌧 ❄️, such as corruptions, simulation-to-real discrepancies, and other differences between training and testing data. 
- 2️⃣ SAR aims to adapt a model in dymamic wild world, i.e., the test data stream may have mixed domain shifts, small batch size, and online imbalanced label distribution shifts (as shown in the figure below).

<p align="center">
<img src="figures/wild_settings.png" alt="wild_settings" width="100%" align=center />
</p>

Method: **S**harpness-**A**ware and **R**eliable Entropy Minimization (SAR)
- 1️⃣ SAR conducts selective entropy minimization by excluding partial samples with noisy gradients out of online adaptation.

- 2️⃣ SAR optimizes both entropy and the sharpness of entropy surface simutaneously, so that the model update is robust to those remaining samples with noisy gradients.


**Installation**:

SAR depends on

- Python 3
- [PyTorch](https://pytorch.org/) = 1.9.0
- [timm](https://github.com/huggingface/pytorch-image-models)==1.0.28 for current Kaggle/Python runtimes

The original environment used `timm==0.6.11`, but that release raises a `MaxxVitConvCfg` mutable-default error on current Kaggle Python. In a Kaggle notebook install the compatible version with:

```
!pip install --upgrade timm==1.0.28
```

The model identifiers are pinned in `main.py` to the same ResNet50-GN and ViT-Base pretrained weight variants used by the original code, so upgrading `timm` does not silently select a different default checkpoint.


**Data preparation**:

This repository contains code for evaluation on [ImageNet-C 🔗](https://arxiv.org/abs/1903.12261) with ResNet-50 and VitBase. It also accepts 64x64 ImageNet-C mirrors by automatically applying the standard ImageNet resize-and-crop preprocessing. But feel free to use your own data and models!

- Step 1: Download [ImageNet-C 🔗](https://github.com/hendrycks/robustness) dataset from [here 🔗](https://zenodo.org/record/2235448#.YpCSLxNBxAc). 

- Step 2: Put IamgeNet-C at "--data_corruption".

- Step 3 [optional, for EATA]: Put ImageNet **test/val set**  at  "--data".

The expected corruption layout is unchanged for both standard and 64x64 data:

```
/path/to/imagenet-c/
├── gaussian_noise/5/<class>/*.JPEG
├── shot_noise/5/<class>/*.JPEG
└── ...
```

With the 64x64 Kaggle mirror, `--data_corruption` should point to the directory that directly contains `gaussian_noise`, `shot_noise`, and the other corruption folders. The default `--corruption_resize auto` detects a small sample and upsamples it before the 224x224 center crop. Use `--corruption_resize never` only to reproduce the repository's former zero-padding behavior.



**Usage**:

```
import sar
from sam import SAM

model = TODO_model()

model = sar.configure_model(model)
params, param_names = sar.collect_params(model)
base_optimizer = torch.optim.SGD
optimizer = SAM(params, base_optimizer, lr=args.lr, momentum=0.9)
adapt_model = sar.SAR(net, optimizer, margin_e0=0.4*math.log(1000))

outputs = adapt_model(inputs)  # now it infers and adapts!
```


## Example: Adapting a pre-trained model on ImageNet-C (Corruption).

**Usage**:

```
python3 main.py --data_corruption /path/to/imagenet-c --exp_type [normal/bs1/mix_shifts/label_shifts] --method [no_adapt/tent/eata/sar] --model [resnet50_gn_timm/vitbase_timm] --output /output/dir
```

For example, a Kaggle smoke test on only Gaussian noise can use:

```
python3 main.py \
  --data_corruption /kaggle/input/imagenet-c/ImageNet-C \
  --exp_type normal \
  --method sar \
  --model resnet50_gn_timm \
  --test_batch_size 16 \
  --corruptions gaussian_noise \
  --output /kaggle/working/sar-results
```

Omit `--corruptions gaussian_noise` to evaluate all 15 corruption types. Accuracy measured on a 64x64 mirror is useful for comparing methods under the same preprocessing, but it is not directly comparable with the paper's standard-resolution ImageNet-C numbers.

'--exp_type' is choosen from:

- 'normal' means the same test setting to prior mild data stream in Tent and EATA

- 'bs1' means single sample adaptation, only one sample comes each time-step

- 'mix_shifts' conducts exps over the mixture of 15 corruption types in ImageNet-C

- 'label_shifts' means exps under online imbalanced label distribution shifts. Moreover, imbalance_ratio indicates the imbalance extent

Note: For EATA method, you need also to set "--data /path/to/imagenet" of clean ImageNet test/validation set to compute the weight importance for regularization.

**Experimental results**:

The Table below shows the results **under online imbalanced label distribution shifts**. The reported **average accuracy** is averaged over 15 different corruption types in ImageNet-C (severity level 5).
|            | ResNet-50 (BN) | ResNet-50 (GN)          | VitBase (LN)            |
| :---------- | :--------------: | :-----------------------: | :-----------------------: |
| No adapt   | 18.0           | 30.6                    | 29.9                    |
| MEMO       | 24.0           | 31.3                    | 39.1                    |
| DDA        | 27.2           | 35.1                    | 36.2                    |
| Tent       | 2.1            | 22.0                    | 47.3                    |
| EATA       | 0.9            | 31.6                    | 49.9                    |
| SAR (ours) | --            | **37.2 $\pm$ 0.6** | **58.0 $\pm$ 0.5** |

Please see our [PAPER 🔗](https://openreview.net/pdf?id=g2YraF75Tj) for more detailed results.



## Correspondence 

Please contact Shuaicheng Niu by [niushuaicheng at gmail.com] if you have any questions.  📬


## Citation
If our SAR method or wild test-time adaptation settings are helpful in your research, please consider citing our paper:
```
@inproceedings{niu2023towards,
  title={Towards Stable Test-Time Adaptation in Dynamic Wild World},
  author={Niu, Shuaicheng and Wu, Jiaxiang and Zhang, Yifan and Wen, Zhiquan and Chen, Yaofo and Zhao, Peilin and Tan, Mingkui},
  booktitle = {Internetional Conference on Learning Representations},
  year = {2023}
}
```

## Acknowledgment
The code is inspired by the [Tent 🔗](https://github.com/DequanWang/tent) and [EATA 🔗](https://github.com/mr-eggplant/EATA).
