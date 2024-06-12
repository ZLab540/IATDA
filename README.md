# Illumination Guided Unsupervised Domain Adaptation Object Detection in Thermal Imagery

## Introduction

Source code for our submission "Illumination Guided Unsupervised Domain Adaptation Object Detection in Thermal Imagery"


### Prerequisites

* Python 2.7 or 3.6
* CUDA 8.0 or higher


### Our Trained Model

* For KAIST: [Pan Baidu](https://pan.baidu.com/s/17EfQTojRNjeW3i9-e-517g) 提取码: njem
* For FLIR: [Pan Baidu](https://pan.baidu.com/s/1D4on4OaxuHlucDab_wO4zA) 提取码: 3drt

This model should be put into ./trained_model

### Compilation

Compile the cuda depended modules, e.g. NMS, ROI Pooling, ROI Align and ROI Crop, using following simple commands:

```
cd lib
sh make.sh
```

## Test

Test KAIST:
```
CUDA_VISIBLE_CEVICES=0 python test_net.py --dataset kaist_ir --net res101 --load_dir ./tarined_model --checksession 79 --checkepoch 7 --checkpoint 15141 --cuda --pos_r 0.25

```

Test FLIR:
```
CUDA_VISIBLE_CEVICES=0 python test_net.py --dataset flir_ir --net res101 --load_dir ./tarined_model --checksession 82 --checkepoch 9 --checkpoint 8253 --cuda --pos_r 0.25

```

