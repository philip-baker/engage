# Engage - Computer Vision for Classroom Engagement

By Jia Guo and [Jiankang Deng](https://jiankangdeng.github.io/)

## License

The code of InsightFace is released under the MIT License. There is no limitation for both acadmic and commercial usage.

The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only.

## Introduction

InsightFace is an open source 2D&3D deep face analysis toolbox, mainly based on MXNet. 

The master branch works with **MXNet 1.2 to 1.6**, with **Python 3.x**.


 ## ArcFace Video Demo

[![ArcFace Demo](https://github.com/deepinsight/insightface/blob/master/resources/facerecognitionfromvideo.PNG)](https://www.youtube.com/watch?v=y-D1tReryGA&t=81s)

Please click the image to watch the Youtube video. For Bilibili users, click [here](https://www.bilibili.com/video/av38041494?from=search&seid=11501833604850032313).

## Recent Update

**`2020-10-13`**: A new training method and one large training set(360K IDs) were released [here](https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc) by DeepGlint.

**`2020-10-09`**: We opened a large scale recognition test benchmark [IFRT](https://github.com/deepinsight/insightface/tree/master/IFRT)

**`2020-08-01`**: We released lightweight facial landmark models with fast coordinate regression(106 points). See detail [here](https://github.com/deepinsight/insightface/tree/master/alignment/coordinateReg).

**`2020-04-27`**: InsightFace pretrained models and MS1M-Arcface are now specified as the only external training dataset, for iQIYI iCartoonFace challenge, see detail [here](http://challenge.ai.iqiyi.com/detail?raceId=5def71b4e9fcf68aef76a75e).

**`2020.02.21`**: Instant discussion group created on QQ with group-id: 711302608. For English developers, see install tutorial [here](https://github.com/deepinsight/insightface/issues/1069).

**`2020.02.16`**: RetinaFace now can detect faces with mask, for anti-CoVID19, see detail [here](https://github.com/deepinsight/insightface/tree/master/RetinaFaceAntiCov)

**`2019.08.10`**: We achieved 2nd place at [WIDER Face Detection Challenge 2019](http://wider-challenge.org/2019.html).

**`2019.05.30`**: [Presentation at cvmart](https://pan.baidu.com/s/1v9fFHBJ8Q9Kl9Z6GwhbY6A)

**`2019.04.30`**: Our Face detector ([RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace)) obtains state-of-the-art results on [the WiderFace dataset](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html).

**`2019.04.14`**: We will launch a [Light-weight Face Recognition challenge/workshop](https://github.com/deepinsight/insightface/tree/master/iccv19-challenge) on ICCV 2019.

**`2019.04.04`**: Arcface achieved state-of-the-art performance (7/109) on the NIST Face Recognition Vendor Test (FRVT) (1:1 verification)
[report](https://www.nist.gov/sites/default/files/documents/2019/04/04/frvt_report_2019_04_04.pdf) (name: Imperial-000 and Imperial-001). Our solution is based on [MS1MV2+DeepGlintAsian, ResNet100, ArcFace loss]. 

**`2019.02.08`**: Please check [https://github.com/deepinsight/insightface/tree/master/recognition/ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/ArcFace) for our parallel training code which can easily and efficiently support one million identities on a single machine (8* 1080ti).

**`2018.12.13`**: Inference acceleration [TVM-Benchmark](https://github.com/deepinsight/insightface/wiki/TVM-Benchmark).

**`2018.10.28`**: Light-weight attribute model [Gender-Age](https://github.com/deepinsight/insightface/tree/master/gender-age). About 1MB, 10ms on single CPU core. Gender accuracy 96% on validation set and 4.1 age MAE.

**`2018.10.16`**: We achieved state-of-the-art performance on [Trillionpairs](http://trillionpairs.deepglint.com/results) (name: nttstar) and [IQIYI_VID](http://challenge.ai.iqiyi.com/detail?raceId=5afc36639689443e8f815f9e) (name: WitcheR). 


## Contents
[Deep Face Recognition](#deep-face-recognition)
- [Introduction](#introduction)
- [Training Data](#training-data)
- [Train](#train)
- [Pretrained Models](#pretrained-models)
- [Verification Results On Combined Margin](#verification-results-on-combined-margin)
- [Test on MegaFace](#test-on-megaface)
- [512-D Feature Embedding](#512-d-feature-embedding)
- [Third-party Re-implementation](#third-party-re-implementation)

[Face Detection](#face-detection)
- [RetinaFace](#retinaface)
- [RetinaFaceAntiCov](#retinafaceanticov)

[Face Alignment](#face-alignment)
- [DenseUNet](#denseunet)
- [CoordinateReg](#coordinatereg)


[Citation](#citation)

[Contact](#contact)

## Deep Face Recognition

### Introduction

In this module, network settings and loss designs for deep face recognition.



## Face Detection


## Recongition

For facial recognition we used ArcFace, which developed by the Insight Face and was initially described in an [arXiv technical report](https://arxiv.org/abs/1801.07698). The authors of this model were able to achieve LFW 99.83%+ and Megaface 98%+. Further information about ArcFace is available at the origional authors' github repository [InsightFace](https://github.com/deepinsight/insightface/blob/master/README.md)

For models please see the Insight Face teams's 
[Model Zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo), we reccomend using the LResNet50E-IR,ArcFace@ms1m-refine-v1 model.  This model will have to be downloaded, and then stored in the models folder. 


## Key Steps 
1. Prepare a pre-trained model
2. Put the model under *`$engage/models/`*. For example, *`$engage/models/model-r100-ii`*.



## Citation

If you find the ArcFace facial recongition method useful please consider citing the origional authors work.


```
@inproceedings{deng2018arcface,
title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
booktitle={CVPR},
year={2019}
}
```

## Contact

```
[Keith Spencer-Edgar]
[Philip Baker](philipbaker[at]hotmail.co.nz)



