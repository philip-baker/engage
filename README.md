# Computer Vision for Attendance Management

By Keith Spencer-Edgar and Philip Baker 

## License

The code of engage is released under the MIT License. There is no limitation for both acadmic and commercial usage.

The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only.

### Introduction

This module was developed as part of our BE(Hons) Enginering Science - Part IV Project undertaken as part of our undergraduate study at the University of Auckland under the supervision of Dr Nicholas Rattenbury. In this module we present a system using the Tiny Face Detector and ArcFace recognition model to mononitor classroom attendnace. 

## Face Detection

[here](https://drive.google.com/file/d/1V8c8xkMrQaCnd3MVChvJ2Ge-DUfXPHNu/view)

## Recongition

For facial recognition we used ArcFace, which developed by the Insight Face and was initially described in an [arXiv technical report](https://arxiv.org/abs/1801.07698). The authors of this model were able to achieve LFW 99.83%+ and Megaface 98%+. Further information about ArcFace is available at the origional authors' github repository [InsightFace](https://github.com/deepinsight/insightface/blob/master/README.md)

For models please see the Insight Face teams's 
[Model Zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo), we reccomend using the LResNet50E-IR,ArcFace@ms1m-refine-v1 model.  This model will have to be downloaded, and then stored in the models folder. 


## Key Steps 
1. Prepare a pre-trained model
2. Put the model under *`$engage/models/`*. For example, *`$engage/models/model-r100-ii`*.


## Citation

If you find either ArcFace or the Tiny Face Detector useful please consider citing the original works.

```
@inproceedings{deng2018arcface,
title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
booktitle={CVPR},
year={2019}
}

@inproceedings{tinyfaces,
author = {Hu, Peiyun and Ramanan, Deva},
year = {2017},
month = {07},
pages = {1522-1530},
title = {Finding Tiny Faces},
doi = {10.1109/CVPR.2017.166}
}
```

## Contact

```
[Keith Spencer-Edgar]
[Philip Baker](philipbaker@hotmail.co.nz)



