# Engage - Computer Vision for Classroom Engagement

By Keith Spencer-Edgar and Philip Baker

## License

The code of engage is released under the MIT License. There is no limitation for both acadmic and commercial usage.

The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only.

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



