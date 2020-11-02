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

For facial recognition we used ArcFace, eveloped by the Insight Face team. More information about the mdoel can be found in [arXiv technical report](https://arxiv.org/abs/1801.07698). The authors of this model were able to achieve LFW 99.83%+ and Megaface 98%+. Further information about ArcFace is available at the origional authors' github repository [InsightFace](https://github.com/deepinsight/insightface/blob/master/README.md)

For models please see the Insight Face teams's 
[Model Zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo), we reccomend using the LResNet50E-IR,ArcFace@ms1m-refine-v1 model.  This model will have to be downloaded, and then stored in the models folder. 


## OBA - Out of the Box Attendance
### Setup 
As part of the project we have developed a system capable of measuring attendance for a single class requiring minimal setup. 
1. Clone the egage repository.
```
git clone --recursive https://github.com/philip-baker/engage/edit/main/README.md
```
2. Download the Tiny Face Detector and ArcFace models, then put both models in the *`$engagemodels`* directory. If you are using the model-r100-ii, then you will need to put hte model-r100-ii directory directly under models. You will need to put checkpoint_50.pth (the Tiny Face Detector model) under *`$engage/models/tinyfaces`*.
3. Download the required packages as a per requirements.txt. You can try pip3 install -r requirements.txt
### Usage
1. In the file students.csv enter the UPI (Unique Personal Identifier), last name, first name, and age for every student in the class. Please not that this file does not have a reader row at the beginning. 

2. In the directory student_profiles put a profile picture for each student in the class. The profile pictures should be .jpg files and the file name should be the sudents UPI. Run build_system.py to create the SQLite (.db) database file.

3. If you would like to take a picture of the class using an attached camera simply run check_attendance.py at the beginning of the class. A pop up window will appear, when you are satisfied with the picture, press q to take the image. If you would like to manually provide an image, simply put an image named sample.jpg under sample_images and run check_attendance.py

4. To view the results you will need to use third-party software to view the contents of the .db SQLite database.  To do this the authors have been using DB Browser for SQLite. 



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



