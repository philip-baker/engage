# Computer Vision for Attendance Management

This module was developed by Keith Spencer-Edgar and Philip Baker as part of our the BE(Hons) Enginering Science programme at the University of Auckland, under the supervision of Dr Nicholas Rattenbury. In this module we present a system using the Tiny Face Detector and ArcFace recognition model to mononitor classroom attendnace. 

## Deployment
### Prerequisites
Install the required packages as a per requirements.txt. 

### Setup 
As part of the project we have developed a system capable of measuring attendance for a single class requiring minimal setup. 
1. Clone the egage repository. 
```
git clone --recursive https://github.com/philip-baker/engage/edit/main/README.md
```

2. Download the pre-trained weights for the Tiny Face Detector from [here](https://drive.google.com/file/d/1V8c8xkMrQaCnd3MVChvJ2Ge-DUfXPHNu/view), and store checkpoint_50.pth under *`$engage/models/tinyfaces`*.


3. Download the LResNet50E-IR, ArcFace@ms1m-refine-v1 model from [here](https://github.com/deepinsight/insightface/wiki/Model-Zoo) and store the model in the *`$engage/models`* directory directory. 

### Usage
1. For every student in the class enter their UPI (Unique Personal Identifier / Student ID), last name, first name, into the file students.csv this file should not have a header row.

2. Save a profile picture for each student in the class in the directory student_profiles. The profile pictures should be .jpg files and the file name should be the sudents UPI. 

3. Run build_system.py this will create an SQLite (.db) database storing the information entered in the file students.csv this database will then be used to store attendance records.

4. If you would like to take a picture of the class using an attached camera simply run check_attendance.py at the beginning of the class. A pop up window will appear, when you are satisfied with the picture, press q to take the image. If you would like to manually provide an image, simply put an image named sample.jpg under sample_images and run check_attendance.py

5. To view the results you will need to use third-party software to view the contents of the .db SQLite database.  To do this the authors have been using [DB Browser for SQLite](https://sqlitebrowser.org/)

## Theory
### Face Detection
Face detection was done using the Tiny Face Detector (https://arxiv.org/pdf/1612.04402.pdf). A Pytorch impelementation of the model, as well as the model weights are available at https://github.com/varunagrawal/tiny-faces-pytorch.

### Recongition
For facial recognition we used ArcFace, developed by the Insight Face team. More information about the model can be found in [arXiv technical report](https://arxiv.org/abs/1801.07698). The authors of this model were able to achieve LFW 99.83%+ and Megaface 98%+. Further information about ArcFace is available at the origional authors' github repository [InsightFace](https://github.com/deepinsight/insightface/blob/master/README.md). The models provided earlier are from the Insight Face teams's [Model Zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo). 

## License

The code of engage is released under the MIT License. There is no limitation for both acadmic and commercial usage.

The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only.

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

[Philip Baker](mailto:philipbaker@hotmail.co.nz?subject=[GitHub]%20Engage%20Project)

Keith Spencer-Edgar

