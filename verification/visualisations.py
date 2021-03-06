"""Documentation for visualisations.py
   This script processes the output from verify_resolutions.py in order to find the number of true positive, false positives etc.
   The script produces two graphs: one for accuracy and one for recall, plotted against face width. 
"""

import csv
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

__author__ = "Philip Baker & Keith Spencer-Edgar"
__date__ = "25-10-2020"

with open("roll_call.json", "r") as read_file:
    json_data = json.load(read_file)

# reaad in csv file to be processed - change file name to suit
csv_data = pd.read_csv("verify_resolutions_csv.csv", names=["UPI", "Present", "Class_name", "Class_photo", "quality", "face_width"])
results = pd.DataFrame(columns=['photo', 'face_width', 'tp_rate', 'tn_rate', 'fp_rate', 'fn_rate', 'precision',
                                'recall', 'specificity', 'ba'])
csv_data.face_width = csv_data.face_width.fillna(0)


for class_p in csv_data.Class_photo.unique():
    # find number of quals
    max_width = max(csv_data.loc[(csv_data.Class_photo == class_p), 'face_width'])
    csv_data['face_width'] = csv_data.apply(lambda row: row.quality * max_width, axis=1)

for photo in csv_data.Class_photo.unique():
    roll = json_data.get(photo)
    for width in csv_data.face_width[csv_data.Class_photo == photo].unique():
        # subset relevant data
        single_photo_df = csv_data[(csv_data.Class_photo == photo) & (csv_data.face_width == width)]
        # calculate accuracy, fp, fn etc
        fp = 0
        fn = 0
        tp = 0
        tn = 0
        for upi in single_photo_df['UPI']:  # for each person enrolled in the class

            if str(upi) in roll:  # if they should be detected

                if single_photo_df[single_photo_df.UPI == upi].Present.tolist()[0] == 1:  # if they are detected
                    tp += 1  # it's a true positive
                else:  # if they aren't detected
                    fn += 1  # it's a false negative

            else:  # if they shouldn't be detected

                if single_photo_df[single_photo_df.UPI == upi].Present.tolist()[0] == 0:  # if they aren't detected
                    tn += 1  # it's a true negative
                else:  # if they are detected
                    fp += 1  # it's a false positive

        total = fp + fn + tp + tn
        fp_rate = fp/total
        fn_rate = fn/total
        tp_rate = tp/total
        tn_rate = tn/total

        precision = tp/(tp + fp + 1e-9)
        recall = tp/(tp + fn + 1e-9)
        specificity = tn/(tn + fp)
        ba = (recall + specificity)/2

        results.loc[len(results)] = [photo, width] + [tp_rate, tn_rate, fp_rate, fn_rate, precision, recall,
                                                      specificity, ba]


for i in results.photo.unique():
    x = results[results.photo == i]['face_width']
    y = results[results.photo == i]['tp_rate'] + results[results.photo == i]['tn_rate']
    plt.plot(x, y, label=i)
plt.xlabel("average face width (pixels)", fontsize=11)
plt.ylabel("accuracy", fontsize=11)
plt.legend()
plt.title("NASA test set, accuracy against average width of face", fontsize=13)
plt.show()


for i in results.photo.unique():
    x = results[results.photo == i]['face_width']
    y = results[results.photo == i]['recall']
    plt.plot(x,y, label = i)
plt.xlabel("average face width (pixels)", fontsize=11)
plt.ylabel("sensitivity (recall)", fontsize=11)
plt.legend()
plt.title('NASA test set, recall against average width of face', fontsize=13)
plt.show()
