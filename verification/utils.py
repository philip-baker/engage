# contains various functions needed for our combined system
# Authors: Keith Spencer-Edgar & Philip Baker, The University of Auckland 2020
import sklearn

def process_image(img, class_folder, model):
    """
    Processes a single image
    Parameters:
        img: the image of the lecture as a PILimage
        class_folder: path to the folder containing call face ID photos

    Returns:
         nested list containing scores for each person
    """

    # run the image through tinyfaces
    dets = get_detections(model, img_tensor, templates, rf, val_transforms,
                          prob_thresh=args.prob_thresh, nms_thresh=args.nms_thresh, device=device)
    # fn = os.path.basename(os.path.normpath(img.split(".jpg")[0]))
    # os.mkdir(args.results_dir + fn)

    for i in range(len(dets)):
        score = dets[i][4]
        img.crop(dets[i][0:4])


    # run the output through arcface

    # return results