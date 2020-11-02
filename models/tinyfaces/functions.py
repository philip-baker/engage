import csv
import numpy as np
from torchvision import transforms
import torch


def regression_refinement(tx, ty, tw, th, cx, cy, cw, ch, indices):
    # refine the bounding boxes
    dcx = cw * tx[indices]
    dcy = ch * ty[indices]

    rcx = cx + dcx
    rcy = cy + dcy

    rcw = cw * np.exp(tw[indices])
    rch = ch * np.exp(th[indices])

    # create bbox array
    rcx = rcx.reshape((rcx.shape[0], 1))
    rcy = rcy.reshape((rcy.shape[0], 1))
    rcw = rcw.reshape((rcw.shape[0], 1))
    rch = rch.reshape((rch.shape[0], 1))

    # transpose so that it is (N, 4)
    bboxes = np.array(
        [rcx - rcw / 2, rcy - rch / 2, rcx + rcw / 2, rcy + rch / 2]).T

    return bboxes


def nms(dets, thresh):
    """
    Courtesy of Ross Girshick
    [https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py]
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep).astype(np.int)


def get_detections(model, img, templates, rf, img_transforms,
                   prob_thresh=0.65, nms_thresh=0.3, scales=(-2, -1, 0, 1), device=None):
    model = model.to(device)
    model.eval()

    dets = np.empty((0, 5))  # store bbox (x1, y1, x2, y2), score

    num_templates = templates.shape[0]

    # Evaluate over multiple scale
    scales_list = [2 ** x for x in scales]

    # convert tensor to PIL image so we can perform resizing
    image = transforms.functional.to_pil_image(img)

    min_side = np.min(image.size)

    for scale in scales_list:
        # scale the images
        scaled_image = transforms.functional.resize(image,
                                                    np.int(min_side * scale))

        # normalize the images
        img = img_transforms(scaled_image)

        # add batch dimension
        img.unsqueeze_(0)

        # now run the model
        x = img.float().to(device)

        output = model(x)

        # first `num_templates` channels are class maps
        score_cls = output[:, :num_templates, :, :]
        prob_cls = torch.sigmoid(score_cls)

        score_cls = score_cls.data.cpu().numpy().transpose((0, 2, 3, 1))
        prob_cls = prob_cls.data.cpu().numpy().transpose((0, 2, 3, 1))

        score_reg = output[:, num_templates:, :, :]
        score_reg = score_reg.data.cpu().numpy().transpose((0, 2, 3, 1))

        t_bboxes, scores = get_bboxes(score_cls, score_reg, prob_cls,
                                      templates, prob_thresh, rf, scale)

        scales = np.ones((t_bboxes.shape[0], 1)) / scale
        # append scores at the end for NMS
        d = np.hstack((t_bboxes, scores))

        dets = np.vstack((dets, d))

    # Apply NMS
    keep = nms(dets, nms_thresh)
    dets = dets[keep]

    return dets


def get_bboxes(score_cls, score_reg, prob_cls, templates, prob_thresh, rf, scale=1, refine=True):
    """
    Convert model output tensor to a set of bounding boxes and their corresponding scores
    """

    num_templates = templates.shape[0]

    # template to evaluate at every scale (Type A templates)
    all_scale_template_ids = np.arange(4, 12)

    # templates to evaluate at a single scale aka small scale (Type B templates)
    one_scale_template_ids = np.arange(18, 25)

    ignored_template_ids = np.setdiff1d(np.arange(25), np.concatenate((all_scale_template_ids,
                                                                       one_scale_template_ids)))

    template_scales = templates[:, 4]

    # if we down-sample, then we only need large templates
    if scale < 1:
        invalid_one_scale_idx = np.where(
            template_scales[one_scale_template_ids] >= 1.0)
    elif scale == 1:
        invalid_one_scale_idx = np.where(
            template_scales[one_scale_template_ids] != 1.0)
    elif scale > 1:
        invalid_one_scale_idx = np.where(
            template_scales[one_scale_template_ids] != 1.0)

    invalid_template_id = np.concatenate((ignored_template_ids,
                                          one_scale_template_ids[invalid_one_scale_idx]))

    # zero out prediction from templates that are invalid on this scale
    prob_cls[:, :, :, invalid_template_id] = 0.0

    indices = np.where(prob_cls > prob_thresh)
    fb, fy, fx, fc = indices

    scores = score_cls[fb, fy, fx, fc]
    scores = scores.reshape((scores.shape[0], 1))

    stride, offset = rf['stride'], rf['offset']
    cy, cx = fy * stride[0] + offset[0], fx * stride[1] + offset[1]
    cw = templates[fc, 2] - templates[fc, 0] + 1
    ch = templates[fc, 3] - templates[fc, 1] + 1

    # bounding box refinements
    tx = score_reg[:, :, :, 0:num_templates]
    ty = score_reg[:, :, :, 1 * num_templates:2 * num_templates]
    tw = score_reg[:, :, :, 2 * num_templates:3 * num_templates]
    th = score_reg[:, :, :, 3 * num_templates:4 * num_templates]

    if refine:
        bboxes = regression_refinement(tx, ty, tw, th,
                                       cx, cy, cw, ch,
                                       indices)

    else:
        bboxes = np.array([cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2])

    # bboxes has a channel dim so we remove that
    bboxes = bboxes[0]

    # scale the bboxes
    factor = 1 / scale
    bboxes = bboxes * factor

    return bboxes, scores


def savescores(dets, saveloc):
    """
          writes scores from Tiny Face detections to csv file

    Parameters:
    ----------
        dets: list of numpy arrays, one array for each detection
            detections (bboxes and scores as [x0, y0, x1, y1, score])
        saveloc: string
            location to save the csv file
    Returns :
    ----------
        NA

    """
    with open(saveloc, 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(dets)):
            writer.writerow([dets[i][4]])
