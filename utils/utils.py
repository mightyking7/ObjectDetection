import torch
import math
import numpy as np

def build_targets(pred_boxes, pred_conf, pred_cls, target, anchors, n_anchors, n_classes, grid_size, ignore_thres, img_dim):

    nB = target.size(0)
    nA = n_anchors
    nC = n_classes
    nG = grid_size
    mask = torch.zeros(nB, nA, nG, nG)
    conf_mask = torch.ones(nB, nA, nG, nG)
    tx = torch.zeros(nB, nA, nG, nG)
    ty = torch.zeros(nB, nA, nG, nG)
    tw = torch.zeros(nB, nA, nG, nG)
    th = torch.zeros(nB, nA, nG, nG)
    tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
    tcls = torch.ByteTensor(nB, nA, nG, nG).fill_(0)

    nGT = 0
    nCorrect = 0

    for b in range(nB):
        for t in range(target.shape[1]):

            if target[b, t].sum() == 0:
                continue

            nGT += 1

            # convert position relative to box
            gx = target[b, t, 1] * nG
            gy = target[b, t, 2] * nG
            gw = target[b, t, 3] * nG
            gh = target[b, t, 4] * nG

            # get grid box indices
            gi = int(gx)
            gj = int(gy)

            # get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)

            # get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))

            # calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes)

            # set mask to zero where overlap is larger than threshold
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0

            # find best matching anchor box
            best_n = np.argmax(anch_ious)

            # get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)

            # get best prediction
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)

            # masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1

            # coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj

            # width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)

            # one hot encoding of label
            target_label = int(target[b, t, 0])
            tcls[b, best_n, gj, gi, target_label] = 1
            tconf[b, best_n, gj, gi] = 1

            # calculate iou between ground truth and best matching prediction
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            score = pred_conf[b, best_n, gj, gi]

            if iou > 0.5 and pred_label == target_label and score > 0.5:
                nCorrect += 1

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls
