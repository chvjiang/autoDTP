from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import argparse

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_json', type=str, default='DATA/val_coco.json')
    parser.add_argument('--pr_json', type=str, default='predictions.json')
    args = parser.parse_args()

    annFile = args.gt_json
    resFile = args.pr_json

    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(resFile)
    
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    precision = cocoEval.eval['precision']  # shape [T_IoU, Recall, Kinds, Size, Max_detection]
    recall = cocoEval.eval['recall']        # shape [T, K, S, M]

    print(' Average Recall'.ljust(19), '(AR)', '@[ IoU=0.50'.ljust(16), '|', 'area=   all', '|', 'maxDets=100 ] =', '%.3f'%np.mean(recall[0, 0, :, -1]))
    print(' Average Recall'.ljust(19), '(AR)', '@[ IoU=0.75'.ljust(16), '|', 'area=   all', '|', 'maxDets=100 ] =', '%.3f'%np.mean(recall[5, 0, :, -1]))

    # print(' P'.ljust(9), '|', np.mean(precision[0, 60, 0, :, -1]))

    