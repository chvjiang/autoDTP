import json
import matplotlib.pyplot as plt

def smooth_ema(data, weight=0.6):
    smoothed = []
    last = data[0]
    for point in data:
        last = weight * last + (1 - weight) * point
        smoothed.append(last)
    return smoothed

def draw(file_path):
    data_list, epochs, loss_bbox, loss_cls, loss_iou, mAP_50, mAP_50_95 = [], [], [], [], [], [], []
    with open(file_path,'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data_list.append(json.loads(line))
    for data in data_list:
        epochs.append(data['epoch']+1)
        loss_bbox.append(data['train_loss_bbox'])
        loss_iou.append(data['train_loss_giou'])
        loss_cls.append(data['train_loss_vfl'])
        mAP_50.append(data['test_coco_eval_bbox'][1])
        mAP_50_95.append(data['test_coco_eval_bbox'][0])

    titles = ['train/box_loss','train/cls_loss','train/iou_loss','mAP50(B)','mAP50-95(B)']
    plots = [loss_bbox, loss_cls, loss_iou, mAP_50, mAP_50_95]

    plt.figure(figsize=(17, 4))
    for i,(title,temp) in enumerate(zip(titles,plots)):
        plt.subplot(1,len(titles),i+1)
        plt.plot(epochs, temp, 'o-', markersize=5, linewidth=2, label='results')
        plt.plot(epochs[1:], smooth_ema(temp[1:]), ':', linewidth=2, color='orange', label='smooth')
        if i == 1:
            plt.legend()
        plt.title(title)
    plt.savefig('loss_curves.png',dpi=300,bbox_inches='tight') 

if __name__=='__main__':

    log_file = 'output/rtdetrv2_trained/log.txt' 
    draw(log_file)    