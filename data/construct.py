import os
import cv2
import json
import argparse
from tqdm import tqdm
from check import *
from PIL import Image
from nuscenes.nuscenes import NuScenes

def construct_yolo_data(image_path, save_path, data_mode, labels):
    assert data_mode in ['train','val','test']
    img = cv2.imread(image_path)
    img_name = image_path.split('/')[-1]
    Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)).save(os.path.join(save_path, 'images', data_mode, img_name))
    with open(os.path.join(save_path, 'labels', data_mode, img_name.replace('.jpg','.txt')),'w') as f:
        for label in labels:
            x_min, y_min, x_max, y_max = label
            image_width, image_height = img.shape[1], img.shape[0]
            x_center = ((x_min + x_max) / 2) / image_width
            y_center = ((y_min + y_max) / 2) / image_height
            bbox_width = (x_max - x_min) / image_width
            bbox_height = (y_max - y_min) / image_height
            assert x_center < 1 or y_center < 1 or bbox_height < 1 or bbox_width < 1
            f.write(f"0 {x_center} {y_center} {bbox_width} {bbox_height}\n")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='root path to NuScenes')
    parser.add_argument('--version', type=str, default='v1.0-mini')
    parser.add_argument('--format', choices=['detection','tracking'], default='detection')
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.path, verbose=True)

    with open(os.path.join(args.path, f'{args.version}/category.json'),'r') as f:
        categories = json.load(f)
    token2category = {c['token']:c['name'] for c in categories}
    with open(os.path.join(args.path, f'{args.version}/visibility.json'),'r') as f:
        visibility = json.load(f)
    token2visible = {v['token']:v['level'] for v in visibility}
    with open(os.path.join(args.path, f'{args.version}/attribute.json'),'r') as f:
        attributes = json.load(f)
    token2attribute = {a['token']:a['name'] for a in attributes}
    with open(os.path.join(args.path, f'{args.version}/instance.json'),'r') as f:
        instances = json.load(f)
    with open(os.path.join(args.path, f'{args.version}/ego_pose.json'),'r') as f:
        ego_poses = json.load(f)
    token2ego_pose = {p['token']:{'translation':p['translation'],'rotation':p['rotation']} for p in ego_poses}
    with open(os.path.join(args.path, f'{args.version}/calibrated_sensor.json'),'r') as f:
        calibrated = json.load(f)
    token2calibrate = {c['token']:{'translation':c['translation'], 'rotation':c['rotation'], 'camera_intrinsic':c['camera_intrinsic']} for c in calibrated}

    datas = []
    if args.format == 'tracking':
        for i in tqdm(instances):
            temp, index = {}, 0
            if token2category[i['category_token']] == 'human.pedestrian.adult':
                temp['category'] = 'human.pedestrian.adult'
                temp['length'] = i['nbr_annotations']
                anno, next_token = [], i['first_annotation_token']
                while next_token != '':
                    data = nusc.get('sample_annotation', next_token)
                    sample = nusc.get('sample', data['sample_token'])
                    anno_cell = {
                        'index': index+1,
                        'visibility': token2visible[data['visibility_token']],
                        'attribute': token2attribute[data['attribute_tokens'][0]],
                        'translation': data['translation'],
                        'size': data['size'],
                        'rotation': data['rotation'],
                        'num_lidar_pts': data['num_lidar_pts'],
                        'num_radar_pts': data['num_radar_pts'],
                    }
                    data_cells = {}
                    for k,v in sample['data'].items():
                        temp_data = nusc.get('sample_data', v)
                        data_cell = {}
                        if k.startswith('CAM'):
                            if check_instance_in_camera(nusc, temp_data, data):
                                data_cell['file_path'] = temp_data['filename']
                                data_cell['ego_pose'] = token2ego_pose[temp_data['ego_pose_token']]
                                data_cell['calibrated_sensor'] = token2calibrate[temp_data['calibrated_sensor_token']]
                        elif k.startswith('RADAR'):
                            if check_instance_in_radar(nusc, temp_data, data):
                                data_cell['file_path'] = temp_data['filename']
                                data_cell['ego_pose'] = token2ego_pose[temp_data['ego_pose_token']]
                                data_cell['calibrated_sensor'] = token2calibrate[temp_data['calibrated_sensor_token']]
                        elif check_instance_in_lidar(nusc, temp_data, data):
                            data_cell['file_path'] = temp_data['filename']
                            data_cell['ego_pose'] = token2ego_pose[temp_data['ego_pose_token']]
                            data_cell['calibrated_sensor'] = token2calibrate[temp_data['calibrated_sensor_token']]
                        data_cells[k] = data_cell
                    anno_cell['data'] = data_cells
                    anno.append(anno_cell)
                    next_token = data['next']
                    index += 1
                temp['annotations'] = anno
                assert index == i['nbr_annotations']
                datas.append(temp)

    elif args.format == 'detection':
        n,i = 0,0
        for scene in tqdm(nusc.scene):
            next_sample_token = scene['first_sample_token']
            while next_sample_token != '':
                sample = nusc.get('sample', next_sample_token)
                cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
                data = {'sample_data': cam_data['token'],'image_path': cam_data['filename']}
                temp_datas = []
                for anns_token in sample['anns']:
                    temp_data = {}
                    annotation = nusc.get('sample_annotation', anns_token)
                    if annotation['category_name'] == 'human.pedestrian.adult':
                        i += 1
                        box = check_instance_in_camera(nusc, cam_data, annotation)
                        if box: 
                            n += 1
                            temp_data['instance'] = annotation['instance_token']
                            temp_data['2d_box'] = box
                    if temp_data:
                        temp_datas.append(temp_data)
                data['annotation'] = temp_datas
                if temp_datas:
                    datas.append(data)
                next_sample_token = sample['next']
        print(f'camera-front/total:\t{n}/{i}')
        
    with open(f'{args.format}.json', 'w') as f:
        json.dump(datas, f, indent=2)
