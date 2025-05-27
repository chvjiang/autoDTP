import sys
sys.path.append('.')
import os
import json
import argparse
from tqdm import tqdm
from check import *
from nuscenes.nuscenes import NuScenes

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='root path to NuScenes')
    parser.add_argument('--version', type=str, default='v1.0-mini')
    parser.add_argument('--format', choices=['detection','tracking','pose'], default='detection')
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
    token2status = {a['token']: n for n, a in enumerate(attributes)}
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
        token2index = {instance['token']: i for i,instance in enumerate(instances) if token2category[instance['category_token']] == 'human.pedestrian.adult'}
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
                    if annotation['category_name'] == 'human.pedestrian.adult' and \
                        token2visible[annotation['visibility_token']] not in ["v0-40","v40-60"]:
                        i += 1
                        box = check_instance_in_camera(nusc, cam_data, annotation)
                        if box and (box[2]-box[0])*(box[3]-box[1])>100*100: 
                            n += 1
                            temp_data['index'] = token2index[annotation['instance_token']]
                            temp_data['2d_box'] = box
                    if temp_data:
                        temp_datas.append(temp_data)
                data['annotation'] = temp_datas
                if temp_datas:
                    datas.append(data)
                next_sample_token = sample['next']
        print(f'camera-front/total:\t{n}/{i}')

    elif args.format == 'pose':
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
                        if box and annotation['attribute_tokens']:
                            n += 1
                            temp_data['instance'] = annotation['instance_token']
                            temp_data['2d_box'] = box
                            temp_data['status'] = token2status[annotation['attribute_tokens'][0]]-4
                            temp_data['description'] = token2attribute[annotation['attribute_tokens'][0]]
                    if temp_data:
                        temp_datas.append(temp_data)
                data['annotation'] = temp_datas
                if temp_datas:
                    datas.append(data)
                next_sample_token = sample['next']
        print(f'camera-front/total:\t{n}/{i}')

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
