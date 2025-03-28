import json
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot='DATA/', verbose=True)

with open('DATA/v1.0-mini/category.json','r') as f:
    categories = json.load(f)
token2category = {c['token']:c['name'] for c in categories}
with open('DATA/v1.0-mini/visibility.json','r') as f:
    visibility = json.load(f)
token2visible = {v['token']:v['level'] for v in visibility}
with open('DATA/v1.0-mini/attribute.json','r') as f:
    attributes = json.load(f)
token2attribute = {a['token']:a['name'] for a in attributes}
with open('DATA/v1.0-mini/instance.json','r') as f:
    instances = json.load(f)
with open('DATA/v1.0-mini/ego_pose.json','r') as f:
    ego_poses = json.load(f)
token2ego_pose = {p['token']:{'translation':p['translation'],'rotation':p['rotation']} for p in ego_poses}
with open('DATA/v1.0-mini/calibrated_sensor.json','r') as f:
    calibrated = json.load(f)
token2calibrate = {c['token']:{'translation':c['translation'], 'rotation':c['rotation'], 'camera_intrinsic':c['camera_intrinsic']} for c in calibrated}

p = []
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
            for k in sample['data'].keys():
                data_cell = {}
                temp_data = nusc.get('sample_data',sample['data'][k])
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
        p.append(temp)
with open('pedestrian_data.json','w') as f:
    json.dump(p, f, indent=2)