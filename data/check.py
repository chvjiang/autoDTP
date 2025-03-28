from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box, LidarPointCloud, RadarPointCloud
from pyquaternion import Quaternion
import numpy as np
import json 
import os
import argparse

def check_point_in_box(box: Box, point):
    center = np.array(box.center)
    rotation = box.orientation.rotation_matrix
    local_point = rotation.T @ (point - center)
    size = np.array(box.wlh) / 2
    return np.all(np.abs(local_point) <= size)

def check_point_in_camera(nusc, cam_data, annotation):
    calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    ego_pose = nusc.get('ego_pose',cam_data['ego_pose_token'])
    box = nusc.get_box(annotation['token'])
    corners_3d = box.corners()
    corners_3d -= np.array(ego_pose['translation']).reshape(3, 1)
    corners_3d = Quaternion(ego_pose['rotation']).inverse.rotation_matrix @ corners_3d
    corners_3d -= np.array(calib['translation']).reshape(3, 1)
    corners_3d = Quaternion(calib['rotation']).inverse.rotation_matrix @ corners_3d
    corners_img = view_points(corners_3d, view=np.array(calib['camera_intrinsic']), normalize=True)
    try:
        in_front = corners_3d[2, :] > 0.1
        in_image = (corners_img[0, :] > 0) & (corners_img[0, :] < 1600) & (corners_img[1, :] > 0) & (corners_img[1, :] < 900)
        x_min = max(0, int(np.min(corners_img[0, corners_3d[2, :] > 0.1])))
        y_min = max(0, int(np.min(corners_img[1, corners_3d[2, :] > 0.1])))
        x_max = min(1600, int(np.max(corners_img[0, corners_3d[2, :] > 0.1])))
        y_max = min(900, int(np.max(corners_img[1, corners_3d[2, :] > 0.1])))
        if any(in_front & in_image) and x_min < x_max and y_min < y_max:
            return True
        else:
            return False
    except:
        return False

def check_point_in_lidar(lidar_path, annotation):
    pc = LidarPointCloud.from_file(lidar_path)
    try:
        points = pc.points.T[:,:3]
        box = Box(
            annotation['translation'], annotation['size'], Quaternion(annotation['rotation']),
            name=annotation['category_name'], token=annotation['token']
        )
        inside = [check_point_in_box(box, point) for point in points]
        return np.any(inside)
    except:
        return False

def check_point_in_radar(radar_path, annotation):
    pc = RadarPointCloud.from_file(radar_path)
    try:
        points = pc.points.T[:,:3]
        box = Box(
            annotation['translation'], annotation['size'], Quaternion(annotation['rotation']),
            name=annotation['category_name'], token=annotation['token']
        )
        inside = [check_point_in_box(box, point) for point in points]
        return np.any(inside)
    except:
        return False

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='root path to NuScenes')
    parser.add_argument('--version', type=str, default='v1.0-mini')
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.path, verbose=True)
    with open(os.path.join(args.path, f'{args.version}/instance.json'),'r') as f:
        instances = json.load(f)
    with open(os.path.join(args.path, f'{args.version}/category.json'),'r') as f:
        categories = json.load(f)

    token2category = {c['token']:c['name'] for c in categories}
    camera_lost, radar, lidar, front, back, total = 0, 0, 0, 0, 0, 0
    sensor_channels = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
                       'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'LIDAR_TOP']

    for instance in instances:
        if token2category[instance['category_token']] == 'human.pedestrian.adult':
            total += 1
            f, b, r = 0, 0, 0
            annotation = nusc.get('sample_annotation',instance['first_annotation_token'])
            sample = nusc.get('sample',annotation['sample_token'])
            for s in sensor_channels:
                if s.startswith('CAM'):
                    cam_data = nusc.get('sample_data', sample['data'][s])
                    if check_point_in_camera(nusc, cam_data, annotation):
                        if s.startswith('CAM_FRONT'):
                            f += 1
                        else:
                            b += 1
                elif s.startswith('RADAR'):
                    radar_data = nusc.get('sample_data', sample['data'][s])
                    radar_data_path = os.path.join(args.path, radar_data['filename'])
                    if check_point_in_radar(radar_data_path, annotation):
                        r += 1
                else:
                    lidar_data = nusc.get('sample_data', sample['data'][s])
                    lidar_data_path = os.path.join(args.path, lidar_data['filename'])
                    if check_point_in_lidar(lidar_data_path, annotation):
                        lidar += 1
            front += (f!=0)
            back += (b!=0)
            radar += (r!=0)
            camera_lost += (f==0 and b==0)
    print(f'camera\tfront/total/instances:\t{front}/{total}/{len(instances)}')
    print(f'camera\tback/total/instances:\t{back}/{total}/{len(instances)}')
    print(f'camera\tlost/total/instances:\t{camera_lost}/{total}/{len(instances)}')
    print(f'radar\tradar/total/instances:\t{radar}/{total}/{len(instances)}')
    print(f'lidar\tlidar/total/instances:\t{lidar}/{total}/{len(instances)}')
