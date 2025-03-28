from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box, LidarPointCloud
from pyquaternion import Quaternion
import numpy as np
import json 

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

def check_point_in_lidar(nusc, lidar_data, annotation):
    points = nusc.get_sample_data(lidar_data['token'])[0]
    pc = LidarPointCloud(points.T)
    box = Box(
        annotation['translation'], annotation['size'], Quaternion(annotation['rotation']),
        name=annotation['category_name'], token=annotation['token']
    )
    inside = box.in_range(pc.points[:3, :])
    return np.any(inside)

if __name__=='__main__':
    nusc = NuScenes(version='v1.0-mini', dataroot='DATA/', verbose=True)

    with open('DATA/v1.0-mini/instance.json','r') as f:
        instances = json.load(f)
    with open('DATA/v1.0-mini/category.json','r') as f:
        categories = json.load(f)

    token2category = {c['token']:c['name'] for c in categories}
    radar, lidar, front, back, total = 0, 0, 0, 0, 0
    sensor_channels = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
                    'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'LIDAR_TOP']
    camera_lost, radar_lost = 0, 0

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
                    pass
                else:
                    lidar_data = nusc.get('sample_data', sample['data'][s])
                    if check_point_in_lidar(nusc, lidar_data, annotation):
                        lidar += 1
            front += (f!=0)
            back += (b!=0)
            radar += (r!=0)
            camera_lost += (f==0 and b==0)
            radar_lost += (r==0)
    print(f'camera\tfront/total/instances:\t{front}/{total}/{len(instances)}')
    print(f'camera\tback/total/instances:\t{back}/{total}/{len(instances)}')
    print(f'camera\tlost/total/instances:\t{camera_lost}/{total}/{len(instances)}')
    print(f'radar\tradar/total/instances:\t{radar}/{total}/{len(instances)}')
    print(f'radar\tlost/total/instances:\t{radar_lost}/{total}/{len(instances)}')
    print(f'lidar\tlidar/total/instances:\t{lidar}/{total}/{len(instances)}')
