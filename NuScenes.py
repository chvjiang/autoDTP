from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot='DATA/', verbose=True)

nusc.list_scenes()
