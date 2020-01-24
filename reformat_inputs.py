from utils_satellite import *
dataset_root_dir = '/cvlabdata1/cvlab/datasets_kgerard/speed'
dataset = SatellitePoseEstimationDataset(root_dir=dataset_root_dir)

dataset.convert_input()
