import torch
import numpy as np
import json
import os
import argparse
import cv2
from environment import ReKepOGEnv
# from keypoint_proposal import KeypointProposer
from keypoint_proposal_multi import KeypointProposer
# (Other imports such as ConstraintGenerator, IKSolver, etc. are not needed here.)
from utils import bcolors, get_config

class Main:
    def __init__(self, scene_file, visualize=False):
        global_config = get_config(config_path="./configs/config.yaml")
        self.config = global_config['main']
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.visualize = visualize
        # set random seed
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        # initialize the keypoint proposer (which now supports two cameras)
        self.keypoint_proposer = KeypointProposer(global_config['keypoint_proposer'])
        # initialize environment
        self.env = ReKepOGEnv(global_config['env'], scene_file, verbose=False)

    def perform_task(self, instruction, rekep_program_dir=None, disturbance_seq=None):
        # Reset the environment and get camera observations.
        self.env.reset()
        cam_obs = self.env.get_cam_obs()
        # Retrieve data from camera 1 (using key from self.config, e.g. "vlm_camera")
        rgb1 = cam_obs[self.config['vlm_camera']]['rgb']
        points1 = cam_obs[self.config['vlm_camera']]['points']
        mask1 = cam_obs[self.config['vlm_camera']]['seg']
        # Retrieve data from camera 2 (using key "camera_2")
        rgb2 = cam_obs[self.config['camera_2']]['rgb']
        points2 = cam_obs[self.config['camera_2']]['points']
        mask2 = cam_obs[self.config['camera_2']]['seg']

        # Run keypoint proposal on the two views. This function returns:
        #   - global_keypoints: merged 3D keypoints
        #   - projected_img1: keypoint overlay on camera 1 image
        #   - projected_img2: keypoint overlay on camera 2 image
        keypoints, projected_img1, projected_img2 = self.keypoint_proposer.get_keypoints(
            rgb1, points1, mask1, rgb2, points2, mask2
        )
        print(f"{bcolors.HEADER}Got {len(keypoints)} proposed keypoints from two cameras{bcolors.ENDC}")

        # Display the projected images using OpenCV.
        cv2.imshow("Projected Image - Camera 1", projected_img1)
        cv2.imshow("Projected Image - Camera 2", projected_img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='pen', help='task to perform')
    parser.add_argument('--use_cached_query', action='store_true', help='instead of querying the VLM, use the cached query')
    parser.add_argument('--apply_disturbance', action='store_true', help='apply disturbance to test the robustness')
    parser.add_argument('--visualize', action='store_true', help='visualize each solution before executing (NOTE: this is blocking and needs to press "ESC" to continue)')
    args = parser.parse_args()

    # For this demo we only show the keypoint projection. No optimization or execution is performed.
    task_list = {
        'pen': {
            'scene_file': './configs/og_scene_file_bottle.json',
            'instruction': 'pickup the red bottle and pour water in the black cup',
            'rekep_program_dir': './vlm_query/pen',
        },
    }
    task = task_list['pen']
    scene_file = task['scene_file']
    instruction = task['instruction']
    main = Main(scene_file, visualize=args.visualize)
    main.perform_task(instruction)
