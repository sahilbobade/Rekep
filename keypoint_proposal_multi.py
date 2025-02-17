import numpy as np
import torch
import cv2
from torch.nn.functional import interpolate
from kmeans_pytorch import kmeans
from utils import filter_points_by_bounds
from sklearn.cluster import MeanShift

class KeypointProposer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config['device'])
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().to(self.device)
        # local_model_path = '/omnigibson-src/ReKep/dinov2_vits14_pretrain.pth'
        # checkpoint = torch.load(local_model_path)
        # self.dinov2 = checkpoint
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.mean_shift = MeanShift(bandwidth=self.config['min_dist_bt_keypoints'], bin_seeding=True, n_jobs=32)
        self.patch_size = 14  # dinov2
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])

    def get_keypoints(self, rgb1, points1, masks1, rgb2, points2, masks2):
        """
        Accepts inputs from two cameras.
         - For each view, runs the keypoint detection pipeline.
         - Merges the candidate 3D keypoints (and keeps track of which candidate came from which view).
         - Projects each unique keypoint back onto its original image.
        
        Returns:
         - global_keypoints: N x 3 array of unique 3D keypoints (in world frame).
         - projected1: rgb image from camera 1 with overlaid keypoint numbers.
         - projected2: rgb image from camera 2 with overlaid keypoint numbers.
        """
        # ----- Process view 1 -----
        transformed_rgb1, rgb1_orig, points1, masks1, shape_info1 = self._preprocess(rgb1, points1, masks1)
        features_flat1 = self._get_features(transformed_rgb1, shape_info1)
        candidate_keypoints1, candidate_pixels1, candidate_rigid_group_ids1 = self._cluster_features(points1, features_flat1, masks1)
        within_space1 = filter_points_by_bounds(candidate_keypoints1, self.bounds_min, self.bounds_max, strict=True)
        candidate_keypoints1 = candidate_keypoints1[within_space1]
        candidate_pixels1 = candidate_pixels1[within_space1]
        candidate_rigid_group_ids1 = candidate_rigid_group_ids1[within_space1]
        view_ids1 = np.ones(len(candidate_keypoints1), dtype=int)  # mark these as coming from view 1

        # ----- Process view 2 -----
        transformed_rgb2, rgb2_orig, points2, masks2, shape_info2 = self._preprocess(rgb2, points2, masks2)
        features_flat2 = self._get_features(transformed_rgb2, shape_info2)
        candidate_keypoints2, candidate_pixels2, candidate_rigid_group_ids2 = self._cluster_features(points2, features_flat2, masks2)
        within_space2 = filter_points_by_bounds(candidate_keypoints2, self.bounds_min, self.bounds_max, strict=True)
        candidate_keypoints2 = candidate_keypoints2[within_space2]
        candidate_pixels2 = candidate_pixels2[within_space2]
        candidate_rigid_group_ids2 = candidate_rigid_group_ids2[within_space2]
        view_ids2 = 2 * np.ones(len(candidate_keypoints2), dtype=int)  # mark these as coming from view 2

        # ----- Combine candidates from both views -----
        all_keypoints = np.concatenate([candidate_keypoints1, candidate_keypoints2], axis=0)
        all_pixels    = np.concatenate([candidate_pixels1, candidate_pixels2], axis=0)
        all_rigid_ids = np.concatenate([candidate_rigid_group_ids1, candidate_rigid_group_ids2], axis=0)
        all_view_ids  = np.concatenate([view_ids1, view_ids2], axis=0)

        # ----- Merge duplicates using mean-shift clustering in 3D -----
        merged_indices = self._merge_clusters(all_keypoints)
        global_keypoints = all_keypoints[merged_indices]
        global_pixels    = all_pixels[merged_indices]
        global_rigid_ids = all_rigid_ids[merged_indices]
        global_view_ids  = all_view_ids[merged_indices]

        # ----- Sort global keypoints (using 3D coordinates for a consistent order) -----
        sort_idx = np.lexsort((global_keypoints[:, 0], global_keypoints[:, 1]))
        global_keypoints = global_keypoints[sort_idx]
        global_pixels    = global_pixels[sort_idx]
        global_rigid_ids = global_rigid_ids[sort_idx]
        global_view_ids  = global_view_ids[sort_idx]

        # ----- Project keypoints back onto their original images -----
        projected1 = rgb1_orig.copy()
        projected2 = rgb2_orig.copy()
        # Use the global index (after sorting) as the consistent keypoint number.
        for i, (pixel, view_id) in enumerate(zip(global_pixels, global_view_ids)):
            if view_id == 1:
                projected1 = self._overlay_keypoint(projected1, pixel, i)
            elif view_id == 2:
                projected2 = self._overlay_keypoint(projected2, pixel, i)

        return global_keypoints, projected1, projected2

    def _preprocess(self, rgb, points, masks):
        if masks.is_cuda:
            masks = masks.cpu()
        rgb = rgb.cpu().numpy()  # move to CPU and convert to numpy
        # convert masks to binary masks for each unique id
        masks = [masks == uid for uid in np.unique(masks.numpy())]
        # ensure input shape is compatible with dinov2
        H, W, _ = rgb.shape
        patch_h = int(H // self.patch_size)
        patch_w = int(W // self.patch_size)
        new_H = patch_h * self.patch_size
        new_W = patch_w * self.patch_size
        transformed_rgb = cv2.resize(rgb, (new_W, new_H))
        transformed_rgb = transformed_rgb.astype(np.float32) / 255.0  # scale to [0,1]
        shape_info = {
            'img_h': H,
            'img_w': W,
            'patch_h': patch_h,
            'patch_w': patch_w,
        }
        return transformed_rgb, rgb, points, masks, shape_info
    
    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def _get_features(self, transformed_rgb, shape_info):
        img_h = shape_info['img_h']
        img_w = shape_info['img_w']
        patch_h = shape_info['patch_h']
        patch_w = shape_info['patch_w']
        # Prepare image tensor
        img_tensors = torch.from_numpy(transformed_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        assert img_tensors.shape[1] == 3, "unexpected image shape"
        features_dict = self.dinov2.forward_features(img_tensors)
        raw_feature_grid = features_dict['x_norm_patchtokens']  # shape: [1, patch_h*patch_w, feature_dim]
        raw_feature_grid = raw_feature_grid.reshape(1, patch_h, patch_w, -1)
        # Bilinear interpolate to original image size
        interpolated_feature_grid = interpolate(raw_feature_grid.permute(0, 3, 1, 2),
                                                size=(img_h, img_w),
                                                mode='bilinear').permute(0, 2, 3, 1).squeeze(0)
        features_flat = interpolated_feature_grid.reshape(-1, interpolated_feature_grid.shape[-1])
        return features_flat

    def _cluster_features(self, points, features_flat, masks):
        candidate_keypoints = []
        candidate_pixels = []
        candidate_rigid_group_ids = []
        # Process each binary mask (each unique object/region)
        for rigid_group_id, binary_mask in enumerate(masks):
            binary_mask = binary_mask.cpu().numpy()
            if np.mean(binary_mask) > self.config['max_mask_ratio']:
                continue
            # Select foreground features
            obj_features_flat = features_flat[binary_mask.reshape(-1)]
            feature_pixels = np.argwhere(binary_mask)
            feature_points = points[binary_mask]
            # Dimensionality reduction via PCA to reduce noise sensitivity
            obj_features_flat = obj_features_flat.double()
            (u, s, v) = torch.pca_lowrank(obj_features_flat, center=False)
            features_pca = torch.mm(obj_features_flat, v[:, :3])
            features_pca = (features_pca - features_pca.min(0)[0]) / (features_pca.max(0)[0] - features_pca.min(0)[0])
            X = features_pca
            # Append normalized pixel coordinates as extra dimensions
            feature_points_torch = torch.tensor(feature_pixels, dtype=features_pca.dtype, device=features_pca.device)
            feature_points_torch = (feature_points_torch - feature_points_torch.min(0)[0]) / (feature_points_torch.max(0)[0] - feature_points_torch.min(0)[0])
            X = torch.cat([X, feature_points_torch], dim=-1)
            # Cluster features to get candidate regions
            cluster_ids_x, cluster_centers = kmeans(
                X=X,
                num_clusters=self.config['num_candidates_per_mask'],
                distance='euclidean',
                device=self.device,
            )
            cluster_centers = cluster_centers.to(self.device)
            for cluster_id in range(self.config['num_candidates_per_mask']):
                cluster_center = cluster_centers[cluster_id][:3]
                member_idx = cluster_ids_x == cluster_id
                member_points = feature_points[member_idx]
                member_pixels = feature_pixels[member_idx]
                member_features = features_pca[member_idx]
                dist = torch.norm(member_features - cluster_center, dim=-1)
                closest_idx = torch.argmin(dist)
                candidate_keypoints.append(member_points[closest_idx])
                candidate_pixels.append(member_pixels[closest_idx])
                candidate_rigid_group_ids.append(rigid_group_id)
        candidate_keypoints = np.array(candidate_keypoints)
        candidate_pixels = np.array(candidate_pixels)
        candidate_rigid_group_ids = np.array(candidate_rigid_group_ids)
        return candidate_keypoints, candidate_pixels, candidate_rigid_group_ids

    def _merge_clusters(self, candidate_keypoints):
        """
        Merges candidate keypoints (from one or multiple views) using MeanShift.
        Returns the indices (from the input array) corresponding to the chosen cluster representatives.
        """
        self.mean_shift.fit(candidate_keypoints)
        cluster_centers = self.mean_shift.cluster_centers_
        merged_indices = []
        for center in cluster_centers:
            dist = np.linalg.norm(candidate_keypoints - center, axis=-1)
            merged_indices.append(np.argmin(dist))
        return merged_indices

    def _overlay_keypoint(self, image, pixel, keypoint_number):
        """
        Draws a box and number on the provided image at the specified pixel location.
        """
        displayed_text = f"{keypoint_number}"
        text_length = len(displayed_text)
        box_width = self.config['box_width'] + 5 * (text_length - 1)
        box_height = self.config['box_height']
        top_left = (int(pixel[1] - box_width // 2), int(pixel[0] - box_height // 2))
        bottom_right = (int(pixel[1] + box_width // 2), int(pixel[0] + box_height // 2))
        cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), -1)
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), 2)
        org = (int(pixel[1] - 7 * (text_length)), int(pixel[0] + 7))
        cv2.putText(image, str(keypoint_number), org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return image
