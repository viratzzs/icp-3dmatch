import os
import torch
import numpy as np
import open3d as o3d
import pickle
from icp import ICP3DMatchTester

# Load test data
with open('test.pkl', 'rb') as f:
    test_infos = pickle.load(f)

data_root = "/workspace/icp-3dmatch/test"

# Create ICP tester instance
tester = ICP3DMatchTester(data_root, 'test.pkl')

# Test the first pair exactly as the main script does
idx = 0
print(f"Testing pair {idx}")

# Load ground truth transformation
gt_rot = test_infos['rot'][idx].astype(np.float32)
gt_trans = test_infos['trans'][idx].astype(np.float32)
if gt_trans.ndim == 1:
    gt_trans = gt_trans[:, None]

print(f"GT rotation shape: {gt_rot.shape}")
print(f"GT translation shape: {gt_trans.shape}")
print("GT rotation matrix:")
print(gt_rot)
print("GT translation:")
print(gt_trans.flatten())

# Load point clouds
src_path = os.path.join(data_root, test_infos['src'][idx])
tgt_path = os.path.join(data_root, test_infos['tgt'][idx])

print(f"Source path: {src_path}")
print(f"Target path: {tgt_path}")

src_pcd = torch.load(src_path, weights_only=False).astype(np.float32)
tgt_pcd = torch.load(tgt_path, weights_only=False).astype(np.float32)

print(f"Source shape: {src_pcd.shape}")
print(f"Target shape: {tgt_pcd.shape}")

# Run improved ICP using the tester class
print("Running improved ICP with RANSAC initialization...")
pred_transform = tester.run_icp(src_pcd, tgt_pcd)
pred_rot = pred_transform[:3, :3]
pred_trans = pred_transform[:3, 3:4]

print("Predicted transformation:")
print(pred_transform)
print("Predicted rotation:")
print(pred_rot)
print("Predicted translation:")
print(pred_trans.flatten())

# Test registration recall computation
rr, rot_err, trans_err = tester.compute_registration_recall(pred_rot, pred_trans, gt_rot, gt_trans, 0.2)

print(f"Rotation error: {rot_err:.2f} degrees")
print(f"Translation error: {trans_err:.4f} meters")
print(f"Registration recall: {rr}")
print(f"Rotation success (< 15deg): {rot_err < 15.0}")
print(f"Translation success (< 0.2m): {trans_err < 0.2}")
