import os
import torch
import numpy as np
import pickle
from icp import ICP3DMatchTester

# Load test data
with open('test.pkl', 'rb') as f:
    test_infos = pickle.load(f)

data_root = "/workspace/icp-3dmatch/test"

# Create ICP tester instance
tester = ICP3DMatchTester(data_root, 'test.pkl')

# Test multiple pairs to see if it's just pair 0
for idx in [0, 50, 100, 150, 200]:
    print(f"\n=== Testing pair {idx} ===")
    
    try:
        # Load ground truth transformation
        gt_rot = test_infos['rot'][idx].astype(np.float32)
        gt_trans = test_infos['trans'][idx].astype(np.float32)
        if gt_trans.ndim == 1:
            gt_trans = gt_trans[:, None]
        
        # Load point clouds
        src_path = os.path.join(data_root, test_infos['src'][idx])
        tgt_path = os.path.join(data_root, test_infos['tgt'][idx])
        
        print(f"Scene: {test_infos['src'][idx].split('/')[0]}")
        print(f"Files: {os.path.basename(test_infos['src'][idx])} -> {os.path.basename(test_infos['tgt'][idx])}")
        
        src_pcd = torch.load(src_path, weights_only=False).astype(np.float32)
        tgt_pcd = torch.load(tgt_path, weights_only=False).astype(np.float32)
        
        print(f"Point cloud sizes: {src_pcd.shape} -> {tgt_pcd.shape}")
        
        # Check ground truth transformation magnitude
        gt_trans_norm = np.linalg.norm(gt_trans.flatten())
        rot_angle = np.degrees(np.arccos((np.trace(gt_rot) - 1) / 2))
        print(f"GT transformation: rotation={rot_angle:.1f}°, translation={gt_trans_norm:.3f}m")
        
        # Run improved ICP
        pred_transform = tester.run_icp(src_pcd, tgt_pcd)
        pred_rot = pred_transform[:3, :3]
        pred_trans = pred_transform[:3, 3:4]
        
        # Compute errors
        rr, rot_err, trans_err = tester.compute_registration_recall(pred_rot, pred_trans, gt_rot, gt_trans, 0.2)
        
        print(f"Results: rotation_error={rot_err:.1f}°, translation_error={trans_err:.3f}m, success={rr}")
        
    except Exception as e:
        print(f"Error processing pair {idx}: {e}")

print("\n=== Summary ===")
print("If all pairs are failing, there might be an issue with:")
print("1. Ground truth transformation format/coordinate system")
print("2. Point cloud coordinate system")
print("3. ICP algorithm or parameters")
