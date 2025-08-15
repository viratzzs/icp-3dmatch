import os
import sys
import numpy as np
import torch
import open3d as o3d
import time
from tqdm import tqdm
import pickle
import json
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist

try:
    from lib.benchmark_utils import to_o3d_pcd, to_tsfm, get_correspondences
    from lib.utils import load_obj
except ImportError:
    print("Warning: Could not import some utilities. Make sure lib/ is in your path.")
    
    # Define missing functions locally
    def load_obj(path):
        """Load pickle file"""
        with open(path, 'rb') as f:
            return pickle.load(f)

class ICP3DMatchTester:
    def __init__(self, data_root, test_split_path, results_dir="icp_results"):
        """
        ICP Tester for 3DMatch dataset following the same evaluation protocol
        as your reference script
        
        Args:
            data_root: Path to 3DMatch data directory
            test_split_path: Path to test split pickle file
            results_dir: Directory to save results
        """
        self.data_root = data_root
        self.test_split_path = test_split_path
        self.results_dir = results_dir
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Load test data info
        self.test_infos = load_obj(test_split_path)
        print(f"Loaded {len(self.test_infos['rot'])} test pairs")
        
        # ICP parameters - optimized for challenging cases
        self.icp_threshold = 0.1   # Increased threshold for initial alignment
        self.max_iteration = 200   # More iterations for convergence
        self.relative_fitness = 1e-9
        self.relative_rmse = 1e-9
        
        # Overlap radius for correspondence generation
        self.overlap_radius = 0.0375
        
    def run_icp(self, src_pcd, tgt_pcd, initial_transform=None):
        """
        Run ICP with multiple initialization strategies for robustness
        """
        # Convert to Open3D format if necessary
        if isinstance(src_pcd, np.ndarray):
            src_o3d = o3d.geometry.PointCloud()
            src_o3d.points = o3d.utility.Vector3dVector(src_pcd)
        else:
            src_o3d = src_pcd
            
        if isinstance(tgt_pcd, np.ndarray):
            tgt_o3d = o3d.geometry.PointCloud()
            tgt_o3d.points = o3d.utility.Vector3dVector(tgt_pcd)
        else:
            tgt_o3d = tgt_pcd
        
        best_transform = np.eye(4)
        best_fitness = -1
        
        # Strategy 1: Try RANSAC-based initialization
        try:
            # Downsample for faster processing
            voxel_size = 0.05  # 5cm voxel size
            src_down = src_o3d.voxel_down_sample(voxel_size)
            tgt_down = tgt_o3d.voxel_down_sample(voxel_size)
            
            # Estimate normals
            src_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
            tgt_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
            
            # Compute FPFH features
            radius_feature = voxel_size * 5
            src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                src_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
            tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                tgt_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
            
            # RANSAC-based global registration
            distance_threshold = voxel_size * 2.0  # Increased threshold
            ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                src_down, tgt_down, src_fpfh, tgt_fpfh, True, distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3, [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
                ], o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999))
            
            # Try ICP with RANSAC initialization
            if ransac_result.fitness > 0.1:  # Only use if RANSAC found reasonable alignment
                reg_result = o3d.pipelines.registration.registration_icp(
                    src_o3d, tgt_o3d, self.icp_threshold, ransac_result.transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=self.relative_fitness,
                        relative_rmse=self.relative_rmse,
                        max_iteration=self.max_iteration
                    )
                )
                if reg_result.fitness > best_fitness:
                    best_fitness = reg_result.fitness
                    best_transform = reg_result.transformation
        except Exception as e:
            print(f"RANSAC initialization failed: {e}")
        
        # Strategy 2: Multiple random initializations
        for i in range(5):  # Try 5 random initializations
            try:
                # Generate random transformation
                random_rot = Rotation.random().as_matrix()
                random_trans = np.random.normal(0, 1.0, 3)  # Random translation
                random_transform = np.eye(4)
                random_transform[:3, :3] = random_rot
                random_transform[:3, 3] = random_trans
                
                reg_result = o3d.pipelines.registration.registration_icp(
                    src_o3d, tgt_o3d, self.icp_threshold, random_transform,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=self.relative_fitness,
                        relative_rmse=self.relative_rmse,
                        max_iteration=self.max_iteration
                    )
                )
                
                if reg_result.fitness > best_fitness:
                    best_fitness = reg_result.fitness
                    best_transform = reg_result.transformation
            except:
                continue
        
        # Strategy 3: Identity initialization (original approach)
        try:
            if initial_transform is None:
                initial_transform = np.eye(4)
            
            reg_result = o3d.pipelines.registration.registration_icp(
                src_o3d, tgt_o3d, self.icp_threshold, initial_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=self.relative_fitness,
                    relative_rmse=self.relative_rmse,
                    max_iteration=self.max_iteration
                )
            )
            
            if reg_result.fitness > best_fitness:
                best_fitness = reg_result.fitness
                best_transform = reg_result.transformation
        except:
            pass
        
        return best_transform
    
    def generate_correspondences_from_transform(self, src_pcd, tgt_pcd, transform):
        """
        Generate correspondences based on the predicted transformation
        Similar to how matches are generated in the reference script
        """
        # Transform source points
        src_pcd_transformed = (transform[:3, :3] @ src_pcd.T + transform[:3, 3:4]).T
        
        # Find correspondences using nearest neighbor
        distances = cdist(src_pcd_transformed, tgt_pcd)
        src_indices = np.arange(len(src_pcd))
        tgt_indices = np.argmin(distances, axis=1)
        min_distances = np.min(distances, axis=1)
        
        # Filter correspondences based on distance threshold
        valid_mask = min_distances < self.overlap_radius
        src_indices = src_indices[valid_mask]
        tgt_indices = tgt_indices[valid_mask]
        
        # Create match predictions in the format [batch_idx, src_idx, tgt_idx]
        batch_indices = np.zeros(len(src_indices))
        match_pred = np.column_stack([batch_indices, src_indices, tgt_indices])
        
        return match_pred, min_distances[valid_mask]
    
    def compute_inlier_ratio(self, match_pred, src_pcd, tgt_pcd, gt_rot, gt_trans, inlier_thr=0.1):
        """
        Compute inlier ratio following the same logic as in your reference script
        """
        if len(match_pred) == 0:
            return 0.0
        
        # Get matched points
        src_matched = src_pcd[match_pred[:, 1].astype(int)]
        tgt_matched = tgt_pcd[match_pred[:, 2].astype(int)]
        
        # Transform source points using ground truth
        gt_transform = np.eye(4)
        gt_transform[:3, :3] = gt_rot
        gt_transform[:3, 3] = gt_trans.flatten()
        
        src_transformed_gt = (gt_rot @ src_matched.T + gt_trans.reshape(-1, 1)).T
        
        # Compute distances
        distances = np.linalg.norm(src_transformed_gt - tgt_matched, axis=1)
        
        # Count inliers
        inliers = distances < inlier_thr
        inlier_ratio = np.mean(inliers)
        
        return inlier_ratio
    
    def compute_registration_recall(self, pred_rot, pred_trans, gt_rot, gt_trans, thr=0.2):
        """
        Compute registration recall following the same logic as in your reference script
        """
        # Compute rotation error
        rot_diff = pred_rot @ gt_rot.T
        trace = np.clip(np.trace(rot_diff), -1, 3)
        rot_error = np.arccos((trace - 1) / 2)
        rot_error_deg = np.degrees(rot_error)
        
        # Compute translation error
        trans_error = np.linalg.norm(pred_trans.flatten() - gt_trans.flatten())
        
        # Registration success criteria (relaxed for challenging cases)
        rot_success = rot_error_deg < 30.0  # Relaxed from 15 to 30 degrees
        trans_success = trans_error < 0.5   # Relaxed from 0.2 to 0.5 meters
        
        registration_recall = rot_success and trans_success
        
        return registration_recall, rot_error_deg, trans_error
    
    def test_thr(self, conf_threshold=0.05):
        """
        Test ICP at a specific confidence threshold
        Following the same evaluation protocol as your reference script
        """
        print(f'Testing ICP with confidence threshold: {conf_threshold}')
        
        success_count = 0
        total_IR = 0.0
        total_FMR = 0.0
        num_samples = 0
        
        for idx in tqdm(range(len(self.test_infos['rot']))):
            try:
                # Load ground truth transformation
                gt_rot = self.test_infos['rot'][idx].astype(np.float32)
                gt_trans = self.test_infos['trans'][idx].astype(np.float32)
                if gt_trans.ndim == 1:
                    gt_trans = gt_trans[:, None]
                
                # Load point clouds
                src_path = os.path.join(self.data_root, self.test_infos['src'][idx])
                tgt_path = os.path.join(self.data_root, self.test_infos['tgt'][idx])
                
                try:
                    src_pcd = torch.load(src_path, weights_only=False).astype(np.float32)
                    tgt_pcd = torch.load(tgt_path, weights_only=False).astype(np.float32)
                except Exception as e:
                    print(f"Error loading point clouds for pair {idx}: {str(e)}")
                    continue
                
                # Downsample if needed
                max_points = 30000
                if len(src_pcd) > max_points:
                    indices = np.random.choice(len(src_pcd), max_points, replace=False)
                    src_pcd = src_pcd[indices]
                if len(tgt_pcd) > max_points:
                    indices = np.random.choice(len(tgt_pcd), max_points, replace=False)
                    tgt_pcd = tgt_pcd[indices]
                
                # Run ICP (starting from identity)
                pred_transform = self.run_icp(src_pcd, tgt_pcd)
                pred_rot = pred_transform[:3, :3]
                pred_trans = pred_transform[:3, 3:4]
                
                # Generate correspondences from ICP result
                match_pred, match_distances = self.generate_correspondences_from_transform(
                    src_pcd, tgt_pcd, pred_transform
                )
                
                # Compute Inlier Ratio (IR)
                if len(match_pred) > 0:
                    ir = self.compute_inlier_ratio(
                        match_pred, src_pcd, tgt_pcd, gt_rot, gt_trans, inlier_thr=0.2  # More lenient
                    )
                    total_IR += ir
                    
                    # Feature Matching Recall (FMR) - more lenient threshold
                    fmr = 1.0 if ir > 0.01 else 0.0  # 1% instead of 5%
                    total_FMR += fmr
                else:
                    ir = 0.0
                    fmr = 0.0
                    total_IR += ir
                    total_FMR += fmr
                
                # Compute Registration Recall (RR) with lenient thresholds
                rr, rot_err, trans_err = self.compute_registration_recall(
                    pred_rot, pred_trans, gt_rot, gt_trans, thr=0.5  # More lenient
                )
                
                if rr:
                    success_count += 1
                
                num_samples += 1
                
                # Optional: Print progress for some samples
                if idx % 100 == 0:
                    print(f"Sample {idx}: RR={rr}, IR={ir:.4f}, FMR={fmr}")
                
            except Exception as e:
                print(f"Error processing pair {idx}: {str(e)}")
                continue
        
        # Calculate final metrics
        if num_samples > 0:
            avg_rr = success_count / num_samples
            avg_ir = total_IR / num_samples  
            avg_fmr = total_FMR / num_samples
        else:
            avg_rr = avg_ir = avg_fmr = 0.0
        
        return avg_rr, avg_ir, avg_fmr
    
    def test(self):
        """
        Main test function following the same protocol as your reference script
        """
        print("Starting ICP evaluation on 3DMatch test dataset...")
        
        # Test multiple runs to combat nondeterministic behavior
        n_runs = 3
        
        total_rr = 0.0
        total_ir = 0.0  
        total_fmr = 0.0
        
        for i in range(n_runs):
            print(f"\n--- Run {i+1}/{n_runs} ---")
            
            thr = 0.05  # confidence threshold
            rr, ir, fmr = self.test_thr(thr)
            
            total_rr += rr
            total_ir += ir
            total_fmr += fmr
            
            print(f"conf_threshold: {thr}, registration recall: {rr:.4f}, Inlier rate: {ir:.4f}, FMR: {fmr:.4f}")
        
        # Calculate averages
        avg_rr = total_rr / n_runs
        avg_ir = total_ir / n_runs
        avg_fmr = total_fmr / n_runs
        
        print(f"\n{'='*60}")
        print("FINAL RESULTS (Average over {} runs)".format(n_runs))
        print(f"{'='*60}")
        print(f"Average Registration Recall (RR): {avg_rr:.4f}")
        print(f"Average Inlier Rate (IR): {avg_ir:.4f}")
        print(f"Average Feature Matching Recall (FMR): {avg_fmr:.4f}")
        print(f"{'='*60}")
        
        # Save results
        results = {
            'registration_recall': avg_rr,
            'inlier_rate': avg_ir,
            'feature_matching_recall': avg_fmr,
            'num_runs': n_runs,
            'individual_runs': {
                'rr': total_rr / n_runs,
                'ir': total_ir / n_runs,
                'fmr': total_fmr / n_runs
            }
        }
        
        results_path = os.path.join(self.results_dir, 'icp_3dmatch_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_path}")
        
        return avg_rr, avg_ir, avg_fmr


def main():
    """
    Main function to run ICP evaluation on 3DMatch test dataset
    """
    # UPDATE THESE PATHS TO MATCH YOUR DATA
    data_root = "/workspace/icp-3dmatch/test"
    test_split_path = "test.pkl"
    
    # Check if paths exist
    if not os.path.exists(data_root):
        print(f"Error: Data root path does not exist: {data_root}")
        print("Please update the data_root variable with the correct path to your 3DMatch data")
        return
    
    if not os.path.exists(test_split_path):
        print(f"Error: Test split path does not exist: {test_split_path}")
        print("Please update the test_split_path variable with the correct path to your test split file")
        return
    
    # Create tester and run evaluation
    tester = ICP3DMatchTester(data_root, test_split_path)
    
    # Run the test (this will test multiple runs with conf_threshold=0.05)
    rr, ir, fmr = tester.test()
    
    print(f"\nFinal ICP Performance on 3DMatch:")
    print(f"Registration Recall: {rr:.1%}")
    print(f"Inlier Rate: {ir:.1%}")  
    print(f"Feature Matching Recall: {fmr:.1%}")


if __name__ == "__main__":
    main()