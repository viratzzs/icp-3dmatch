import os
import glob
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
import re

def parse_transformation_file(trans_file):
    """Parse transformation file to get rotation and translation"""
    try:
        with open(trans_file, 'r') as f:
            lines = f.readlines()
        
        # Parse 4x4 transformation matrix
        matrix = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Skip lines that don't look like matrix rows (e.g., scene info)
            if not any(char.isdigit() or char == '-' or char == 'e' or char == 'E' or char == '.' for char in line):
                continue
            
            # Try to parse as matrix row
            parts = line.split()
            # Filter out non-numeric parts (like scene names)
            numeric_parts = []
            for part in parts:
                try:
                    float(part)
                    numeric_parts.append(part)
                except ValueError:
                    continue
            
            if len(numeric_parts) == 4:
                row = [float(x) for x in numeric_parts]
                matrix.append(row)
        
        if len(matrix) == 4:
            transform = np.array(matrix)
            rotation = transform[:3, :3]
            translation = transform[:3, 3]
            return rotation, translation
        else:
            return None, None
    except:
        return None, None

def find_point_cloud_files(scene_dir):
    """Find all .pth files in a scene directory"""
    pth_files = glob.glob(os.path.join(scene_dir, "**/*.pth"), recursive=True)
    return sorted(pth_files)

def extract_frame_number(filepath):
    """Extract frame number from filepath for proper ordering"""
    basename = os.path.basename(filepath)
    # Look for numbers in the filename
    numbers = re.findall(r'\d+', basename)
    if numbers:
        return int(numbers[-1])  # Use the last number found
    return 0

def create_test_pairs_from_scenes(data_root, max_pairs_per_scene=50):
    """
    Create test pairs from scene directories
    This is a simplified approach - in practice, 3DMatch uses specific pair selections
    """
    
    print("Scanning for scene directories...")
    scene_dirs = []
    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path):
            scene_dirs.append(item_path)
    
    print(f"Found {len(scene_dirs)} scene directories")
    
    src_list = []
    tgt_list = []
    rot_list = []
    trans_list = []
    
    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        print(f"Processing scene: {scene_name}")
        
        # Find all point cloud files
        pt_files = find_point_cloud_files(scene_dir)
        
        if len(pt_files) < 2:
            print(f"  Skipping {scene_name}: only {len(pt_files)} point clouds found")
            continue
            
        print(f"  Found {len(pt_files)} point cloud files")
        
        # Sort files by frame number if possible
        try:
            pt_files = sorted(pt_files, key=extract_frame_number)
        except:
            pt_files = sorted(pt_files)
        
        # Look for transformation files
        trans_files = []
        trans_extensions = ['.info.txt', '.txt', '.log', '.trans']
        for ext in trans_extensions:
            if ext == '.info.txt':
                trans_files.extend(glob.glob(os.path.join(scene_dir, f"**/*{ext}"), recursive=True))
            else:
                trans_files.extend(glob.glob(os.path.join(scene_dir, f"**/*{ext}"), recursive=True))
        
        print(f"  Found {len(trans_files)} potential transformation files")
        
        # Strategy 1: If we have transformation files, try to parse them
        if trans_files:
            pairs_added = 0
            for i, src_file in enumerate(pt_files):
                for j, tgt_file in enumerate(pt_files):
                    if i >= j or pairs_added >= max_pairs_per_scene:
                        continue
                    
                    # Look for corresponding transformation file
                    src_base = os.path.splitext(os.path.basename(src_file))[0]
                    tgt_base = os.path.splitext(os.path.basename(tgt_file))[0]
                    
                    # Try different transformation file naming conventions
                    possible_trans_names = [
                        f"{src_base}.info.txt",  # Most likely format for this data
                        f"{tgt_base}.info.txt",  # Also try target's info file
                        f"{src_base}_{tgt_base}.txt",
                        f"{src_base}-{tgt_base}.txt",
                        f"{i:06d}_{j:06d}.txt",
                        f"{i}-{j}.txt"
                    ]
                    
                    trans_file = None
                    for trans_name in possible_trans_names:
                        full_trans_path = os.path.join(scene_dir, trans_name)
                        if os.path.exists(full_trans_path):
                            trans_file = full_trans_path
                            break
                    
                    if trans_file:
                        rot, trans = parse_transformation_file(trans_file)
                        if rot is not None and trans is not None:
                            # Make paths relative to data_root
                            rel_src = os.path.relpath(src_file, data_root)
                            rel_tgt = os.path.relpath(tgt_file, data_root)
                            
                            src_list.append(rel_src)
                            tgt_list.append(rel_tgt)
                            rot_list.append(rot.astype(np.float32))
                            trans_list.append(trans.astype(np.float32))
                            pairs_added += 1
            
            print(f"  Added {pairs_added} pairs with transformations")
        
        # Strategy 2: If no transformation files, create identity transformations
        # (This is just for testing - not realistic for evaluation)
        if len(trans_files) == 0:
            print(f"  No transformation files found, creating identity pairs")
            pairs_added = 0
            for i in range(min(len(pt_files), 10)):  # Limit to first 10 files
                for j in range(i+1, min(len(pt_files), i+6)):  # Max 5 pairs per source
                    if pairs_added >= max_pairs_per_scene:
                        break
                    
                    src_file = pt_files[i]
                    tgt_file = pt_files[j]
                    
                    # Create identity transformation (warning: this is not realistic!)
                    rot = np.eye(3, dtype=np.float32)
                    trans = np.zeros(3, dtype=np.float32)
                    
                    rel_src = os.path.relpath(src_file, data_root)
                    rel_tgt = os.path.relpath(tgt_file, data_root)
                    
                    src_list.append(rel_src)
                    tgt_list.append(rel_tgt)
                    rot_list.append(rot)
                    trans_list.append(trans)
                    pairs_added += 1
            
            print(f"  Added {pairs_added} identity pairs (WARNING: Not realistic for evaluation!)")
    
    return src_list, tgt_list, rot_list, trans_list

def create_3dmatch_split(data_root, output_path="test.pkl"):
    """Create a 3DMatch test split file"""
    
    print(f"Creating 3DMatch test split from: {data_root}")
    print(f"Output file: {output_path}")
    
    # Create test pairs
    src_list, tgt_list, rot_list, trans_list = create_test_pairs_from_scenes(data_root)
    
    if len(src_list) == 0:
        print("ERROR: No valid pairs found!")
        print("\nPlease check:")
        print("1. Are there .pth files in your scene directories?")
        print("2. Are there transformation files (.txt, .log, .trans)?")
        print("3. Is the directory structure correct?")
        return False
    
    # Create the split dictionary
    split_data = {
        'src': src_list,
        'tgt': tgt_list,
        'rot': rot_list,
        'trans': trans_list
    }
    
    print(f"\nCreated split with {len(src_list)} pairs")
    print(f"Sample pairs:")
    for i in range(min(3, len(src_list))):
        print(f"  {src_list[i]} -> {tgt_list[i]}")
    
    # Save to pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(split_data, f)
    
    print(f"\nSplit saved to: {output_path}")
    
    # Verify the saved file
    with open(output_path, 'rb') as f:
        loaded_data = pickle.load(f)
    
    print(f"Verification: Loaded {len(loaded_data['src'])} pairs")
    
    return True

def inspect_scene_structure(data_root):
    """Inspect the structure of your 3DMatch data to understand the format"""
    
    print("=== INSPECTING 3DMATCH DATA STRUCTURE ===")
    print(f"Data root: {data_root}")
    
    scene_dirs = []
    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path):
            scene_dirs.append(item_path)
    
    print(f"\nFound {len(scene_dirs)} scene directories:")
    
    for i, scene_dir in enumerate(scene_dirs[:3]):  # Show first 3 scenes
        scene_name = os.path.basename(scene_dir)
        print(f"\n--- Scene {i+1}: {scene_name} ---")
        
        # Count different file types
        pth_files = glob.glob(os.path.join(scene_dir, "**/*.pth"), recursive=True)
        txt_files = glob.glob(os.path.join(scene_dir, "**/*.txt"), recursive=True)
        log_files = glob.glob(os.path.join(scene_dir, "**/*.log"), recursive=True)
        
        print(f"  .pth files: {len(pth_files)}")
        print(f"  .txt files: {len(txt_files)}")  
        print(f"  .log files: {len(log_files)}")
        
        # Show sample files
        if pth_files:
            print(f"  Sample .pth files:")
            for pth_file in pth_files[:3]:
                rel_path = os.path.relpath(pth_file, scene_dir)
                print(f"    {rel_path}")
        
        if txt_files:
            print(f"  Sample .txt files:")
            for txt_file in txt_files[:3]:
                rel_path = os.path.relpath(txt_file, scene_dir)
                print(f"    {rel_path}")
    
    if len(scene_dirs) > 3:
        print(f"\n... and {len(scene_dirs) - 3} more scenes")
    
    print("\n" + "="*50)

def main():
    data_root = "/workspace/icp-3dmatch/test"
    
    # First, inspect the structure
    if os.path.exists(data_root):
        inspect_scene_structure(data_root)
        
        # Ask user if they want to proceed
        response = input("\nDo you want to create the test split file? (y/n): ")
        if response.lower() == 'y':
            success = create_3dmatch_split(data_root, "test.pkl")
            if success:
                print("\n✓ Success! You can now use 'test.pkl' in your ICP evaluation script.")
            else:
                print("\n✗ Failed to create split file. Please check the data structure.")
    else:
        print(f"Error: Directory {data_root} does not exist!")
        print("Please update the data_root variable with the correct path.")

if __name__ == "__main__":
    main()