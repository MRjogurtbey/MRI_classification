"""
Extract Unique Files from New Dataset
Only copies files that are NOT in current dataset and NOT duplicates within themselves
"""

import os
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict
import json

def compute_file_hash(filepath, chunk_size=8192):
    """Compute MD5 hash of a file"""
    md5 = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def get_dataset_hashes(root_dir, extensions=('.jpg', '.jpeg', '.png', '.h5')):
    """
    Scan directory and compute hashes for all image files
    
    Returns:
        dict: {hash: [list of file paths with that hash]}
    """
    hash_to_files = defaultdict(list)
    file_count = 0
    
    print(f"Scanning: {root_dir}")
    
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith(extensions):
                filepath = os.path.join(root, filename)
                file_hash = compute_file_hash(filepath)
                
                if file_hash:
                    hash_to_files[file_hash].append(filepath)
                    file_count += 1
                    
                    if file_count % 100 == 0:
                        print(f"  Processed {file_count} files...", end='\r')
    
    print(f"  Total: {file_count} files processed")
    return dict(hash_to_files)

def extract_unique_files(current_hashes, new_hashes, output_dir):
    """
    Extract files from new dataset that are:
    1. NOT in current dataset
    2. NOT duplicates within new dataset (take first occurrence)
    
    Preserves Train/Test split structure
    """
    print("\n" + "="*70)
    print("EXTRACTING UNIQUE FILES")
    print("="*70)
    
    # Find unique hashes
    current_hash_set = set(current_hashes.keys())
    
    unique_files = []
    duplicate_count = 0
    
    for file_hash, file_paths in new_hashes.items():
        # Check if hash exists in current dataset
        if file_hash in current_hash_set:
            continue  # Skip - exists in current dataset
        
        # Take only first file if multiple with same hash (internal duplicates)
        if len(file_paths) > 1:
            duplicate_count += len(file_paths) - 1
        
        unique_files.append(file_paths[0])  # Take first occurrence
    
    print(f"\nFound {len(unique_files)} unique files to extract")
    print(f"Skipped {duplicate_count} internal duplicates")
    
    # Copy files preserving structure
    copied_count = 0
    train_count = 0
    test_count = 0
    
    class_counts = defaultdict(lambda: {'train': 0, 'test': 0})
    
    for source_path in unique_files:
        # Parse path to determine Train/Test and class
        source_path_obj = Path(source_path)
        
        # Extract relative path from new dataset root
        # Path structure: .../extracted/Epic and CSCR hospital Dataset/Train|Test/class/filename.jpg
        parts = source_path_obj.parts
        
        # Find Train or Test in path
        split_type = None
        class_name = None
        
        for i, part in enumerate(parts):
            if part in ['Train', 'Test']:
                split_type = part
                if i + 1 < len(parts):
                    class_name = parts[i + 1]
                break
        
        if not split_type or not class_name:
            print(f"Warning: Could not parse path: {source_path}")
            continue
        
        # Create destination path
        # Map Train -> Training, Test -> Testing
        dest_split = 'Training' if split_type == 'Train' else 'Testing'
        dest_path = os.path.join(output_dir, dest_split, class_name, source_path_obj.name)
        
        # Create directory if needed
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Copy file
        try:
            shutil.copy2(source_path, dest_path)
            copied_count += 1
            
            if split_type == 'Train':
                train_count += 1
                class_counts[class_name]['train'] += 1
            else:
                test_count += 1
                class_counts[class_name]['test'] += 1
            
            if copied_count % 100 == 0:
                print(f"  Copied {copied_count}/{len(unique_files)} files...", end='\r')
        except Exception as e:
            print(f"\nError copying {source_path}: {e}")
    
    print(f"\n\nSuccessfully copied {copied_count} unique files!")
    print(f"  - Training: {train_count} files")
    print(f"  - Testing: {test_count} files")
    
    print("\nPer-class breakdown:")
    for class_name, counts in sorted(class_counts.items()):
        print(f"  {class_name:12} - Train: {counts['train']:4}, Test: {counts['test']:4}, Total: {counts['train'] + counts['test']:4}")
    
    return {
        'total_copied': copied_count,
        'train': train_count,
        'test': test_count,
        'class_counts': dict(class_counts)
    }

def main():
    # Paths
    current_dataset_dir = r"C:\Users\ASUS\MRI_classification"
    new_dataset_dir = r"C:\Users\ASUS\Desktop\Hackathon\Brain Tumor MRI Dataset (Glioma, Meningioma, Pitui\extracted"
    output_dir = current_dataset_dir  # Add to current dataset
    
    print("="*70)
    print("UNIQUE FILE EXTRACTION")
    print("="*70)
    
    # Load current dataset hashes
    print("\n[1/3] Loading current dataset hashes...")
    current_training = os.path.join(current_dataset_dir, "Training")
    current_testing = os.path.join(current_dataset_dir, "Testing")
    
    current_hashes = {}
    if os.path.exists(current_training):
        current_hashes.update(get_dataset_hashes(current_training))
    if os.path.exists(current_testing):
        current_hashes.update(get_dataset_hashes(current_testing))
    
    print(f"\nCurrent dataset: {len(current_hashes)} unique hashes")
    
    # Load new dataset hashes
    print("\n[2/3] Loading new dataset hashes...")
    new_hashes = get_dataset_hashes(new_dataset_dir)
    print(f"\nNew dataset: {len(new_hashes)} unique hashes")
    
    # Extract unique files
    print("\n[3/3] Extracting unique files...")
    stats = extract_unique_files(current_hashes, new_hashes, output_dir)
    
    # Save report
    report = {
        'operation': 'extract_unique_files',
        'source': new_dataset_dir,
        'destination': output_dir,
        'current_dataset_size': len(current_hashes),
        'new_dataset_size': len(new_hashes),
        'extracted_files': stats
    }
    
    report_path = os.path.join(output_dir, 'extraction_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nExtraction report saved: {report_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL DATASET SIZE")
    print("="*70)
    
    # Re-scan to get final counts
    print("\nScanning updated dataset...")
    final_training_hashes = get_dataset_hashes(current_training)
    final_testing_hashes = get_dataset_hashes(current_testing)
    
    print(f"\nFINAL Dataset:")
    print(f"  Training: {len(final_training_hashes)} unique files")
    print(f"  Testing:  {len(final_testing_hashes)} unique files")
    print(f"  Total:    {len(final_training_hashes) + len(final_testing_hashes)} unique files")
    
    print(f"\nGrowth:")
    original_total = len(current_hashes)
    final_total = len(final_training_hashes) + len(final_testing_hashes)
    growth = final_total - original_total
    growth_pct = (growth / original_total) * 100
    
    print(f"  Original: {original_total} files")
    print(f"  Added:    +{growth} files (+{growth_pct:.1f}%)")
    print(f"  Final:    {final_total} files")
    
    print("\n" + "="*70)
    print("Ready for re-training with expanded dataset!")
    print("="*70)

if __name__ == "__main__":
    main()
