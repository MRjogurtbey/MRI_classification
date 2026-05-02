"""
Duplicate Dataset Detection Script
Compares two datasets by file hash to detect duplicates
"""

import os
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
    
    print(f"\nScanning: {root_dir}")
    
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

def find_duplicates_between_datasets(dataset1_hashes, dataset2_hashes):
    """
    Find common hashes between two datasets
    
    Returns:
        dict: {hash: {'dataset1': [paths], 'dataset2': [paths]}}
    """
    common_hashes = set(dataset1_hashes.keys()) & set(dataset2_hashes.keys())
    
    duplicates = {}
    for hash_val in common_hashes:
        duplicates[hash_val] = {
            'dataset1': dataset1_hashes[hash_val],
            'dataset2': dataset2_hashes[hash_val]
        }
    
    return duplicates

def find_duplicates_within_dataset(dataset_hashes):
    """Find duplicate files within a single dataset"""
    duplicates = {h: files for h, files in dataset_hashes.items() if len(files) > 1}
    return duplicates

def main():
    # Paths
    current_dataset_dir = r"C:\Users\ASUS\MRI_classification"
    new_dataset_dir = r"C:\Users\ASUS\Desktop\Hackathon\Brain Tumor MRI Dataset (Glioma, Meningioma, Pitui\extracted"
    
    # Check if paths exist
    if not os.path.exists(current_dataset_dir):
        print(f"ERROR: Current dataset not found: {current_dataset_dir}")
        return
    
    if not os.path.exists(new_dataset_dir):
        print(f"WARNING: New dataset not yet extracted: {new_dataset_dir}")
        print("Please wait for extraction to complete.")
        return
    
    print("="*70)
    print("DUPLICATE DATASET DETECTION")
    print("="*70)
    
    # Scan current dataset (Training + Testing)
    print("\n[1/3] Scanning CURRENT dataset...")
    current_training = os.path.join(current_dataset_dir, "Training")
    current_testing = os.path.join(current_dataset_dir, "Testing")
    
    current_hashes = {}
    if os.path.exists(current_training):
        current_hashes.update(get_dataset_hashes(current_training))
    if os.path.exists(current_testing):
        current_hashes.update(get_dataset_hashes(current_testing))
    
    print(f"\nCurrent dataset: {len(current_hashes)} unique files")
    
    # Scan new dataset
    print("\n[2/3] Scanning NEW dataset...")
    new_hashes = get_dataset_hashes(new_dataset_dir)
    print(f"\nNew dataset: {len(new_hashes)} unique files")
    
    # Find duplicates between datasets
    print("\n[3/3] Comparing datasets...")
    duplicates_between = find_duplicates_between_datasets(current_hashes, new_hashes)
    
    # Find duplicates within new dataset
    duplicates_within_new = find_duplicates_within_dataset(new_hashes)
    
    # Report
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"\nCurrent Dataset:")
    print(f"  - Total files: {sum(len(files) for files in current_hashes.values())}")
    print(f"  - Unique hashes: {len(current_hashes)}")
    
    print(f"\nNew Dataset:")
    print(f"  - Total files: {sum(len(files) for files in new_hashes.values())}")
    print(f"  - Unique hashes: {len(new_hashes)}")
    
    print(f"\nDuplicates BETWEEN datasets:")
    print(f"  - Common hashes: {len(duplicates_between)}")
    print(f"  - Duplicate files: {sum(len(d['dataset1']) + len(d['dataset2']) for d in duplicates_between.values())}")
    
    if duplicates_between:
        print(f"\n  WARNING: {len(duplicates_between)} files are IDENTICAL between datasets!")
        print(f"  -> New dataset contains {sum(len(d['dataset2']) for d in duplicates_between.values())} duplicate images")
        print(f"  -> These should NOT be added to training/testing")
        
        # Show first 5 examples
        print("\n  Examples (first 5):")
        for i, (hash_val, paths) in enumerate(list(duplicates_between.items())[:5]):
            print(f"\n  [{i+1}] Hash: {hash_val[:16]}...")
            print(f"      Current: {paths['dataset1'][0]}")
            print(f"      New:     {paths['dataset2'][0]}")
    else:
        print(f"\n  OK: No duplicates found between datasets!")
        print(f"  -> All {len(new_hashes)} files in new dataset are unique")
        print(f"  -> Safe to add to training data")
    
    print(f"\nDuplicates WITHIN new dataset:")
    print(f"  - Duplicate hashes: {len(duplicates_within_new)}")
    
    if duplicates_within_new:
        print(f"  WARNING: New dataset has {len(duplicates_within_new)} duplicate files within itself!")
        print(f"  -> Should remove duplicates before using")
    else:
        print(f"  OK: No internal duplicates in new dataset")
    
    # Save detailed report
    report = {
        'current_dataset': {
            'path': current_dataset_dir,
            'total_files': sum(len(files) for files in current_hashes.values()),
            'unique_hashes': len(current_hashes)
        },
        'new_dataset': {
            'path': new_dataset_dir,
            'total_files': sum(len(files) for files in new_hashes.values()),
            'unique_hashes': len(new_hashes)
        },
        'duplicates_between': {
            'count': len(duplicates_between),
            'percentage': len(duplicates_between) / len(new_hashes) * 100 if new_hashes else 0
        },
        'duplicates_within_new': {
            'count': len(duplicates_within_new)
        },
        'recommendation': 'DO_NOT_USE' if duplicates_between else 'SAFE_TO_USE'
    }
    
    report_path = os.path.join(current_dataset_dir, 'duplicate_check_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved: {report_path}")
    print("="*70)
    
    # Return status
    return len(duplicates_between) == 0

if __name__ == "__main__":
    is_unique = main()
    exit(0 if is_unique else 1)
