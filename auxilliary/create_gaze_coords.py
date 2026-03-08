#!/usr/bin/env python3
import csv
import glob
import os

base_dir = "/home/s3758869/vlm_datasets/AriaEA_vlm_ann_3_10_llava-v1.6-34b-hf"
annotation_files = glob.glob(os.path.join(base_dir, "*/annotations.csv"))

print(f"Found {len(annotation_files)} annotation files")

for ann_file in sorted(annotation_files):
    clip_dir = os.path.dirname(ann_file)
    clip_name = os.path.basename(clip_dir)
    output_file = os.path.join(clip_dir, "gaze_coord.csv")
    
    print(f"Processing {clip_name}...")
    
    with open(ann_file, 'r') as infile:
        reader = csv.DictReader(infile)
        
        with open(output_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['frame_index', 'frame_file', 'gaze_x', 'gaze_y'])
            
            for row in reader:
                writer.writerow([
                    row['frame_index'],
                    row['frame_file'],
                    row['gaze_x'],
                    row['gaze_y']
                ])

print(f"\nCreated gaze_coord.csv in {len(annotation_files)} clip directories")