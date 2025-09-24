import csv
import json
import os
import requests

base_url = "https://raw.githubusercontent.com/DDMAL/salami-data-public/master/annotations/"

# File paths
base_dir = os.getcwd()
validated_file = os.path.join(base_dir, 'spectrogram-generation/SALAMI-data/validated-salami.csv')
salami_folder = os.path.join(base_dir,'salami-data-public')
output_dir = os.path.join(base_dir, 'spectrogram-generation/SALAMI-data')
os.makedirs(output_dir, exist_ok=True)

# Get validated SALAMI IDs
validated_ids = set()
with open(validated_file, 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        validated_ids.add(row['SALAMI_ID'])

# Convert to list and take only first 20 for now
validated_ids = sorted([int(salami_id) for salami_id in validated_ids])[:20]

# Process each validated SALAMI_ID
all_data = {}
for salami_id in validated_ids:
    url = base_url + str(salami_id) + "/textfile1.txt"
    response = requests.get(url)
    if response.status_code == 200:
        print(f"Downloaded textfile1 for SALAMI_ID {salami_id}")
        
        # Parse the response text
        annotations = []
        lines = response.text.strip().split('\n')
        
        # Process each line to create start-end segments
        for i in range(len(lines)):
            line = lines[i].strip()
            if not line:
                continue
                
            # Split by tab character
            parts = line.split('\t')
            if len(parts) == 2:
                start_time = parts[0].strip()
                label_text = parts[1].strip()
                
                # Calculate end time (next segment's start time, or assume end of track for last segment)
                if i < len(lines) - 1:
                    next_line = lines[i + 1].strip()
                    if next_line:
                        next_parts = next_line.split('\t')
                        if len(next_parts) == 2:
                            end_time = next_parts[0].strip()
                        else:
                            end_time = start_time  # Fallback
                    else:
                        end_time = start_time  # Fallback
                else:
                    # For the last segment no end time
                    end_time = ""
                
                # Parse section and function labels
                # Assuming format like: "A, a, Intro" or "B, c, Verse, (voice"
                label_parts = [part.strip() for part in label_text.split(',')]
                
                section_label = ""
                function_label = ""
                
                if len(label_parts) >= 3:
                    # First part is usually uppercase letter (A, B, C, etc.)
                    # Second part is usually lowercase letter (a, b, c, etc.)
                    # Third part and beyond is the function label
                    section_label = label_parts[0] if label_parts[0] else ""
                    function_label = ", ".join(label_parts[2:]) if len(label_parts) > 2 else ""
                elif len(label_parts) == 2:
                    section_label = label_parts[0]
                    function_label = label_parts[1]
                elif len(label_parts) == 1:
                    section_label = label_parts[0]
                
                # Clean up function label (ex: removing parentheses)
                function_label = function_label.replace('(', '').replace(')', '').strip()
                
                # Clean full_label by removing ALL parentheses and extra spaces
                full_label = label_text.replace('(', '').replace(')', '').strip()
                # Also clean up any extra commas or spaces that might result
                full_label = ', '.join([part.strip() for part in full_label.split(',') if part.strip()])
                
                annotation = {
                    "start_time": start_time,
                    "end_time": end_time,
                    "section_label": section_label,
                    "function_label": function_label,
                    "full_label": full_label  # Use the cleaned version
                }
                annotations.append(annotation)
        
        # Add to combined data dictionary with salami_id as key
        all_data[salami_id] = {
            "salami_id": salami_id,
            "annotations": annotations
        }
        
        print(f"Processed annotations for SALAMI_ID {salami_id}")
        
    else:
        print(f"Failed to download file for SALAMI_ID {salami_id}. Status code: {response.status_code}")

# Save all data to a single JSON file
combined_output_file = os.path.join(output_dir, 'combined_data.json')
with open(combined_output_file, 'w') as f:
    json.dump(all_data, f, indent=2)

print(f"All data saved to {combined_output_file}")
print(f"Processed {len(all_data)} SALAMI IDs (first 10 only)")