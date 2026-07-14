import time
import torch
import numpy as np
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window, predict_sliding_window_return_logits
from nnunetv2.utilities.file_and_folder_operations import join, load_json
import nibabel as nib
import os

# Paths
model_folder = "/content/Datasets_nnUNet/nnUNet_results/Dataset801_MSD_Pancreas"
input_folder = "/content/Datasets_nnUNet/nnUNet_raw_data/Dataset801_MSD_Pancreas/imagesTs"
output_folder = "/content/Datasets_nnUNet/nnUNet_results/Dataset801_MSD_Pancreas/inference_timing"

os.makedirs(output_folder, exist_ok=True)

# Initialize predictor
predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=False,  # Set to False for timing
    perform_everything_on_device=True,
    device=torch.device('cuda', 0),
    verbose=False
)

# Load model
predictor.initialize_from_trained_model_folder(
    model_folder,
    use_folds=(0,),
    checkpoint_name='checkpoint_best.pth',
    configuration='3d_fullres',
    planner_name='nnUNetPlans'
)

# Get list of test images
test_files = [f for f in os.listdir(input_folder) if f.endswith('.nii.gz')]

print(f"Found {len(test_files)} test cases")
print("Measuring inference time per case...")

times = []
case_times = {}

for i, file in enumerate(test_files):
    # Load image
    img_path = join(input_folder, file)
    img = nib.load(img_path)
    data = img.get_fdata()
    data = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0).cuda()
    
    # Warmup for first case only
    if i == 0:
        print("Warming up...")
        for _ in range(3):
            with torch.no_grad():
                _ = predictor.predict_logits_from_list_of_lists([[data]])
        torch.cuda.synchronize()
    
    # Measure
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        result = predictor.predict_logits_from_list_of_lists([[data]])
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    case_times[file] = elapsed
    times.append(elapsed)
    print(f"{file}: {elapsed:.4f} seconds")

# Summary
mean_time = np.mean(times)
std_time = np.std(times)
min_time = np.min(times)
max_time = np.max(times)

print("\n" + "="*50)
print("INFERENCE TIMING SUMMARY")
print("="*50)
print(f"Total cases: {len(times)}")
print(f"Mean time per case: {mean_time:.4f} ± {std_time:.4f} seconds")
print(f"Min time: {min_time:.4f} seconds")
print(f"Max time: {max_time:.4f} seconds")
print(f"Total time: {sum(times):.4f} seconds")

# Save results
import json
results = {
    "num_cases": len(times),
    "mean_time": mean_time,
    "std_time": std_time,
    "min_time": min_time,
    "max_time": max_time,
    "total_time": sum(times),
    "case_times": case_times
}

with open(join(output_folder, "inference_timing.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {join(output_folder, 'inference_timing.json')}")