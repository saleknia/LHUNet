#!/bin/bash

# source /dss/dssfs04/lwp-dss-0002/pn36fu/pn36fu-dss-0000/reza/AmirSaleknia/miniconda3/etc/profile.d/conda.sh
# conda activate lhunet
# export PYTHONNOUSERSITE=1
export nnUNet_raw="/content/Datasets_nnUNet/nnUNet_raw_data"
export nnUNet_preprocessed="/content/Datasets_nnUNet/nnUNet_preprocessed"
export nnUNet_results="/content/Datasets_nnUNet/nnUNet_results"
export nnUNet_compile=False ### LHU-Net is not compatible with PyTorch Compile Yet

DatasetNumber=801 # choices: 700 (Synapse), 703 (Brats), 708 (LA), 709 (Lung), 800 (NVD Pancreas), 801 (MSD Pancreas)
trainer=lhunetV2MSDPancreasTrainer # choices: lhunetSynapseTrainer, lhunetBratsTrainer, lhunetLATrainer, lhunetLungTrainer, lhunetV2NIHPancreasTrainer

# Training
nnUNetv2_train $DatasetNumber 3d_fullres 0 -tr $trainer
nnUNetv2_train $DatasetNumber 3d_fullres 0 --val --val_best -tr $trainer -p nnUNetPlans

# ============================================
# MEASURE INFERENCE TIME
# ============================================

# Set input and output paths (adjust these paths)
INPUT_FOLDER="/content/Datasets_nnUNet/nnUNet_raw_data/Dataset801_MSD_Pancreas/imagesTs"
OUTPUT_FOLDER="/content/Datasets_nnUNet/nnUNet_results/Dataset801_MSD_Pancreas/inference_timing"

# Create output folder if it doesn't exist
mkdir -p $OUTPUT_FOLDER

# Run inference with timing flag
echo "Measuring inference time..."
nnUNetv2_predict -i $INPUT_FOLDER \
                 -o $OUTPUT_FOLDER \
                 -d $DatasetNumber \
                 -c 3d_fullres \
                 -f 0 \
                 -tr $trainer \
                 -p nnUNetPlans \
                 --timing

# Display the timing results
echo ""
echo "Inference timing results:"
cat $OUTPUT_FOLDER/inference_timing.json

echo ""
echo "Done! Inference timing saved to $OUTPUT_FOLDER/inference_timing.json"