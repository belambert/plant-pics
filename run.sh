#!/bin/bash

# Required Configuration
export PROJECT_ID="llm-exp-405305"
export IMAGE_URI="us-docker.pkg.dev/${PROJECT_ID}/inat-data/inat-data:latest"

# Optional Configuration
export REGION="us-central1"
#export GPU_TYPE="nvidia-l4"      # or nvidia-tesla-a100, nvidia-tesla-t4, etc.
# export GPU_COUNT=1
#export BUCKET="your-gcs-bucket"  # Mounted at /mnt/disks/gcs

# Source the library from the submodule
source "gcp-batch/lib/gcp_batch.sh"

# Submit your command
submit_batch_job $@
