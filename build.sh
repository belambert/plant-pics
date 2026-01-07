#!/bin/bash
set -e

# Utility script to build and push the Docker image.
# Usage:
#   PROJECT_ID=... IMAGE_NAME=... [IMAGE_TAG=...] ./build.sh


PROJECT_ID="${PROJECT_ID:-llm-exp-405305}"
REGION="${REGION:-us}"
IMAGE_NAME="${IMAGE_NAME:-inat-data}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REPO_NAME="${REPO_NAME:-inat-data}"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Ensure you have:"
echo "  1. Created the Artifact Registry repository:"
echo "     gcloud artifacts repositories create ${REPO_NAME} --repository-format=docker --location=${REGION}"
echo "  2. Configured Docker authentication:"
echo "     gcloud auth configure-docker ${REGION}-docker.pkg.dev"

echo "Building Docker image..."
# Use --platform linux/amd64 for Cloud Batch compatibility if building from ARM (M1/M2 Mac)
docker build --platform linux/amd64 -t "${IMAGE_NAME}:${IMAGE_TAG}" .

echo "Tagging image for Artifact Registry..."
docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${IMAGE_URI}"

echo "Pushing image to Artifact Registry..."
docker push "${IMAGE_URI}"

echo ""
echo "Image successfully pushed to:"
echo "  ${IMAGE_URI}"
echo ""
