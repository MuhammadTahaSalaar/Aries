#!/bin/bash
###############################################################################
# ARIES — Sync repository to Hydra HPC
#
# This script creates the target directory on Hydra (if it does not exist)
# and synchronises the local repository, excluding heavy data/env folders.
#
# Usage:
#   chmod +x sync_to_hydra.sh
#   ./sync_to_hydra.sh
###############################################################################

# Variables
LOCAL_DIR="/home/taha-salaar/data/FYP/Aries/"
HYDRA_REMOTE="hydra"
HYDRA_TARGET="/data/brussel/vo/000/bvo00010/vsc11249/Aries_SOAR/"

echo "=============================================="
echo "Starting synchronization to Hydra HPC"
echo "Local:  ${LOCAL_DIR}"
echo "Remote: ${HYDRA_REMOTE}:${HYDRA_TARGET}"
echo "=============================================="

# 1. Create the target directory on Hydra
echo "[1/2] Ensuring target directory exists on Hydra..."
ssh "${HYDRA_REMOTE}" "mkdir -p ${HYDRA_TARGET}"

# 2. Rsync files using the exclusion rules
echo "[2/2] Synchronising files..."
rsync -avz --progress --exclude-from="${LOCAL_DIR}.rsync_exclude" "${LOCAL_DIR}" "${HYDRA_REMOTE}:${HYDRA_TARGET}"

echo "=============================================="
echo "Synchronization complete!"
echo "=============================================="
