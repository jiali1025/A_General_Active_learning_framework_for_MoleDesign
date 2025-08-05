#!/usr/bin/env bash
# ================================================================
# Google-Drive model downloader for Unified Active Learning Framework
# ---------------------------------------------------------------
# Requires:   gdown >= 4.6 ( pip install gdown )
# Usage:
#   bash scripts/download_models.sh all               # download every model archive
#   bash scripts/download_models.sh pure_uncertainty  # download specific strategy
#   bash scripts/download_models.sh mx_synth_v2_u174   # download single archive by ID
# ---------------------------------------------------------------
# Google-Drive folder (public):
#   https://drive.google.com/drive/folders/1LG8ly9VcvSZcGj9lQ6Z-wVzkhSarclat
# Each *.tar.gz archive inside the folder corresponds to one strategy variant.
#   naming rule: <strategy_name>.tar.gz , e.g.  pure_uncertainty.tar.gz
# ================================================================
set -e

# base folder id
FOLDER_ID="1LG8ly9VcvSZcGj9lQ6Z-wVzkhSarclat"

need_gdown() {
  if ! command -v gdown >/dev/null 2>&1; then
    echo "[ERROR] gdown not found. Install via: pip install gdown" >&2
    exit 1
  fi
}

# download_from_drive <file_name>
download_from_drive() {
  local fname="$1"
  echo "[INFO] downloading $fname ..."
  gdown --quiet --fuzzy "https://drive.google.com/uc?id=$FOLDER_ID" --search "$fname" --output "$fname"
}

extract_tar() {
  local fname="$1"
  local dest="$2"
  mkdir -p "$dest"
  tar -xzf "$fname" -C "$dest"
}

main() {
  need_gdown
  mkdir -p models

  case "$1" in
    "all"|"")
      echo "[INFO] downloading ALL model archives from Google Drive folder …"
      filelist=$(gdown --quiet --fuzzy "https://drive.google.com/uc?id=$FOLDER_ID" --list | awk '{print $NF}' | grep ".tar.gz$")
      for f in $filelist; do
        download_from_drive "$f"
        extract_tar "$f" models/
      done
      ;;
    "pure_uncertainty"|"target_property"|"expected_improvement"|"threshold_test"|"retro"|"mix_version")
      download_from_drive "$1.tar.gz"
      extract_tar "$1.tar.gz" "models/$1" ;;
    *)  # assume direct file name argument
      download_from_drive "$1"
      extract_tar "$1" models/ ;;
  esac
  echo "[DONE] models ready in ./models directory."
}

main "$@"
