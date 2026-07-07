#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-$HOME/OmniXAS_data/materialscloud_omnixas_raw}"
EXTRACT="${EXTRACT:-1}"
DOWNLOAD_VASP="${DOWNLOAD_VASP:-0}"  # all-8 FEFF encoder only needs FEFF; set to 1 for VASP too.

for cmd in curl md5sum tar; do
  command -v "$cmd" >/dev/null || { echo "Missing required command: $cmd" >&2; exit 1; }
done

mkdir -p "$DEST"
cd "$DEST"

echo "Downloading OmniXAS raw Materials Cloud data to:"
echo "  $DEST"

curl -L --fail --retry 5 --continue-at - \
  -o README.md \
  "https://archive.materialscloud.org/records/ahy6s-txh07/files/README.md?download=1"

curl -L --fail --retry 5 --continue-at - \
  -o files_description.md \
  "https://archive.materialscloud.org/records/ahy6s-txh07/files/files_description.md?download=1"

curl -L --fail --retry 5 --continue-at - \
  -o FEFF.tar.bz2 \
  "https://archive.materialscloud.org/records/ahy6s-txh07/files/FEFF.tar.bz2?download=1"

echo "Checking FEFF MD5..."
echo "e7841a5642bd880080b902fe2659c5af  FEFF.tar.bz2" | md5sum -c -

if [[ "$DOWNLOAD_VASP" == "1" ]]; then
  curl -L --fail --retry 5 --continue-at - \
    -o VASP.tar.bz2 \
    "https://archive.materialscloud.org/records/ahy6s-txh07/files/VASP.tar.bz2?download=1"
  echo "Checking VASP MD5..."
  echo "ce6f99e2cf9bca599d0e3dcd5b07e4fe  VASP.tar.bz2" | md5sum -c -
fi

if [[ "$EXTRACT" == "1" ]]; then
  mkdir -p extracted
  echo "Extracting FEFF..."
  tar --warning=no-unknown-keyword -xjf FEFF.tar.bz2 -C extracted
  if [[ "$DOWNLOAD_VASP" == "1" ]]; then
    echo "Extracting VASP..."
    tar --warning=no-unknown-keyword -xjf VASP.tar.bz2 -C extracted
  fi
else
  echo "Skipping extraction because EXTRACT=$EXTRACT"
fi

echo "Done. For OmniXAS scripts, set:"
echo "  export OMNIXAS_DATA_ROOT=\"$(dirname "$DEST")\""
