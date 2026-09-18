#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-$HOME/OmniXAS_data/figshare_kedge_xanes}"
URL="https://ndownloader.figshare.com/files/9932248"
API_URL="https://api.figshare.com/v2/articles/5678998"
ARCHIVE="xas.json.tgz"
PARTIAL="$ARCHIVE.part"
EXPECTED_MD5="e866677ebb9270aeb2e15c725bef7e05"

for cmd in curl md5sum; do
  command -v "$cmd" >/dev/null || {
    echo "Missing required command: $cmd" >&2
    exit 1
  }
done

mkdir -p "$DEST"
cd "$DEST"

echo "Downloading Figshare article metadata to $DEST/article_5678998.json"
curl -L --fail --retry 5 \
  -o article_5678998.json.tmp \
  "$API_URL"
mv -f article_5678998.json.tmp article_5678998.json

if [[ -f "$ARCHIVE" ]] && echo "$EXPECTED_MD5  $ARCHIVE" | md5sum -c - >/dev/null 2>&1; then
  echo "The verified archive already exists at $DEST/$ARCHIVE"
else
  if [[ -f "$ARCHIVE" ]]; then
    echo "The existing archive failed verification; replacing it safely"
    rm -f "$ARCHIVE"
  fi
  # Start from zero: blindly resuming is unsafe when a server ignores Range
  # and appends a second copy to the partial file.
  rm -f "$PARTIAL"
  echo "Downloading the 5.56 GB archive to $DEST/$PARTIAL"
  if ! curl -L --fail --retry 5 -o "$PARTIAL" "$URL"; then
    echo "Download failed. Remove $DEST/$PARTIAL and run this script again." >&2
    exit 1
  fi
  echo "Checking archive MD5"
  if ! echo "$EXPECTED_MD5  $PARTIAL" | md5sum -c -; then
    echo "Checksum failed. Remove $DEST/$PARTIAL and run this script again." >&2
    exit 1
  fi
  mv -f "$PARTIAL" "$ARCHIVE"
fi

echo "Download complete. Keep the archive compressed."
echo "The Python extractor reads $DEST/$ARCHIVE directly."
