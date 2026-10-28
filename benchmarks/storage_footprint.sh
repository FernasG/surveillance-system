#!/usr/bin/env bash
# Metric 4 - disk footprint after X hours of recording.
#
# Usage:
#   benchmarks/storage_footprint.sh --hours 3 [--root <repo>] [--out results/storage.csv]
#
# Sums separately: recorded .mp4 files, ChromaDB persistent volume, SQLite database
# (including WAL/SHM files). Model weights are reported as static footprint, outside the total.

set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
hours=""
out=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hours) hours="$2"; shift 2 ;;
        --root) root="$(cd "$2" && pwd)"; shift 2 ;;
        --out) out="$2"; shift 2 ;;
        *) sed -n '2,9p' "$0"; exit 0 ;;
    esac
done

out="${out:-$root/results/storage.csv}"

human() { numfmt --to=iec --suffix=B --format="%.2f" "$1"; }

dir_bytes() { [[ -d "$1" ]] && du -sb "$1" | cut -f1 || echo 0; }

# Videos: apparent size of every .mp4 in the watched directory
read -r video_files videos_bytes < <(
    find "$root/videos" -maxdepth 1 -type f -name '*.mp4' -printf '%s\n' 2>/dev/null \
        | awk '{ n++; s += $1 } END { printf "%d %d\n", n, s }'
)

# ChromaDB: host bind mount (./chroma_data -> /data)
chroma_bytes=$(dir_bytes "$root/chroma_data")
chroma_note=""

if [[ "$chroma_bytes" -le 4096 ]] && docker inspect chromadb >/dev/null 2>&1; then
    chroma_bytes=$(docker exec chromadb du -sb /data | cut -f1)
    chroma_note=" (read from inside the container: ./chroma_data is not mounted on /data)"
fi

# SQLite: main file + write-ahead log + shared memory index
sqlite_bytes=0
for f in "$root"/data/guard_data.db{,-wal,-shm}; do
    [[ -f "$f" ]] && sqlite_bytes=$((sqlite_bytes + $(stat -c %s "$f")))
done

total_bytes=$((videos_bytes + chroma_bytes + sqlite_bytes))
models_bytes=$(dir_bytes "$root/llama.cpp/models")

per_hour=""
if [[ -n "$hours" ]]; then
    per_hour=$(awk -v t="$total_bytes" -v h="$hours" 'BEGIN { if (h > 0) printf "%d", t / h }')
fi

printf "=== Storage footprint (%s)%s ===\n" "$root" "${hours:+ after ${hours}h of recording}"
printf "%-22s %12s  (%d files)\n" "Videos (.mp4)" "$(human "$videos_bytes")" "$video_files"
printf "%-22s %12s%s\n" "ChromaDB volume" "$(human "$chroma_bytes")" "$chroma_note"
printf "%-22s %12s\n" "SQLite (db+wal+shm)" "$(human "$sqlite_bytes")"
printf "%-22s %12s\n" "TOTAL" "$(human "$total_bytes")"
[[ -n "$per_hour" ]] && printf "%-22s %12s\n" "Growth per hour" "$(human "$per_hour")"
printf "\n%-22s %12s  (static, not in total)\n" "Models (llama.cpp)" "$(human "$models_bytes")"

mkdir -p "$(dirname "$out")"
[[ -f "$out" ]] || echo "timestamp,hours,video_files,videos_bytes,chroma_bytes,sqlite_bytes,total_bytes,bytes_per_hour,models_bytes" > "$out"
echo "$(date -Iseconds),$hours,$video_files,$videos_bytes,$chroma_bytes,$sqlite_bytes,$total_bytes,$per_hour,$models_bytes" >> "$out"
echo "Row appended to $out"
