#!/usr/bin/env bash
# Metric 3 - RAM and CPU of every running container, sampled together every N seconds.
#
# Usage:
#   nohup benchmarks/monitor_resources.sh -i 2 -o results/resources.csv >/dev/null 2>&1 &
#   ...run the stress test...
#   kill "$(cat results/monitor.pid)"     # prints/saves peak usage per container
#
# Output:
#   <out>              timestamp_iso,epoch,container,cpu_percent,mem_used_mib,mem_limit_mib,mem_percent
#   <out>_host.csv     timestamp_iso,epoch,host_mem_used_mib,host_mem_total_mib,load1,cpu_temp_c
#   <out>_peaks.txt    peak summary written on exit
#
# cpu_percent follows `docker stats`: 100% = one core (Raspberry Pi 5 max = 400%).

set -euo pipefail

interval=2
out="results/resources.csv"

while getopts "i:o:h" opt; do
    case "$opt" in
        i) interval="$OPTARG" ;;
        o) out="$OPTARG" ;;
        *) sed -n '2,15p' "$0"; exit 0 ;;
    esac
done

mkdir -p "$(dirname "$out")"
host_out="${out%.csv}_host.csv"
peaks_out="${out%.csv}_peaks.txt"
pid_file="$(dirname "$out")/monitor.pid"

echo "timestamp_iso,epoch,container,cpu_percent,mem_used_mib,mem_limit_mib,mem_percent" > "$out"
echo "timestamp_iso,epoch,host_mem_used_mib,host_mem_total_mib,load1,cpu_temp_c" > "$host_out"
echo $$ > "$pid_file"

cpu_temp() {
    if command -v vcgencmd >/dev/null 2>&1; then
        vcgencmd measure_temp | sed -E "s/temp=([0-9.]+).*/\1/"
    elif [[ -r /sys/class/thermal/thermal_zone0/temp ]]; then
        awk '{ printf "%.1f", $1 / 1000 }' /sys/class/thermal/thermal_zone0/temp
    fi
}

summarize() {
    {
        echo "=== Peak usage per container ($out) ==="
        awk -F, 'NR > 1 {
            if ($5 > mem[$3]) mem[$3] = $5
            if ($4 > cpu[$3]) cpu[$3] = $4
            total[$2] += $5
        }
        END {
            printf "%-28s %14s %12s\n", "container", "peak_mem_mib", "peak_cpu_%"
            for (c in mem) printf "%-28s %14.1f %12.2f\n", c, mem[c], cpu[c]
            for (t in total) if (total[t] > peak) { peak = total[t]; peak_t = t }
            printf "\nPeak sum of container RAM: %.1f MiB (epoch %s)\n", peak, peak_t
        }' "$out"
        awk -F, 'NR > 1 { if ($3 > m) m = $3; if ($6 > t) t = $6 }
            END { printf "Peak host RAM used: %.0f MiB | Peak CPU temp: %s C\n", m, t }' "$host_out"
    } | tee "$peaks_out"
    rm -f "$pid_file"
}

trap 'summarize; exit 0' INT TERM

echo "Sampling every ${interval}s into $out (PID $$). Stop with: kill \$(cat $pid_file)"

start=$(date +%s.%N)
k=0

while true; do
    ts_iso=$(date -Iseconds)
    epoch=$(date +%s.%N)

    docker stats --no-stream --format '{{.Name}};{{.CPUPerc}};{{.MemUsage}};{{.MemPerc}}' \
        | awk -F';' -v ts="$ts_iso" -v ep="$epoch" '
            function to_mib(v,   n, u) {
                n = v; gsub(/[^0-9.]/, "", n)
                u = v; gsub(/[0-9. ]/, "", u)
                if (u == "B")   return n / 1048576
                if (u == "KiB" || u == "kB") return n / 1024
                if (u == "GiB" || u == "GB") return n * 1024
                if (u == "TiB") return n * 1048576
                return n
            }
            {
                split($3, mem, "/")
                cpu = $2; gsub(/%/, "", cpu)
                pct = $4; gsub(/%/, "", pct)
                printf "%s,%s,%s,%s,%.2f,%.2f,%s\n", ts, ep, $1, cpu, to_mib(mem[1]), to_mib(mem[2]), pct
            }' >> "$out"

    read -r host_used host_total < <(free -m | awk '/^Mem:/ { print $3, $2 }')
    load1=$(cut -d' ' -f1 /proc/loadavg)
    echo "$ts_iso,$epoch,$host_used,$host_total,$load1,$(cpu_temp)" >> "$host_out"

    # Fixed schedule: `docker stats` itself takes ~1-2s, so sleep only what is left.
    k=$((k + 1))
    sleep_for=$(awk -v s="$start" -v k="$k" -v i="$interval" -v now="$(date +%s.%N)" \
        'BEGIN { d = s + k * i - now; print (d > 0 ? d : 0) }')
    sleep "$sleep_for" &
    wait $!
done
