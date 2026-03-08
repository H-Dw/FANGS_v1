#!/usr/bin/env bash
set -euo pipefail

# Usage: 
#   ./run_boltz.sh <input_path> [out_dir] [cuda_device] [recycling_steps] [diffusion_samples]
# 示例（使用所有默认值 recycling_steps=10, diffusion_samples=25）：
#   ./run_boltz.sh input/
# 指定所有参数：
#   ./run_boltz.sh input/ ./out/ 0 10 25 (the default parameters for AlphaFold3)

if [ $# -lt 1 ]; then
  echo "Usage: $0 <input_path> [out_dir] [cuda_device] [recycling_steps] [diffusion_samples]" >&2
  exit 1
fi

# 必需参数
input_path="$1"

# 可选参数及其默认值
out_dir="${2:-./boltz_output/}"
cuda_device="${3:-3}"
recycling_steps="${4:-3}"
diffusion_samples="${5:-1}"

# 创建输出目录
mkdir -p "$out_dir"

# 指定 GPU
export CUDA_VISIBLE_DEVICES="$cuda_device"

# 记录开始时间（Unix 时间戳，秒）
start_ts=$(date +%s)

# 主命令（始终在前台执行）
boltz predict "$input_path" \
    --out_dir "$out_dir" \
    --cache "/s0/dhuang/boltz/cache/" \
    --use_msa_server \
    --output_format pdb \
    --devices 1 \
    --recycling_steps "$recycling_steps" \
    --diffusion_samples "$diffusion_samples" \
  > grafting_boltz.log 2>&1

# 记录结束时间
end_ts=$(date +%s)

# 计算耗时并打印
elapsed=$((end_ts - start_ts))
hours=$((elapsed/3600))
mins=$(((elapsed%3600)/60))
secs=$((elapsed%60))
printf "Done in %d:%02d:%02d (hh:mm:ss)\n" "$hours" "$mins" "$secs"

echo "Logs: grafting_boltz.log"
echo "Results: $out_dir"
