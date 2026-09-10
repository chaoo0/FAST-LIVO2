#!/usr/bin/env bash
set -eo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 BAG_DIRECTORY SEQUENCE_NAME OUTPUT_DIRECTORY" >&2
  exit 2
fi

bag_directory=$1
sequence_name=$2
output_directory=$3
workspace_root=/home/liu/fast_livo2
source_directory=$workspace_root/src/FAST-LIVO2
trajectory_source=$source_directory/Log/result/$sequence_name.txt

if [[ ! -d $bag_directory ]]; then
  echo "bag directory does not exist: $bag_directory" >&2
  exit 2
fi
if [[ -e $trajectory_source ]]; then
  echo "refusing to overwrite existing trajectory: $trajectory_source" >&2
  exit 2
fi
if [[ -e $output_directory ]]; then
  echo "refusing to overwrite existing output directory: $output_directory" >&2
  exit 2
fi

mkdir -p "$output_directory" "$output_directory/ros_logs"
export ROS_LOG_DIR=$output_directory/ros_logs
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-73}

source /opt/ros/humble/setup.bash
source "$workspace_root/install/setup.bash"
set -u

launch_log=$output_directory/fast_livo.log
play_log=$output_directory/bag_play.log
start_epoch_ns=$(date +%s%N)

setsid ros2 launch fast_livo mapping_m3dgr_mid360_lio.launch.py \
  sequence_name:="$sequence_name" use_rviz:=false >"$launch_log" 2>&1 &
launch_pid=$!

cleanup() {
  if kill -0 "$launch_pid" 2>/dev/null; then
    kill -INT -- "-$launch_pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! kill -0 "$launch_pid" 2>/dev/null; then
        break
      fi
      sleep 0.25
    done
  fi
  if kill -0 "$launch_pid" 2>/dev/null; then
    kill -TERM -- "-$launch_pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! kill -0 "$launch_pid" 2>/dev/null; then
        break
      fi
      sleep 0.25
    done
  fi
  if kill -0 "$launch_pid" 2>/dev/null; then
    kill -KILL -- "-$launch_pid" 2>/dev/null || true
  fi
  wait "$launch_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 4
if ! kill -0 "$launch_pid" 2>/dev/null; then
  echo "FAST-LIVO2 exited before playback; inspect $launch_log" >&2
  exit 1
fi

ros2 bag play "$bag_directory" --rate 1.0 --disable-keyboard-controls --topics \
  /livox/mid360/lidar /livox/mid360/imu >"$play_log" 2>&1

# Allow any bounded DDS backlog to drain. Stop once the output count is stable
# for three checks; the final timestamp is verified separately by the report.
stable_checks=0
previous_lines=-1
for _ in $(seq 1 30); do
  current_lines=0
  if [[ -f $trajectory_source ]]; then
    current_lines=$(wc -l <"$trajectory_source")
  fi
  if [[ $current_lines -eq $previous_lines ]]; then
    stable_checks=$((stable_checks + 1))
  else
    stable_checks=0
  fi
  previous_lines=$current_lines
  if [[ $stable_checks -ge 3 ]]; then
    break
  fi
  sleep 1
done

end_epoch_ns=$(date +%s%N)
cleanup
trap - EXIT INT TERM

if [[ ! -s $trajectory_source ]]; then
  echo "FAST-LIVO2 produced no trajectory: $trajectory_source" >&2
  exit 1
fi

cp "$trajectory_source" "$output_directory/trajectory.tum"
{
  echo "schema_version=1"
  echo "bag_directory=$bag_directory"
  echo "sequence_name=$sequence_name"
  echo "ros_domain_id=$ROS_DOMAIN_ID"
  echo "start_epoch_ns=$start_epoch_ns"
  echo "end_epoch_ns=$end_epoch_ns"
  echo "wall_duration_s=$(( (end_epoch_ns - start_epoch_ns) / 1000000000 ))"
  echo "trajectory_count=$(wc -l <"$trajectory_source")"
  echo "trajectory_first_timestamp=$(awk 'NR==1 {print $1}' "$trajectory_source")"
  echo "trajectory_last_timestamp=$(awk 'END {print $1}' "$trajectory_source")"
} >"$output_directory/run_summary.txt"

echo "completed $sequence_name; results: $output_directory"
