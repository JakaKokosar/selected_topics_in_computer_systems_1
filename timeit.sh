#!/bin/bash

jobid=$1

first_array_iteration="${jobid}_1"
last_array_iteration="${jobid}_1000"

# Extract start time
start_times=$(sacct -j $first_array_iteration --format=Start --noheader  | sort | head -1)

# Extract end time
end_times=$(sacct -j $last_array_iteration --format=End --noheader  | sort | head -1)

# Convert the dates to seconds since 1970-01-01 00:00:00 UTC
start_seconds=$(date --date="$start_times" +%s)
end_seconds=$(date --date="$end_times" +%s)

echo "Job submition time: $(date -u -d @$start_seconds +'%T')"
echo "Job end time: $(date -u -d @$end_seconds +'%T')"

# Calculate the difference in seconds
diff_seconds=$((end_seconds - start_seconds))

# Convert the seconds back to a readable format
diff_readable=$(date -u -d @$diff_seconds +'%T')

echo "Duration of a job: $diff_readable"

avg_python_script=$(sacct -j $jobid --format=JobID,JobName,ElapsedRaw | grep python | awk '{sum+=$3; count++} END {if(count>0) print sum/count; else print 0}')

echo "Avg. runtime of python script: $avg_python_script seconds"