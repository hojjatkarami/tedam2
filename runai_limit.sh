#!/bin/bash


# bgxupdate - update active processes in a group.
#   Works by transferring each process to new group
#   if it is still active.
# in:  bgxgrp - current group of processes.
# out: bgxgrp - new group of processes.
# out: bgxcount - number of processes in new group.

bgxupdate() {
    bgxoldgrp=${bgxgrp}
    bgxgrp=""
    ((bgxcount = 0))
    bgxjobs=" $(jobs -pr | tr '\n' ' ')"
    for bgxpid in ${bgxoldgrp} ; do
        echo "${bgxjobs}" | grep " ${bgxpid} " >/dev/null 2>&1
        if [[ $? -eq 0 ]]; then
            bgxgrp="${bgxgrp} ${bgxpid}"
            ((bgxcount++))
        fi
    done
}

# bgxlimit - start a sub-process with a limit.

#   Loops, calling bgxupdate until there is a free
#   slot to run another sub-process. Then runs it
#   an updates the process group.
# in:  $1     - the limit on processes.
# in:  $2+    - the command to run for new process.
# in:  bgxgrp - the current group of processes.
# out: bgxgrp - new group of processes

bgxlimit() {
    bgxmax=$1; shift
    bgxupdate
    while [[ ${bgxcount} -ge ${bgxmax} ]]; do
        sleep 1
        bgxupdate
    done
    if [[ "$1" != "-" ]]; then
        $* &
        bgxgrp="${bgxgrp} $!"
    fi
}

# Test program, create group and run 6 sleeps with
#   limit of 3.

group1=""
echo 0 $(date | awk '{print $4}') '[' ${group1} ']'
echo
for i in 1 2 3 4 5 6; do
    bgxgrp=${group1}; bgxlimit 3 sleep ${i}0; group1=${bgxgrp}
    echo ${i} $(date | awk '{print $4}') '[' ${group1} ']'
done

# Wait until all others are finished.

echo
bgxgrp=${group1}; bgxupdate; group1=${bgxgrp}
while [[ ${bgxcount} -ne 0 ]]; do
    oldcount=${bgxcount}
    while [[ ${oldcount} -eq ${bgxcount} ]]; do
        sleep 1
        bgxgrp=${group1}; bgxupdate; group1=${bgxgrp}
    done
    echo 9 $(date | awk '{print $4}') '[' ${group1} ']'
done
