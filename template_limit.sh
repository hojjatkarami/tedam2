#!/bin/bash

waitforjobs() {
    while test $(jobs -p | wc -w) -ge "$1"; do wait -n; done
}
N_JOBS=2



waitforjobs $N_JOBS
( (echo begin 1) && (sleep 10) && (echo end 1)) &

waitforjobs $N_JOBS
( (echo begin 2) && (sleep 12) && (echo end 2)) &


waitforjobs $N_JOBS
( (echo begin 3) && (sleep 14) && (echo end 3)) &

waitforjobs $N_JOBS
( (echo begin 4) && (sleep 16) && (echo end 4)) &

waitforjobs $N_JOBS
( (echo begin 5) && (sleep 10) && (echo end 5)) &



