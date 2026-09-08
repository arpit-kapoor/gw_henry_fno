#!/bin/bash

# RDS Config
remote_user=${RDSUSER}
remote_host=research-data-ext.sydney.edu.au

# Filepath
filename=grid_scenarios_20x40.tar.gz
remote_path=/rds/${RDSPROJECT}/results/simple_henry/${filename}

# Gadi:
dest_path=${HOME}/Projects/groundwater/results/fno_henry_results/fno_simple_henry_sweep

# Copy folder with sftp
sftp -r ${remote_user}@${remote_host}:${remote_path} ${dest_path}