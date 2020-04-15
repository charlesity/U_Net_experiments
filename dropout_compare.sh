#!/bin/bash

read -p "Enter number of experiments: " num_exp
read -p "Enter dropout type (0: Batch Normalization, 1: standard_dropout, 2: Max Pool dropout, 3: Both Max-Pool + Standard " dropout_type
read -p "Enter Log Filename " log_filename
read -p "Enter Dataset Location " dataset_location
read -p "Number of Epochs " epochs
read -p "Dropout probability " drop_prob
read -p "Number of classes " n_classes
read -p "Image file extension " f_ext




for ((i=1; i<=$num_exp; i++))
do
  python dropout_comparisons_for_bash.py -ds $dataset_location -dt $dropout_type -exp_index $i -lf $log_filename -ep $epochs -dr_prob $drop_prob -n_classes $n_classes -f_ext $f_ext
done
