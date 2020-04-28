#!/bin/bash

read -p "Experiment Index from : " exp_from
read -p "Experiment Index to : " to_from
read -p "Enter dropout type (0: Batch Normalization, 1: standard_dropout, 2: Max Pool dropout, 3: Both Max-Pool + Standard " dropout_type
read -p "Enter Log Filename " log_filename
read -p "Enter Dataset Location " dataset_location
read -p "Number of Epochs " epochs
read -p "Dropout probability " drop_prob
read -p "Number of classes " n_classes
read -p "Active Learning Batch Size " act_bat_size
read -p "Active learning query type " query_type
read -p "Re-initiatlize weights during each training " weight_reinit
read -p "Early stopping during training " early_stop
read -p "Image file extension " f_ext
read -p "Active acquisition from " from
read -p "Active acquisition to " to

for ((i=$exp_from; i<=$to_from; i++))
do
  for ((j=$from; j<=$to; j++))
  do
    python active_segmentation.py -ds $dataset_location -dt $dropout_type -exp_index $i -lf $log_filename -ep $epochs -dr_prob $drop_prob -n_classes $n_classes -f_ext $f_ext -ac_batch $act_bat_size -qt $query_type -re_w $weight_reinit -early_s $early_stop -act_ac_ind $j
  done
  del_file="$log_filename$dropout_type$query_type"
  find ./active_dataframe_log/ -name "$del_file*" -exec rm {} \;
done
