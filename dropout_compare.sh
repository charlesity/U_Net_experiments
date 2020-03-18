read -p "Enter number of experiments: " num_exp
read -p "Enter dropout type (0: No dropout, 1: standard_dropout, 2: Max Pool dropout, 3: Both Max-Pool + Standard " dropout_type
read -p "Enter Log Filename " log_filename
read -p "Enter Dataset Location " dataset_location
read -p "Number of Epochs " epochs
read -p "Dropout probability " drop_prob



for ((i=1; i<=$num_exp; i++))
do
  python dropout_comparisons_for_bash.py -ds $dataset_location -dt $dropout_type -exp_index $i -lf $log_filename
done
