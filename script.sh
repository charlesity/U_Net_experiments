#~/bin/bash

echo "Welcome"
sleep 1
echo "Starting script"

# source ~/charles/environments/keras0/bin/activate

# echo "Running script without oversampling"
# python ./active_deep_seg_max_entropy.py -ds_type 0 -nexp 1

echo "Running script with no dropout"
python u_net_experiment_bowl_2018.py -ds ./dataset -dt 0

echo "Running script with standard dropout"
python u_net_experiment_bowl_2018.py -ds ./dataset -dt 1

echo "Running script with max pool dropout"
python u_net_experiment_bowl_2018.py -ds ./dataset -dt 2

echo "Running script with both standard and maxpool dropout"
python u_net_experiment_bowl_2018.py -ds ./dataset -dt 3
