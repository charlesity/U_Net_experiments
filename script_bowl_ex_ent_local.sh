# python active_u_net_science_bowl_nuclei.py -ds ./dataset -qt 3 -re_w 1 -mc_p 0 -early_s 0 -reg 1
#
# python active_u_net_science_bowl_nuclei.py -ds ./dataset -qt 3 -re_w 0 -mc_p 0 -early_s 0 -reg 1
#
# python active_u_net_science_bowl_nuclei.py -ds ./dataset -qt 3 -re_w 0 -mc_p 0 -early_s 1 -reg 1

python active_u_net_science_bowl_nuclei.py -ds ./dataset -qt 3 -re_w 0 -mc_p 1 -early_s 0 -reg 1

python active_u_net_science_bowl_nuclei.py -ds ./dataset -qt 3 -re_w 0 -mc_p 1 -early_s 1 -reg 1

python active_u_net_science_bowl_nuclei.py -ds ./dataset -qt 3 -re_w 1 -mc_p 1 -early_s 1 -reg 1
