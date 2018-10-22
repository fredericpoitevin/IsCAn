#!/bin/bash
#
# visualize modes
#
cat << EOF > .tmp.pml
load $1, traj
split_states traj, 1,1
split_states traj, 5,5
modevectors traj_0001,traj_0005,atom=P,cutoff=2,factor=1,tail=0.5,tailrgb=[0,0,0],head=1.0,headrgb=[0,0,1]
set orthoscopic
bg_color white
set specular, off
set ribbon_width, 0.5
set ribbon_trace_atoms
hide nonbonded
as ribbon, traj_0001
util.cbc
orient
zoom center, 150
turn z, 90
ray 1200,1200
png view1.png
turn y, 90 
ray 1200,1200
png view2.png
turn x, 90
ray 1200,1200
png view3.png
EOF
pymol -cq .tmp.pml
echo "open view*.png"
