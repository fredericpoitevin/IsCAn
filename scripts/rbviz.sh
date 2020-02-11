#!/bin/bash
#
# visualize rigid bodies in pymol
#
cat << EOF > .tmp.pml
load $1, traj
create frame0, traj, 1, 0
bg_color white
as spheres
set sphere_scale,0.5
set specular, off
spectrum b
set sphere_scale,1,frame0
EOF
pymol .tmp.pml
