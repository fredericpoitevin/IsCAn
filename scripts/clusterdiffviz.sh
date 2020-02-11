#!/bin/bash
#
# visualize difference between cluster centers
#
cat << EOF > .tmp.pml
load $1, ref
load $2, target
alter *, vdw=b**0.5
align target,ref
modevectors ref,target,atom=P,cutoff=2,factor=1,tail=0.5,tailrgb=[0,0,0],head=1.0,headrgb=[0,0,1]
set orthoscopic
bg_color white
set specular, off
#set ribbon_width, 0.5
#set ribbon_trace_atoms
hide nonbonded
#as ribbon, ref
as spheres, ref
util.cbc
orient
zoom center, 150
turn z, 90
ray 1200,1200
png mode_view1.png
turn y, 90 
ray 1200,1200
png mode_view2.png
turn x, 90
ray 1200,1200
png mode_view3.png
spectrum b, selection=ref,minimum=0,maximum=5
spectrum b, selection=target,minimum=0,maximum=5
hide
as spheres, ref
ray 1200,1200
png ref_view3.png
hide spheres
as spheres, target
ray 1200,1200
png target_view3.png
turn x, -90
hide
as spheres, ref
ray 1200,1200
png ref_view2.png
hide spheres
as spheres, target
ray 1200,1200
png target_view2.png
turn y, -90
hide
as spheres, ref
ray 1200,1200
png ref_view1.png
hide spheres
as spheres, target
ray 1200,1200
png target_view1.png
EOF
pymol -cq .tmp.pml > /dev/null 2>&1
for typ in "mode ref target"
do
  montage ${typ}_view1.png ${typ}_view2.png ${typ}_view3.png -geometry 1280x1280 -frame 2 ${typ}_view0.jpg
done
