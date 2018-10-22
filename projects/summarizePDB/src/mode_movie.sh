#!/bin/bash
#
if [ $# -ne 2 ]; then 
  echo "please provide name of PDB traj file, and name of a direcetory where to work"; exit; fi
jobdir=$2
if [ -d $jobdir ]; then echo "abort, jobdir exitst";exit; fi
mkdir $jobdir
#
ilist="1 2 3 4 5 6 7 8 9 10 "
pml=".tmp.pml"
echo "load $1, traj" > $pml
for i in $ilist
do
  echo "split_states traj, $i, $i" >> $pml
done
cat << EOF >> $pml
set orthoscopic
bg_color white
set specular, off
set ribbon_width, 0.5
set ribbon_trace_atoms
hide nonbonded
as ribbon
util.cbc
orient
zoom center, 150
turn z, 90
EOF
for i in $ilist
do
  echo "hide" >> $pml
  if [ $i -lt 10 ]; then
    n="000"$i
  else
    n="00"$i
  fi
  echo "show ribbon, traj_$n" >> $pml
  echo "ray 1200,1200" >> $pml
  echo "png ${jobdir}/movie_${n}.png" >> $pml
done
#ray 1200,1200
#png view1.png
#turn y, 90 
#ray 1200,1200
#png view2.png
#turn x, 90
#ray 1200,1200
#png view3.png
#EOF
pymol -cq .tmp.pml
cd $jobdir
convert -loop 0 -delay 1 movie_*.png out.gif
#echo "open view*.png"
