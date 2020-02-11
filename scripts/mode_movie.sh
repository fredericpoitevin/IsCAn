#!/bin/bash
#
if [ $# -ne 1 ]; then 
  echo "please provide name of PDB traj file, and name of a direcetory where to work"; exit; fi
jobdir=".tmp"
if [ -d $jobdir ]; then echo "abort, .tmp exists";exit; fi
mkdir $jobdir
#
ilist="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 "
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
#set ribbon_width, 0.5
#set ribbon_trace_atoms
hide nonbonded
#as ribbon
as spheres
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
  echo "show spheres, traj_$n" >> $pml
  #echo "show ribbon, traj_$n" >> $pml
  echo "ray 1200,1200" >> $pml
  echo "png ${jobdir}/movie1_${n}.png" >> $pml
done
echo "turn y, 90" >> $pml
for i in $ilist
do
  echo "hide" >> $pml
  if [ $i -lt 10 ]; then
    n="000"$i
  else
    n="00"$i
  fi
  echo "show spheres, traj_$n" >> $pml
  #echo "show ribbon, traj_$n" >> $pml
  echo "ray 1200,1200" >> $pml
  echo "png ${jobdir}/movie2_${n}.png" >> $pml
done
echo "turn x, 90" >> $pml
for i in $ilist
do
  echo "hide" >> $pml
  if [ $i -lt 10 ]; then
    n="000"$i
  else
    n="00"$i
  fi
  echo "show spheres, traj_$n" >> $pml
  #echo "show ribbon, traj_$n" >> $pml
  echo "ray 1200,1200" >> $pml
  echo "png ${jobdir}/movie3_${n}.png" >> $pml
done
pymol -cq .tmp.pml > /dev/null 2>&1
#
cd $jobdir
for i in $ilist
do
  if [ $i -lt 10 ]; then
    n="000"$i
  else
    n="00"$i
  fi
  montage movie1_${n}.png movie2_${n}.png movie3_${n}.png -geometry 1280x1280 -frame 2 movie0_${n}.jpg
done
convert -loop 0 -delay 1 movie0_*.jpg ../movie.gif > /dev/null 2>&1
ffmpeg -i movie0_%04d.jpg -vf scale="trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -profile:v high -pix_fmt yuv420p -g 30 -r 30 ../movie.mp4 > /dev/null 2>&1
for view in 1 2 3
do
  convert -loop 0 -delay 1 movie${view}_*.png ../movie${view}.gif > /dev/null 2>&1
  ffmpeg -i movie${view}_%04d.png -vf scale="trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -profile:v high -pix_fmt yuv420p -g 30 -r 30 ../movie${view}.mp4 > /dev/null 2>&1
done
cd ../
rm -rf "$jobdir"
