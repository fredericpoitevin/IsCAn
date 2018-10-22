#!/bin/bash
#
#echo $PWD
if [ $# -ne 2 ]; then echo "please provide name of reference mean file, and relative path to execs"; exit;fi
ref=$1
relpath="$2"
#### visualize components
for file in *oscill*.pdb
do
  echo "> modeviz $file"
  "$relpath"/modeviz.sh $file
  for png in 1 2 3
  do
    mv view${png}.png ${file%.*}"_"view${png}.png
  done
done
#### visualize inter-cluster motions
for file in *_mean.pdb
do
  if [ $file != $ref ]; then
    echo "> $ref $file"
    "$relpath"/clusterdiffviz.sh $ref $file
  for png in 1 2 3
  do
    mv view${png}.png ${ref%.*}"_to_"${file%.*}"_"view${png}.png
    mv ref_view${png}.png ${ref%.*}"_"view${png}.png
    mv target_view${png}.png ${file%.*}"_"view${png}.png
  done
  fi
done
#####
open *view*.png
