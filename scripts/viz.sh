#!/bin/bash
#
# RETRIEVE ARGUMENTS #
if [ $# -ne 1 ]; then echo "please provide name of reference mean file"; exit;fi
ref=$1
relpath="$(dirname $0)"
#
# PREPARE LOCAL TREE #
if [ -d figs ]; then echo "a figs/ directory is already here. Abort."; exit; fi
rm -rf .tmp
mkdir figs
cd figs
mkdir jpgs
mkdir jpgs/components
mkdir jpgs/clusters
mkdir pngs
mkdir pngs/components
mkdir pngs/clusters
mkdir gifs
mkdir gifs/views
mkdir mp4s
mkdir mp4s/views
cd ../
# 
#### visualize components
for file in *oscill*.pdb
do
  #### X Y Z views of the component in porcupine style
  # skip for now #
  #echo "> modeviz $file"
  #"$relpath"/modeviz.sh $file
  #mv view0.jpg ./figs/jpgs/components/${file%.*}".jpg"
  #for png in 1 2 3
  #do
  #  mv view${png}.png ./figs/pngs/components/${file%.*}"_"view${png}.png
  #done
  #### make gif and mp4
  echo "> mode_movie $file"
  "$relpath"/mode_movie.sh $file
  mv movie.gif ./figs/gifs/${file%.*}".gif"
  mv movie.mp4 ./figs/mp4s/${file%.*}".mp4"
  for png in 1 2 3
  do
    mv movie${png}.gif ./figs/gifs/views/${file%.*}"_"view${png}.gif
    mv movie${png}.mp4 ./figs/mp4s/views/${file%.*}"_"view${png}.mp4
  done
done
#### visualize inter-cluster motions
for file in *_mean.pdb
do
  if [ $file != $ref ]; then
    echo "> $ref $file"
    "$relpath"/clusterdiffviz.sh $ref $file
  mv mode_view0.jpg   ./figs/jpgs/clusters/${ref%.*}"_to_"${file%.*}".jpg"
  mv ref_view0.jpg    ./figs/jpgs/clusters/${ref%.*}".jpg"
  mv target_view0.jpg ./figs/jpgs/clusters/${file%.*}".jpg"
  for png in 1 2 3
  do
    mv mode_view${png}.png   ./figs/pngs/clusters/${ref%.*}"_to_"${file%.*}"_"view${png}.png
    mv ref_view${png}.png    ./figs/pngs/clusters/${ref%.*}"_"view${png}.png
    mv target_view${png}.png ./figs/pngs/clusters/${file%.*}"_"view${png}.png
  done
  fi
done
#### summarize what has been done
ls -ltrh ./figs/*
