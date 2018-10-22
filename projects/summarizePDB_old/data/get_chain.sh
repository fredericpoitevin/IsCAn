#!/bin/bash
#
key=$1
cat ${key}.pdb | awk -v FS="" '{if($1$2$3$4!="ATOM" || $22=="z" || $22=="x" || $22="y" ) print $0}' > ${key}_rna.pdb
