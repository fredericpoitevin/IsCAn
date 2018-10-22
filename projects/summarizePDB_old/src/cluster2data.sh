#!/bin/bash
# 
# takes output from run ($1) and generate input for another one ($2)
#
if [ $# -ne 2 ]; then echo "ERROR"; exit; fi
grep "REMARK" $1 | sed 's/REMARK ID list://g' > ${2}.id
cp $1 ${2}.pdb
