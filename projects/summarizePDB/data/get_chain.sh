#!/bin/bash
#
if [ $# -ne 3 ]; then echo "ERROR USAGE: $0 <input.pdb> <output.pdb> <list of chains to write to output>"; exit; fi
input=$1
output=$2
chain_list="$3"
#
mkdir .tmp; cd .tmp
echo                                      "#!/bin/bash"                                                 >  tmp.sh
echo -n -e                                "cat ../$input | awk -v FS='' '{if(\$1\$2\$3\$4 != \"ATOM\" " >> tmp.sh
for chain in $chain_list; do echo -n -e   "|| \$22 == \"${chain}\""                                     >> tmp.sh; done
echo -n -e                                ") print \$0 }' > ../$output"                                 >> tmp.sh
echo                                      ""                                                            >> tmp.sh
chmod u+x tmp.sh; ./tmp.sh; cd ../; rm -rf .tmp
