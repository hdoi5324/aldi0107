#!/bin/bash

cwd=`pwd`
echo $cwd

output_dir="~/aldi0101/"
cd $output_dir

#rm files_to_copy.txt
#find ./ -type f -wholename '*/final_model.pth' > files_to_copy.txt

rsync -av --files-from model_files.txt hdoi5324@saga.sigma2.no:/cluster/home/hdoi5324/aldi0107/ .
cd ${cwd}
