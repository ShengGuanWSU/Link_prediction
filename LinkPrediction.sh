#!/bin/bash

#PBS -V
#PBS -N PCST
#PBS -q wudb
#PBS -l nodes=1:bigmem:ppn=10,walltime=5:25:00
#PBS -l mem=20gb
#PBS -e /data/wudb/users/sguan/pcst/Optimize_index/result/fig5b_yelp_final_official_err1.txt
#PBS -o /data/wudb/users/sguan/pcst/Optimize_index/result/fig5b_yelp_final_official_out1.txt
#PBS -M sheng.guan@wsu.edu
#PBS -m abe

### -e: error log
### -o: output log stdout

module load compilers/gcc/6.2.0

echo 'START runPCST.sh'
cd /data/wudb/users/sguan/pcst/Optimize_index/
./PCSTStaticV5 \
/data/wudb/users/sguan/pcst/data/yelp/yelp_sample_edgeList_test.txt \
/data/wudb/users/sguan/pcst/data/yelp/yelp_profile.txt \
/data/wudb/users/sguan/pcst/data/yelp/yelp300nodes.penalties_official_000001.txt \
/data/wudb/users/sguan/pcst/data/yelp/yelp300nodes.penalties_official_000001.txt \
10 \

echo 'Done The result is shown in pcst'
