TOOLS FOR MANIPULATING PLINK FILES
(c) Gary K. Chen (2015)

To compile, run 

./compile.sh

To learn more about plink_data, just run with no arguments

./plink_data

To transpose a BED file containing an  N x P matrix, use

./plink_data BEDFILE.bed $P $N --transpose BEDFILE_T.bed
