using StatsBase, PLINK

# generate random binary genotype data
n = 5000
p = 24000
srand(2016)
x_float = sample([0.0, 1.0, 2.0], (n,p))
y = PLINK.compress(x_float)
yt = PLINK.compress(x_float')
yt[3] = PLINK.ZERO8 # indicate row-major ordering

# save data to file
write(open("x_test.bed", "w"), y)
write(open("xt_test.bed", "w"), yt)

# need to make BIM, FAM files
# make BIM file first
BIM = open("x_test.bim", "w")
#for i = 1:(p-1)
for i = 1:p
    @printf(BIM, "%d\t%s\t%d\t%d\t%s\t%s\n", 1, "rs" * "$i", 0, i, "A", "T")
end
#@printf(BIM, "%d\t%s\t%d\t%d\t%s\t%s", 1, "rs" * "$p", 0, p, "A", "T")
close(BIM)

# now make FAM
FAM = open("x_test.fam", "w")
#for i = 1:(n-1)
for i = 1:n
    @printf(FAM, "%d\t%d\t%d\t%d\t%d\t%d\n", i, i, 0, 0, 1, 1)
end
#@printf(FAM, "%d\t%d\t%d\t%d\t%d\t%d", n, n, 0, 0, 1, 1)
close(FAM)
