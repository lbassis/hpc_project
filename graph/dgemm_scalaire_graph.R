file <- "perf_dgemm_scalaire"
data <- read.table(paste(paste("../data/", file, sep = ""), ".data", sep = ""),header=TRUE,sep=",")

ikjdata = subset(data, version == "ikj")
kijdata = subset(data, version == "kij")
ijkdata = subset(data, version == "ijk")
jikdata = subset(data, version == "jik")

pdf('gemm_scalaire.pdf')
plot(ijkdata$n, ikjdata$Mflops, col="blue", type = "o", pch="*", log = "x", 
     xlab = "N", ylab = "Mflops/s", main="Mflops/s by N for GEMM scalaire")
points(kijdata$n, kijdata$Mflops, col="red", pch="*")
lines(kijdata$n, kijdata$Mflops, col="red")
points(ijkdata$n, ijkdata$Mflops, col="green", pch="*")
lines(ijkdata$n, ijkdata$Mflops, col="green")
points(jikdata$n, jikdata$Mflops, col="brown", pch="*")
lines(jikdata$n, jikdata$Mflops, col="brown")


legend("topright", 
       legend = c("ikj", "kij", "ijk", "jik"), 
       col = c("blue", "red", "green", "brown"), 
       pch = c("*","*", "*", "*"), 
       bty = "n", 
       text.col = "black", 
       horiz = F , )
dev.off()