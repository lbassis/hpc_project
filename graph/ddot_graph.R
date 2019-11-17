file <- "perf_ddot"
data <- read.table(paste(paste("../data/", file, sep = ""), ".data", sep = ""),header=TRUE,sep=",")

mkldata = subset(data, version == "mkl")
mklstepone = subset(mkldata, step == 1)
mklsteptwo = subset(mkldata, step == 2)

mydata = subset(data, version == "my")
mystepone = subset(mydata, step == 1)
mysteptwo = subset(mydata, step == 2)

pdf('stepone.pdf')
plot(mklstepone$n, mklstepone$Mflops, col="blue", type = "o", log = "x", 
     xlab = "N", ylab = "Mflops/s", main="Mflops/s by N with step = 1")
points(mystepone$n, mystepone$Mflops, col="red", pch="*")
lines(mystepone$n, mystepone$Mflops, col="red")
legend("topleft", 
       legend = c("mkl", "my"), 
       col = c("blue", 
               "red"), 
       pch = c("o","*"), 
       bty = "n", 
       text.col = "black", 
       horiz = F , )
dev.off()

pdf('steptwo.pdf')
plot(mklsteptwo$n, mklsteptwo$Mflops, col="blue", type = "o", log = "x", 
     xlab = "N", ylab = "Mflops/s", main="Mflops/s by N with step = 2")
points(mysteptwo$n, mysteptwo$Mflops, col="red", pch="*")
lines(mysteptwo$n, mysteptwo$Mflops, col="red")
legend("topleft", 
       legend = c("mkl", "my"), 
       col = c("blue", 
               "red"), 
       pch = c("o","*"), 
       bty = "n", 
       text.col = "black", 
       horiz = F , )
dev.off()