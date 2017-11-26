setwd("D:/VirturalRealityGreenspace/OpenCL_SMA/visualStudioProject")

start.time <- Sys.time()
rawData <- read.csv("RCopy2.csv")
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken