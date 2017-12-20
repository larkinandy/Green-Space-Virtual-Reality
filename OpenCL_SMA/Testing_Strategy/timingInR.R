setwd("D:/VirturalRealityGreenSpace/OpenCL_SMA/visualStudioProject")

start.time <- Sys.time()
rawData <- read.csv("TestData_R_100.csv")
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken*1000
