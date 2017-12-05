setwd("I:/GreenSpaceVirtualReality/OpenCL_SMA/visualStudioProject")

start.time <- Sys.time()
rawData <- read.csv("TestData_R_1000000.csv")
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken*1000
