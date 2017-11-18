# CalculateExpectedValues.R
# Author: Andrew Larkin
# November 18, 2017

# Derive expected outcomes for testing dataset at each testing milestone.  
# A separate output file is created for each test milestone. 
# See https://github.com/larkinandy/Green-Space-Virtual-Reality/tree/master/OpenCL_SMA for more details 
# about the related project and https://github.com/larkinandy/Green-Space-Virtual-Reality/tree/master/OpenCL_SMA/Testing_Strategy
# for more datails about the corresponding testing strategy


#################### Derive expected outcomes for test dataset after code block 2 ###############


# calculate the expected test output for hourly tweet indices function
calcExpectedHourlyTweetIndices <-function(inData) 
{
  hourIndices <- c(0)
  for(tweetIndex in 2:length(inData[,1]))
  {
    if(inData$hour[tweetIndex] != inData$hour[tweetIndex-1]) 
    {
      hourIndices <- c(hourIndices,tweetIndex-1)
    }
  }
  return(hourIndices)
}

# calculate indicator variables for tweet text that contains a list of keywords
calcExpectedKeywordIndices <- function(inData,keywordList) 
{
  keyWordIndices <- c()
  for(keywordIndex in 1:length(keywordList)) 
  {
    keyWordIndices <- cbind(keyWordIndices, (grepl(keywordList[keywordIndex], inData$text)*1))
  }
  return(keyWordIndices)
}


  hourIndices <- calcExpectedHourlyTweetIndices(rawData)
  keywordIndicator <- calcExpectedKeywordIndices(rawData,c("snow", "park","I'm"))
  keywordDataframe <- data.frame(Snow = keywordIndicator[,1], Park = keywordIndicator[,2], Im = keywordIndicator[,3])
  write.csv(hourIndices,"block2Test_HourIndices.csv")
  write.csv(keywordDataframe, "block2Test_keywordIndicator.csv")

############### Derive expected outcomes for test dataset after code block 3 ######################


# calculate mean hourly averages for Tweets containing the keyword 'snow'
calcHourlyStats <- function(inData,hourlyIndices,keywordIndicator)
{
  numHours <- length(hourlyIndices)
  meanSentiment <- rep(0,numHours)
  meanEnvScore <- rep(0,numHours)
  meanSocialScore <- rep(0,numHours)
  meanTime <- rep(0,numHours)
  confEnvScore <- rep(0,numHours)
  confSocialScore <- rep(0,numHours)
  confTime <- rep(0,numHours)
  hourlyIndices <- c(hourlyIndices,length(inData[,1]))
  for(hourIndex in 1:(numHours)) 
  {
    start <- hourlyIndices[hourIndex]+1
    end <- hourlyIndices[hourIndex+1]
    if(hourlyIndices[hourIndex]+1 == hourlyIndices[hourIndex+1]) 
    {
      hourlySubset <- inData[hourlyIndices[hourIndex]+1,]
    }
    else 
    {
      hourlySubset <- inData[start:end,]
      confEnvScore[hourIndex] <- sqrt(var(hourlySubset$envScore))*1.96
      confSocialScore[hourIndex] <- sqrt(var(hourlySubset$socialScore))*1.96
      confTime[hourIndex] <- sqrt(var(hourlySubset$minute))*1.96
    }
      meanSentiment[hourIndex] <- mean(hourlySubset$sentiment)
      meanEnvScore[hourIndex] <- mean(hourlySubset$envScore)
      meanSocialScore[hourIndex] <- mean(hourlySubset$socialScore)
      meanTime[hourIndex] <- mean(hourlySubset$minute)/60 + hourlySubset$hour[1]
  }
  outputData <- data.frame(meanSentiment,meanEnvScore,meanSocialScore,meanTime,confEnvScore,confSocialScore,confTime)
  return(outputData)
}
 
  
# calculate color for mean sentiment value 
calcColorGradient <- function(meanSentiment,minColor,maxColor,sentimentRange) 
{
  red <- rep(0,length(meanSentiment))
  green <- rep(0,length(meanSentiment))
  blue <- rep(0,length(meanSentiment))
  normalizedSentiment <- 0
  for(i in 1:length(meanSentiment))
  {
    tempSentiment <- meanSentiment[i]
    if(tempSentiment > sentimentRange[2]) 
    {
      normalizedSentiment <- (tempSentiment - sentimentRange[2])/(sentimentRange[1] - sentimentRange[2])
      rgbMax <- col2rgb(maxColor)
      red[i] = (1-normalizedSentiment) + normalizedSentiment*(rgbMax[1])/(255*2)
      green[i] = (1-normalizedSentiment) + normalizedSentiment*(rgbMax[2])/255
      blue[i] = (1-normalizedSentiment) + normalizedSentiment*(rgbMax[3])/255
      
    }
    else 
    {
      normalizedSentiment <- (tempSentiment - sentimentRange[3])/(sentimentRange[2] - sentimentRange[3])
      rgbMin <- col2rgb(minColor)
      red[i] = (1-normalizedSentiment) + normalizedSentiment*(rgbMin[1])/255
      green[i] = (1-normalizedSentiment) + normalizedSentiment*(rgbMin[2])/255
      blue[i] = (1-normalizedSentiment) + normalizedSentiment*(rgbMin[3])/255
    }
  }
  colorData <- data.frame(red,green,blue)
  return(colorData)
}


############ main function ##########

setwd("I:/GreenSpaceVirtualReality/OpenCL_SMA/Testing_Strategy")
rawData <- read.csv("TestData_OpenCL_SMA.csv")


############ derive expected output for block 2 operations #############
hourIndices <- calcExpectedHourlyTweetIndices(rawData)
keywordIndicator <- calcExpectedKeywordIndices(rawData,c("snow", "park","I'm"))
keywordDataframe <- data.frame(Snow = keywordIndicator[,1], Park = keywordIndicator[,2], Im = keywordIndicator[,3])
write.csv(hourIndices,"block2ExpectedResults_HourIndices.csv")
write.csv(keywordDataframe, "block2ExpectedResults_keywordIndicator.csv")

########### derive expected output for block 3 operations ############

hourlyStats <- calcHourlyStats(rawData,hourIndices,keywordIndicator)
positiveColor <- rgb(0,0,1)
negativeColor <- rgb(1,0,0)
sentimentScores <- c(4,2,0) # 4 is positive, 2 is neutral, 0 is negative
colorGradient <- calcColorGradient(hourlyStats$meanSentiment,negativeColor,positiveColor,sentimentScores)
block3Data <- data.frame(hourlyStats,colorGradient)
write.csv(block3Data, "block3ExpectedResults.csv")

