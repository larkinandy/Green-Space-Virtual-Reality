## Preprocesssing Twitter Posts for Graph Analytics in Unreal Engine 4 (UE4) <br>

### Summary
This script prepares Twitter text and attributes for graph-based visualization in Unreal Engine 4 using OpenCL.  Given a twitter dataset with timestamps for each twwet, summary statistics are calculated for each hour of coverage and used to generate output nodes in UE4.  Users can specify data subsets based on a time range of subset of twitter containing search keywords.  Specifically, this script performs the following steps:
1. Subset an input dataset based on a time range or list of keywords
2. Calculate the mean and variance of previously dervied numeriously attributes for each tweet, including sentiment, environmental and social dimension scores
3. Convert variance of derived attributes into [ellipsoid coordinates](https://en.wikipedia.org/wiki/Ellipsoid)
4. Convert mean sentiment score into color RGB color gradient value based on the following color scheme:
      //todo: insert color scheme


5. Output results in struct format conducive to creating uobjects (one object for each hourly summary statistic set) in UE4.
//todo: insert image of constrcuted nodes 

### Program Requirements
**Computing Device Types** Designed and optimized for graphics processing units (GPUs) <br>
**Operating System:** Windows 7 and 10 <br>
**GPU Computing Devices:** Tested on NVIDIA Titan X Pascal, AMD (insert name), and Intel Iris <br>
**Unreal Engine 4 v. 4.17.1** 


### Flowchart 
Program overview is shown below. Identifying hourly and keyword subsets are performed asynchronously in separate kernels, followed by a sync barrier to ensure completion before progressing to kernels that are dependent on initial kernel results.  After the sync barrier, hourly subsets are processed in parallel.
![build status](https://github.com/larkinandy/Green-Space-Virtual-Reality/blob/master/OpenCL_SMA/Support%20Documents/Project%20Overview.png) <br>

### Input Data 
Data is stored .csv format.  Example data is provided in Supplemental Files.  Attributes include:<br>
1. Timestamp (integer) - time of Tweet, in number of milliseconds since 1970  <br>
2. Year (integer) - year of Tweet, in YYYY format 
3. Day of year (integer) - Julian day of the year 
4. Hour (integer) - hour of the day 
5. Sentiment (integer) - sentiment of Tweet.  (0,1,2) for negative, netural, and positive sentiment
6. Environmental score (float) - environemntal score of Tweet (//todo: insert environmental score script).  Ranges from 0-100
7. Social score (float) - social score of Tweet (//todo: insert social score script).  Ranges from 0-100
8. Text (string) - raw Twitter text, not preprocessed 

### Output Data
Output data is stored in .csv format for debugging purposes, but will be sent directly to the Unreal Engine 4 main thread when integrated into the target program.  Output data will be stored as a struct with the following variables:
1. MeanTime (float array) - mean timestamp for all hours which contain at least one filtered Tweet
2. SentimentColor (integer array) - Colors for all hours, derived from mean sentiment scores 
3. MeanEnv (float array) - mean environmental score for all hours
4. MeanSocial (float array) - mean social score for all hours
5. ConfidenceRegion (float array) - coordinates for confidence region for each point.
