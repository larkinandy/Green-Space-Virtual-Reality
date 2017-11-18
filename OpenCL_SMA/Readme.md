## Preprocesssing Twitter Posts for Graph Analytics in Unreal Engine 4 (UE4) <br>

### Summary
This script prepares Twitter text and attributes for graph-based visualization in Unreal Engine 4 using OpenCL.  Given a twitter dataset with timestamps for each twwet, summary statistics are calculated for each hour of coverage and used to generate output nodes in UE4.  Users can specify data subsets based on a time range of subset of twitter containing search keywords.  Specifically, this script performs the following steps:
1. Subset an input dataset based on a time range or list of keywords
2. Calculate the mean and variance of previously dervied numeriously attributes for each tweet, including sentiment, environmental and social dimension scores
3. Convert mean sentiment score into color RGB color gradient value based on the following color scheme: <br>
![](https://github.com/larkinandy/Green-Space-Virtual-Reality/blob/master/OpenCL_SMA/Support%20Documents/Sentiment_Color_Gradient-03.png)

4. Output results in struct format used to generate [ellipsoid confidence regions](https://en.wikipedia.org/wiki/Ellipsoid)  (one ellipsoid for each hourly summary statistic set) in UE4, where the x, y, and z axes corrrespond to time, environmental score, and social score, respectively.  
 
![alt text](https://github.com/larkinandy/Green-Space-Virtual-Reality/blob/master/OpenCL_SMA/Support%20Documents/SpherePrototypes.gif "Prototype uobjects in UE4")


### Program Requirements
Hardware and software used for program development and testing are listed below.  Performance on other configurations may vary <br>
**OpenCL v. 1.2 and 2.0** <br>
**Computing Device Types** Designed and optimized for graphics processing units (GPUs) <br>
**Operating System:** Windows 7 and 10.  <br>
**GPU Computing Devices:** NVIDIA Titan X Pascal, NVIDIA GTX 980M, AMD FirePro V4900, and Intel Iris (//todo: insert version) <br>
**Unreal Engine 4 v. 4.18.1** 


### Flowchart 
Program overview is shown below. Identifying hourly and keyword subsets are performed asynchronously in separate kernels, followed by a sync barrier to ensure completion before progressing to kernels that are dependent on initial kernel results.  After the sync barrier, variable statistics are derived in parallel.
![](https://github.com/larkinandy/Green-Space-Virtual-Reality/blob/master/OpenCL_SMA/Support%20Documents/Project%20Flowchart_Nov17_17.png) <br>

**Setup Block 1** - Perform operations needed to read csv file on the device.  Operations include loading and building programs and kernels, calculating buffer size and allocating memory for CSV read, and transferring data from host to device. <br>

**Read CSV** - Kernel operation to read csv file.  Based on the 'Fast C++ CSV Parser' created by Ben Strasser (https://github.com/ben-strasser/fast-cpp-csv-parser). <br>

**Setup Block 2** - Perform operations needed for kernel search operations in Block 2.  Operations include building kernels and allocating buffers.  While these operations could technically be included in setup block 1 with little impact on performance, they're encapsulated in a separate function to improve modularity (e.g. for follow up use with a second keyword search).  

**Identify hourly tweet indices** - Kernel operation to identify index barriers for each hour of data coverage

**Identify keyword indices** - Kernel operation to identify indices of Tweets that contain the keyword provided as an input argument.

**Calc mean env score/social score/sentiment/time** - Kernel operation to calculate hourly mean of a variable of interest

**Calc var env score/social score/time** - Kernel operation to calculate variance of a variable of interest

**Convert sentiment to rgb** - Convert mean sentiment value to rgb tuple using the three color gradient shown above in the summary

**Cleanup** - Release created OpenCL objects

**Return data** - Return derived variables to the calling program in a format that minimizes time required to generate confidence regions in UE4.  Ojbects in UE4 must be created and modified from the main thread.  

**Additional operations** - All custom functions and corresponding syntax are listed in the supplemental file [OpenCL_SMA_functions](https://github.com/larkinandy/Green-Space-Virtual-Reality/blob/master/OpenCL_SMA/Functions.md)

### Program Files
Program code consists of four files:
1. OpenCL_SMA.cpp - Contains the majority of host code and handles program flow.
2. OpenCL_SMA_kernels.cl - Contains device kernels.
3. Support_SMA.cpp - debug and support functions, including identifying platform and device characteristics and choosing optimal options
4. Test_Driver.cpp - simulates the Unreal Engine 4 calling function.  

### Program Testing and Validation
Test harness and corresponding dataset are available in the [Testing Strategy](https://github.com/larkinandy/Green-Space-Virtual-Reality/tree/master/OpenCL_SMA/Testing_Strategy) folder.  Testing strategy is based on testing functions independnetly during code creaation, followed by integrative and iterative testing at the end of each code block.  For example, at the conclusion of block 2 the test harness for both block 1 and 2 are run to ensure edits made after  block 1 verification did not change block 1 code validity.  

### Version Control
SourceTree was used for version control with a .git repository.  Committs were performed on a daily basis, and tags were added after reaching each milestone (code block validation).  

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
