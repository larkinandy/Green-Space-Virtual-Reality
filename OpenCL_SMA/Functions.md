## Custom functions 

### Block 1
int **Setup_Block1** (char * csvFilename, cl_program * program, cl_context * context, cl_mem * rawData, cl_mem * parsedData, cl_kernel CSV_kernel, int numRows) - check for environment requirements, select ideal platform and computer device, allocate memory for buffers, set kernel args and read data from csv file into memory (without parsing).  Return 1 if all operations were sucessfull, 0 otherwise

// todo: refactor setup_block 1. Subfunctions overlap with setup_block 2 and setup_block 3

kernel **readCSV**(char * inBuffer, int * yearArray, int * dayArray, int *hourArray, int *minuteArray, int *sentimentArray, int *envArray, int *socialArray, char ** textArray) - take an inputBuffer read from disk and returns parsed csv data. <br>

int **setupBlock2** (cl_program * program, cl_context * context, cl_mem * parsedData, cl_mem *hrIdx, cl_mem * keywrdInd, cl_kernel * blck2Kernels) - allocate memory for buffers and set kernel args.  Return 1 if all operations were sucessfull, 0 otherwise

// todo: refactor setup_block 2. Subfunctions overlap with setup_block 1 and setup_block 3

### Block 2
kernel **identify_hourly_tweet_indices** (int * tweetHours, int * hourCutoffs) - takes sorted list of hours each tweet was posted, and returns array of indeces where the first tweet from each hour is located.    <br>

kernel **identify_keyword_indices** (char ** tweetText, int * keywordInd) - takes array of tweet text and returns whether each text contains a keyword within the text. <br>

int **setupBlock3** (cl_program * program, cl_context * context, cl_mem *hr_idx, cl_mem * keywrd_ind, cl_mem * parsedData, cl_mem * statsData, cl_mem colorGrad, cl_kernel * blk3Kernels) - allocate memory for buffers and set kernel args.  Return 1 if all operations were sucessfull, 0 otherwise.  

// todo: refactor setup_block 3.  Subfuctions overlap with setup_block 1 and setup_block 2

### Block 3

kernel **calcMean** (int * keywordInd, int * hrIdx, int * rawVals, float * meanVals) - given an array of input values and indicator variables of whether value should be considered in the calculation, calculate mean hourly values for the variable of interest.  Used to derive estimates of mean environmental score, social score, sentiment, and time.

kernel **calcConf** (int * keywordInd, int * hrIdx, int *rawVals, float *meanVals) - given hourly mean and original values, calcualte hourly 95% confidence intervals for the variable of interested.  Used to derive estimates of env score, social score, and time CIs.

kernel **sentimentToRGB** (float * meanSentiment, float * gradientColor) - given houlry mean sentiment estimates, derive rgb values for a red to white to blue color gradient.  

### Block 4

int **Cleanup** () - release all OpenCL objects and host memory objects.  Returns 1 if sucessfull, 0 otherwise

### Debug functions 

// todo: add debug functions as needed for testing.

