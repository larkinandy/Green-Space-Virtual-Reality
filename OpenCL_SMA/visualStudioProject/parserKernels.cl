/* parserKernels.cl
* OpenCL kernels for parsing a csv file on GPU devices 
* Author: Andrew Larkin
* December 5, 2017 */

// todo: is it better to use defined offsets are pass args at input?  Args provide more flexibility, but require more error checking and
#define tweetOffest 280
#define locationOffset 20

/* extract year, month, hour, and minute from timestamp at fixed location and YYYY-mm-dd format in the csv file.  Day of year is provided
at fixed location and number of digits following timestamp */
__kernel void parse_timestamp(const __global char * inputData, __global int * year, __global int * month, __global int * day, __global int * hour, 
	__global int * minute, int maxThread, __global int * startingOffset) 
{
	size_t id = get_global_id(0);
	if (id >= maxThread) { return; }  // records are processed in batches.  Last batch may have more threads than records to parse
	int startingIndex = startingOffset[id] + 1;
	if (id == 0) { startingIndex = 0; }
	
	// get year 
	int timeVal = 0;
	int multiplier = 1000;
	int readInt;
	for (int index = 0; index < 4; index++) // year is in 4 digit YYYY format
	{
		readInt = (inputData[startingIndex + index]) - 48;  // use 48 to convert ascii to decimal
		timeVal += readInt*multiplier;
		multiplier /= 10;
	}
	year[id] = timeVal;

	// get month
	timeVal = 0;
	multiplier = 10;
	readInt = 0;
	for (int index = 5; index < 7; index++) // month is in mm two digit format
	{
		readInt = (inputData[startingIndex + index]) -  48;
		timeVal += readInt*multiplier;
		multiplier /= 10;
	}
	month[id] = timeVal;
	
	// get day of year 
	timeVal = 0;
	multiplier = 100;
	for (int index = 20; index < 23; index++)
	{
		readInt = (inputData[startingIndex + index]) - 48;
		timeVal += readInt*multiplier;
		multiplier /= 10;
	}
	day[id] = timeVal;						// day is in three digit (JJJ) format
	 
	// get hour 
	timeVal = 0;
	multiplier = 10;
	for (int index = 11; index < 13; index++)
	{
		readInt = (inputData[startingIndex + index]) - 48;		// hour is in two digit HH format
		timeVal += readInt*multiplier;
		multiplier /= 10;
	}
	hour[id] = timeVal;

	// get minute
	timeVal = 0;
	multiplier = 10;
	for (int index = 14; index < 16; index++)
	{
		readInt = (inputData[startingIndex + index]) - 48;		// minute is in two digit MM format
		timeVal += readInt*multiplier;
		multiplier /= 10;
	}
	minute[id] = timeVal;
}

// extract numerial variables (scores for this project) from csv file. Starting location and number of digits is after the 24th digit of a row entry but not fixed
__kernel void parse_scores(__global char * inputData, __global int * envScore, __global int * socialScore, __global int * sentimentScore, 
	int maxThread, __global int * startingOffset)
{
	size_t id = get_global_id(0);
	if (id >= maxThread) { return; } // records are processed in batches.  Last batch may have more threads than records

	int multiplier = 1000;
	int currIndex = 25 + startingOffset[id];
	int sentiment = 0;
	int social = 0;
	int environ = 0;
	if(id==0) { currIndex = 24; } // for the first element, the starting offset is 0.  Ignore starting offset	

	// get sentiment score
	int newDigit = inputData[currIndex] - 48;  // 48 is used to convert ascii to decimal 
	while (newDigit != -4)
	{
		sentiment = sentiment * 10 + newDigit;
		currIndex += 1;
		newDigit = inputData[currIndex] - 48;
	}
	sentimentScore[id] = sentiment;
	currIndex += 1;

	// get environmental score
	newDigit = inputData[currIndex] - 48;
	while (newDigit != -4) // while new digit is not a comma.  Asssuming that comma will be reached before newline
	{
		environ = environ * 10 + newDigit;
		currIndex += 1;
		newDigit = inputData[currIndex] - 48;
	}
	envScore[id] = environ;
	currIndex += 1;

	// get social score
	newDigit = inputData[currIndex] - 48;
	while (newDigit != -4)  // same here.  Assuming comma will be reached before newline.
	{
		social = social * 10 + newDigit;
		currIndex += 1;
		newDigit = inputData[currIndex] - 48;
	}
	socialScore[id] = social;
}

// parset text into location and tweet text variables. Length is variable, starting location is after index 25 but not fixed
__kernel void parse_text(__global char * inputData, __global char * tweet, __global char *location, int maxThread, __global int * startingOffset)
{
	size_t id = get_global_id(0);
	if (id >= maxThread) { return; }
	int index = 25;
	int threadOffset = startingOffset[id] + 1;
	if (id == 0) { threadOffset = 0; }
	int locationIndex = 0;

	char tempChar = inputData[threadOffset + index];
	// go through input until quotation is found
	
	while (tempChar != '"') 
	{
		index += 1;
		tempChar = inputData[threadOffset + index];
	}
	
	// extract location text.  Location preceedes twitter text in csv input data
	index += 1;
	tempChar = inputData[threadOffset + index + locationIndex];
	while (tempChar != '"' & locationIndex < 20) 
	{
		location[id*locationOffset + locationIndex] = tempChar;
		locationIndex += 1;
		tempChar = inputData[threadOffset + index + locationIndex];
	} 

	// extract twitter text
	index = index + locationIndex + 2;
	int tweetIndex = 0;
	tempChar = inputData[threadOffset + index];
	while (tempChar != '\n' & tweetIndex < 280)
	{
		tweet[tweetOffest*id + tweetIndex] = tempChar;
		tweetIndex += 1;
		tempChar = inputData[threadOffset + index + tweetIndex];
	}
}

__kernel void new_line(__global char * inputData, volatile __global int * outputData)
{	
	size_t id = get_global_id(0);
	
	// 250 elements per group.  Entire dataset consists of 250*(n group) elements
	int groupId = (id / 250);
	int withinGroupId = id % 250;  
	int startingGroupIndex = groupId * 250;
	// seed entire output with zeros.  Zeros indicate no new line.  Important to 
	// remove stale data from memory before next step
	outputData[id] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);

	// global ids where new lines are located are added to output
	if (inputData[id] == '\n') {
		int tempInt = atomic_inc(&outputData[startingGroupIndex]);
		outputData[startingGroupIndex + tempInt + 1] = id;
	}
		
	barrier(CLK_LOCAL_MEM_FENCE);

	// sort first three elements of batch in private memory
	if (withinGroupId != 0) {return;}
		int index1 = outputData[startingGroupIndex + 1];
		int index2 = outputData[startingGroupIndex + 2];
		int index3 = outputData[startingGroupIndex + 3];
		int tempSwitcher = 0;
		if (index1 > index2 & index2 > 0) 
		{
			tempSwitcher = index2;
			index2 = index1;
			index1 = tempSwitcher;
			if (index2 > index3 & index3 > 0)
			{
				tempSwitcher = index3;
				index3 = index2;
				index2= tempSwitcher;
			}
			if (index2 < index1)
			{
				tempSwitcher = index2;
				index2 = index1;
				index1 = tempSwitcher;
			}
		}
		else if (index2 > index3 & index3 > 0)
		{
			tempSwitcher = index3;
			index3 = index2;
			index2 = tempSwitcher;
			if (index2 < index1)
			{
				index2 = tempSwitcher;
				index2 = index1;
				index1 = tempSwitcher;
			}
		}
		if (index1 > index2 & index2 > 0) 
		{
			tempSwitcher = index2;
			index2 = index1;
			index1 = tempSwitcher;
		}

		// write output of sort to global memory
		outputData[startingGroupIndex + 1] =  index1;
		outputData[startingGroupIndex + 2] =  index2;
		outputData[startingGroupIndex + 3] = index3;
}

// use global id to collapse scattered groups of elements indicating where new lines are located in the csv
// important to keep records in sorted order
__kernel void collapse_vals_global(__global int *inputData, int offset, int groupSize)
{
	size_t id = get_global_id(0);
	int parentGroupIndex = (id / groupSize)*offset*2;
	int blockId = id % groupSize;
	int numAlready = inputData[parentGroupIndex];
	barrier(CLK_LOCAL_MEM_FENCE); // not sure this block is needed given block below, but results are inconsistent without it
	int numToAdd = inputData[parentGroupIndex + offset];
	barrier(CLK_LOCAL_MEM_FENCE); // ensure num to add is read by all threads before updating
	
	// keep one thraed for each newline character to move, 
	//plus one to update number of contiguous newline characterss
	if (blockId > numToAdd) { return; } 
	if (blockId == 0) 
	{ 
		inputData[parentGroupIndex] += numToAdd; 
	}
	else 
	{ 
		inputData[parentGroupIndex + numAlready + blockId] = inputData[parentGroupIndex + blockId + offset];
	}
}
