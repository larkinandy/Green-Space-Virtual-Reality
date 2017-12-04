
#define inputOffset 600
#define tweetOffest 280
#define locationOffset 20

__kernel void parse_timestamp(const __global char * inputData, __global int * year, __global int * month, __global int * day, __global int * hour, __global int * minute, int maxThread) {
	
	
	
	size_t id = get_global_id(0);
	if (id >= maxThread) { return; }
	
	int timeVal = 0;
	int multiplier = 1000;
	int readInt;	
	for (int index = 0; index < 4; index++)
	{
		readInt = (inputData[inputOffset*id + index]) - 48;
		timeVal += readInt*multiplier;
		multiplier /= 10;
	}
	year[id] = timeVal;
	
	timeVal = 0;
	multiplier = 10;
	for(int index = 5; index < 7; index ++) 
	{ 
		readInt = (inputData[inputOffset*id + index]) - 48;
		timeVal += readInt*multiplier;
		multiplier /= 10;
	}
	month[id] = timeVal;

	timeVal = 0;
	multiplier = 100;
	for (int index = 20; index < 23; index++)
	{
		readInt = (inputData[inputOffset*id + index]) - 48;
		timeVal += readInt*multiplier;
		multiplier /= 10;
	}
	day[id] = timeVal;

	timeVal = 0;
	multiplier = 10;
	for (int index = 11; index < 13; index++)
	{
		readInt = (inputData[inputOffset*id + index]) - 48;
		timeVal += readInt*multiplier;
		multiplier /= 10;
	}
	hour[id] = timeVal;

	timeVal = 0;
	multiplier = 10;
	for (int index = 14; index < 16; index++)
	{
		readInt = (inputData[inputOffset*id + index]) - 48;
		timeVal += readInt*multiplier;
		multiplier /= 10;
	}
	minute[id] = timeVal;
	
}

__kernel void parse_scores(__global char * inputData, __global int * envScore, __global int * socialScore, __global int * sentimentScore, int maxThread)
{
		size_t id = get_global_id(0);
		if (id >= maxThread) {	return;}

		int multiplier = 1000;
		int currIndex = 24 + inputOffset*id;
		int sentiment = 0;
		int social = 0;
		int environ = 0;
		
		int newDigit = inputData[currIndex] -48;
		while (newDigit != -4)
		{
			sentiment = sentiment * 10 + newDigit;
			currIndex += 1;
			newDigit = inputData[currIndex] - 48;
		}
		sentimentScore[id] = sentiment;
		currIndex += 1;
		newDigit = inputData[currIndex] - 48;
		
		while (newDigit != -4)
		{
			social = social * 10 + newDigit;
			currIndex += 1;
			newDigit = inputData[currIndex] - 48;
		}
		socialScore[id] = social;
		currIndex += 1;
		newDigit = inputData[currIndex] - 48;
		while (newDigit != -4)
		{
			environ = environ * 10 + newDigit;
			currIndex += 1;
			newDigit = inputData[currIndex] - 48;
		}
		envScore[id] = environ;

	}




__kernel void parse_text(__global char * inputData, __global char * tweet, __global char *location, int maxThread)
{
	
	size_t id = get_global_id(0);
	if (id >= maxThread) { return; }
	int index = 25;
	int threadOffset = inputOffset*get_global_id(0);
	int locationIndex = 0;

	
	char tempChar = inputData[threadOffset + index];
	// go through input until quotation is found
	
	while (tempChar != '"') {
		index += 1;
		tempChar = inputData[threadOffset + index];
	}
	
	index += 1;
	tempChar = inputData[threadOffset + index + locationIndex];
	while (tempChar != '"' & locationIndex < 20) {
		location[id*locationOffset + locationIndex] = tempChar;
		locationIndex += 1;
		tempChar = inputData[threadOffset + index + locationIndex];
	} 
	
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

__kernel void new_line(__global char * inputData, __global int * outputData, int maxThread)
{
	
	size_t id = get_global_id(0);
	size_t local_id = get_local_id(0);
	if (id >= maxThread) { return; }
	__local int numVals[5];
	__local int counter;
	__local int groupOffset;
	int tempInt;
	
	if(local_id == 0) 
	{
		counter = 0;
		numVals[0] = 0;
		groupOffset = get_group_id(0)*250;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if(inputData[id] == '\n') { 
		tempInt = atomic_inc(&numVals[0]);
		numVals[tempInt+1] = id;
		outputData[id] = id;
	}
	else
	{ 
		outputData[id] = 0; 
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if (local_id != 0) { return; }
	int largestIndex;
	if(numVals[0] == 3) 
	{
		if (numVals[1] > numVals [3]) 
		{
			largestIndex = numVals[3];
			numVals[3] = numVals[1];
			numVals[1] = largestIndex;
		}
		else if (numVals[2] > numVals[3]) 
		{
			largestIndex = numVals[3];
			numVals[3] = numVals[2];
			numVals[2] = largestIndex;
		}
	}
	if (numVals[1] > numVals[2] & numVals[2] > 0)
	{
		outputData[groupOffset + 1] = numVals[2];
		outputData[groupOffset + 2] = numVals[1];
		outputData[groupOffset + 3] = numVals[3];
	}
	else 
	{ 
		outputData[groupOffset + 1] = numVals[1];
		outputData[groupOffset + 2] = numVals[2];
		outputData[groupOffset + 3] = numVals[3];
	}
		outputData[groupOffset] = numVals[0];

	
}

// use global id to collapse groups with indeces of new line markers
__kernel void collapseValsGlobal(__global int *inputData, int offset, int maxRange, int groupSize)
{
	size_t id = get_global_id(0);
	int parentGroupIndex = (id / groupSize)*offset*2;
	barrier(CLK_GLOBAL_MEM_FENCE);
	int blockId = id % groupSize;
	barrier(CLK_GLOBAL_MEM_FENCE);
	int numAlready = inputData[parentGroupIndex];
	barrier(CLK_GLOBAL_MEM_FENCE);
	int numToAdd = inputData[parentGroupIndex + offset];
	barrier(CLK_GLOBAL_MEM_FENCE);
	if (blockId > numToAdd) 
	{
		return;
	}
	if (blockId == 0) 
	{
		inputData[parentGroupIndex] += numToAdd;
	}
	else 
	{ 
		inputData[parentGroupIndex + numAlready + blockId] = inputData[parentGroupIndex + blockId + offset];
	}
	
}
