
#define inputOffset 600
#define tweetOffest 280
#define locationOffset 20


__kernel void parse_timestamp(const __global char * inputData, __global int * year, __global int * month, __global int * day, __global int * hour, 
	__global int * minute, int maxThread, __global int * startingOffset, int batchOffset) {

	

	size_t id = get_global_id(0);
	if (id >= maxThread) { return; }
	int startingIndex = startingOffset[id] + 1;
	if (id == 0) { startingIndex = 0; }
	int timeVal = 0;
	int multiplier = 1000;
	int readInt;
	
	for (int index = 0; index < 4; index++)
	{
		readInt = (inputData[startingIndex + index]) - 48;
		timeVal += readInt*multiplier;
		multiplier /= 10;
	}
	year[id] = timeVal;

	timeVal = 0;
	multiplier = 10;
	readInt = 0;
	for (int index = 5; index < 7; index++)
	{
		readInt = (inputData[startingIndex + index]) -  48;
		timeVal += readInt*multiplier;
		multiplier /= 10;
	}
	month[id] = timeVal;
	
	timeVal = 0;
	multiplier = 100;
	for (int index = 20; index < 23; index++)
	{
		readInt = (inputData[startingIndex + index]) - 48;
		timeVal += readInt*multiplier;
		multiplier /= 10;
	}
	day[id] = timeVal;
	 
	timeVal = 0;
	multiplier = 10;
	for (int index = 11; index < 13; index++)
	{
		readInt = (inputData[startingIndex + index]) - 48;
		timeVal += readInt*multiplier;
		multiplier /= 10;
	}
	hour[id] = timeVal;

	timeVal = 0;
	multiplier = 10;
	for (int index = 14; index < 16; index++)
	{
		readInt = (inputData[startingIndex + index]) - 48;
		timeVal += readInt*multiplier;
		multiplier /= 10;
	}
	minute[id] = timeVal;

}

__kernel void parse_scores(__global char * inputData, __global int * envScore, __global int * socialScore, __global int * sentimentScore, int maxThread, __global int * startingOffset, int batchOffset)
{
	size_t id = get_global_id(0);
	if (id >= maxThread) { return; }

	int multiplier = 1000;
	int currIndex = 25 + startingOffset[id];
	int sentiment = 0;
	int social = 0;
	int environ = 0;
	if(id==0) { currIndex = 24; }

	int newDigit = inputData[currIndex] - 48;
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
		environ = environ * 10 + newDigit;
		currIndex += 1;
		newDigit = inputData[currIndex] - 48;
	}
	envScore[id] = environ;
	currIndex += 1;
	newDigit = inputData[currIndex] - 48;
	while (newDigit != -4)
	{
		social = social * 10 + newDigit;
		currIndex += 1;
		newDigit = inputData[currIndex] - 48;
	}
	socialScore[id] = social;
	
}

__kernel void parse_text(__global char * inputData, __global char * tweet, __global char *location, int maxThread, __global int * startingOffset, int startingIndex)
{
	
	size_t id = get_global_id(0);
	if (id >= maxThread) { return; }
	int index = 25;
	int threadOffset = startingOffset[id] + 1;
	if (id == 0) { threadOffset = 0; }
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

__kernel void new_line(__global char * inputData, volatile __global int * outputData, int maxThread)
{
	
	size_t id = get_global_id(0);
	if (id == 0) {
	}
	
	int groupId = (id / 250);
	int withinGroupId = id % 250;
	int startingGroupIndex = groupId * 250;
	if (withinGroupId == 0) {
		outputData[startingGroupIndex] = 0;

	}
	outputData[id] = 0;

	barrier(CLK_LOCAL_MEM_FENCE);
	if (inputData[id] == '\n') {
		int tempInt = atomic_inc(&outputData[startingGroupIndex]);
		outputData[startingGroupIndex + tempInt + 1] = id;
	}
	
	
	barrier(CLK_LOCAL_MEM_FENCE);
	if (groupId != 0) { return; }
	int largestIndex;
	int index1 = outputData[startingGroupIndex + 1];
	int index2 = outputData[startingGroupIndex + 2];
	int index3 = outputData[startingGroupIndex + 3];
	int tempSwitcher = 0;
	if (index1 > index2 & index2 > 0) {
		tempSwitcher = index2;
		index2 = index1;
		index1 = index2;
		if (index1 > index3 & index3 > 0)
		{
			tempSwitcher = index3;
			index1 = index3;
			index1 = tempSwitcher;
		}
	}
	else if (index2 > index3 & index3 > 0) 
	{
		tempSwitcher = index3;
		index2 = index3;
		index2 = tempSwitcher;
		if (index2 < index1) 
		{
			index2 = tempSwitcher;
			index2 = index1;
			index1 = index2;
		}
	}

	outputData[startingGroupIndex + 1] = index1;
	outputData[startingGroupIndex + 2] = index2;
	outputData[startingGroupIndex + 3] = index3;
	
}




// use global id to collapse groups with indeces of new line markers
__kernel void collapseValsGlobal(__global int *inputData, int offset, int maxRange, int groupSize)
{
	size_t id = get_global_id(0);
	int parentGroupIndex = (id / groupSize)*offset*2;
	int blockId = id % groupSize;
	int numAlready = inputData[parentGroupIndex];
	barrier(CLK_LOCAL_MEM_FENCE);
	int numToAdd = inputData[parentGroupIndex + offset];
	barrier(CLK_LOCAL_MEM_FENCE);
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
