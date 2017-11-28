
#define inputOffset 600
#define tweetOffest 280
#define locationOffest 20

__kernel void parse_timestamp(const __global char * inputData, __global int * year, __global int * month, __global int * day, __global int * hour, __global int * minute) {
	size_t id = get_global_id(0);

	
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

__kernel void parse_scores(__global char * inputData, __global int * envScore, __global int * socialScore, __global int * sentimentScore)
{
		size_t id = get_global_id(0);
		envScore[id] = 9;

		int multiplier = 1000;
		int currIndex = 24 + inputOffset*id;
		int sentiment = 0;
		int social = 0;
		int environ = 0;


		envScore[id] = (int)((inputData[currIndex]) - 48);


		while (inputData[currIndex] != ',')
		{
			sentiment = sentiment * 10 + (int)((inputData[currIndex]) - 48);
			currIndex += 1;
		}
		sentimentScore[id] = sentiment;
		currIndex += 1;
		while (inputData[currIndex] != ',')
		{
			social = social * 10 + (int)((inputData[currIndex]) - 48);
			currIndex += 1;
		}
		socialScore[id] = social;
		currIndex += 1;
		while (inputData[currIndex] != ',')
		{
			environ = environ * 10 + (int)((inputData[currIndex]) - 48);
			currIndex += 1;
		}
		envScore[id] = environ;

	}




__kernel void parse_text(__global char * inputData, __global char * tweet, __global char *location)
{

	size_t id = get_global_id(0);
	int index = 25;
	char tempChar = inputData[inputOffset*id + index];
	while (tempChar != '"') {
		index += 1;
		tempChar = inputData[inputOffset*id + index];
	}
	location[id*inputOffset] = tempChar;
	int locationIndex = 1;
	tempChar = inputData[inputOffset*id + index + locationIndex];
	while (tempChar != '"') {
		location[locationOffest*id + locationIndex] = tempChar;
		locationIndex += 1;
		tempChar = inputData[inputOffset*id + index + locationIndex];
	}
	location[locationOffest*id + locationIndex] = tempChar;
	index = index + locationIndex + 2;
	int tweetIndex = 0;
	tempChar = inputData[inputOffset*id + index + tweetIndex];
	while (tempChar != '\n' & tweetIndex < 280)
	{
		tweet[tweetOffest*id + tweetIndex] = tempChar;
		tweetIndex += 1;
		tempChar = inputData[inputOffset*id + index + tweetIndex];
	}
}