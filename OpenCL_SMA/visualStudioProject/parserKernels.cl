

__kernel void parse_timestamp(__global char * inputData, __global int * year, __global int * month, __global int * day, __global int * hour, __global int * minute) {
	size_t id = get_global_id(0);
	int inputOffset = 600;
	int timeVal = 0;
	int multiplier = 1000;
	for (int index = 0; index < 4; index++)
	{
		int readInt = (int)(inputData[inputOffset*id + index]) - 48;
		timeVal += readInt*multiplier;
		multiplier /= 10;
	}
	year[id] = timeVal;
	month[id] = ((int)(inputData[inputOffset*id + 5]) - 48) * 10 + (int)(inputData[inputOffset*id + 6]) - 48;
	day[id] = ((int)(inputData[inputOffset*id + 20]) - 48) * 100 + ((int)(inputData[inputOffset*id + 21]) - 48) * 10 + (int)(inputData[inputOffset*id + 22]) - 48;
	hour[id] = ((int)(inputData[inputOffset*id + 11]) - 48) * 10 + (int)(inputData[inputOffset*id + 12]) - 48;
	minute[id] = ((int)(inputData[inputOffset*id + 14]) - 48) * 10 + (int)(inputData[inputOffset*id + 15]) - 48;
}

__kernel void parse_scores(__global char * inputData, __global int * envScore, __global int * socialScore, __global int * sentimentScore)
{
	size_t id = get_global_id(0);
	envScore[id] = 9;
	

int inputOffset = 600;
int multiplier = 1000;
int currIndex = 24 + inputOffset*id;
int sentiment = 0;
int social = 0;
int environ = 0;

envScore[id] = (int)((inputData[currIndex]) - 48);


while(inputData[currIndex] != ',')
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
	int inputOffset = 600;
	int tweetOffest = 280;
	int locationOffest = 20;

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
	while(tempChar != '\n' & tweetIndex < 280)
	{ 
		tweet[tweetOffest*id + tweetIndex] = tempChar;
		tweetIndex += 1;
		tempChar = inputData[inputOffset*id + index + tweetIndex];
	}

//	tweet[id] = 'a';
	//location[id] = 'b';*/
}