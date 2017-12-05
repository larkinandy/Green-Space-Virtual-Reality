#include "ClParser.hpp"


ClParser::ClParser(cl_context * contextPtr, cl_device_id * deviceIDs, cl_uint numDevices, cl_uint preferredDevice) : DeviceBaseClass(contextPtr, deviceIDs, numDevices, preferredDevice)
{
	deviceFunctionFile = "parserKernels.cl";
}


ClParser::~ClParser() 
{
}


ClParser::ClParser() :DeviceBaseClass() 
{
}

cl_uint ClParser::loadMetaData(ifstream * inFile)
{
	char * metaData = new char[CSV_ROW_LENGTH];
	
	//get number of records 
	inFile->getline(metaData, CSV_ROW_LENGTH, ',');
	inFile->getline(metaData, CSV_ROW_LENGTH, ',');
	csvFile.numRecords = stoi(metaData);

	//get number of columns
	inFile->getline(metaData, CSV_ROW_LENGTH, ',');
	inFile->getline(metaData, CSV_ROW_LENGTH, ',');
	csvFile.numVars = stoi(metaData);
	
	//get colunn headers
	inFile->getline(metaData, CSV_ROW_LENGTH, '\n');
	delete metaData;

	if (csvFile.numRecords < csvFile.batchSize) { csvFile.batchSize = csvFile.numRecords; }
	csvFile.numBatches = (csvFile.numRecords + csvFile.batchSize - 1) / csvFile.batchSize;

	return 0;
}

void ClParser::setupKernel(const char * funcName, int numVars, std::vector<cl_mem> varBuffers) 
{
	
	cl_kernel kernel = clCreateKernel(program, funcName, &errNum);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&unParsedBuffers[unParsedBuffers.size() - 1]);
	checkErr(errNum, "createKernelArg");
	for (int ptrIndex = 0; ptrIndex < numVars; ptrIndex++)
	{
		errNum = clSetKernelArg(kernel, numVars - ptrIndex, sizeof(cl_mem), (void *)&varBuffers[varBuffers.size() - 1 - ptrIndex]);
		checkErr(errNum, "setup kernel");
	}
	kernels.push_back(kernel);
}




void ClParser::setupTimeKernel(const char * funcName,cl_uint numThreadsInBatch,cl_mem * lineBreaks, cl_int startingIndex)
{
	cl_kernel kernel = clCreateKernel(program, funcName, &errNum);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&unParsedBuffers[unParsedBuffers.size() - 1]);
	checkErr(errNum, "createKernelArg");
	errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&csvFile.year[csvFile.year.size()-1]);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&csvFile.month[csvFile.month.size() - 1]);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&csvFile.day[csvFile.day.size() - 1]);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&csvFile.hour[csvFile.hour.size() - 1]);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&csvFile.minute[csvFile.minute.size() - 1]);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&numThreadsInBatch);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)lineBreaks);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 8, sizeof(cl_int), (void *)&startingIndex);
	checkErr(errNum, "setup kernel");
	kernels.push_back(kernel);
	enqeueKernel(timeCommmandQueue, kernels.size() - 1, numThreadsInBatch, preferredDevice,&events[events.size()-1]);
}


void ClParser::setupScoreKernel(const char * funcName, cl_uint numThreadsInBatch,cl_mem * lineBreaks,cl_int startingIndex)
{
	cl_kernel kernel = clCreateKernel(program, funcName, &errNum);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&unParsedBuffers[unParsedBuffers.size() - 1]);
	checkErr(errNum, "createKernelArg");
	errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&csvFile.envScore[csvFile.envScore.size() - 1]);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&csvFile.socialScore[csvFile.socialScore.size() - 1]);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&csvFile.sentiment[csvFile.sentiment.size() - 1]);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 4, sizeof(cl_int),(void*)&numThreadsInBatch);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 5, sizeof(cl_int), (void*)lineBreaks);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 6, sizeof(cl_int), (void*)&startingIndex);
	checkErr(errNum, "setup kernel");
	kernels.push_back(kernel);
	enqeueKernel(scoreCommandQueue, kernels.size() - 1, numThreadsInBatch, preferredDevice,&events[events.size()-1]);
}



void ClParser::setupTextKernel(const char * funcName, cl_uint numThreadsInBatch, cl_mem * lineBreaks,cl_int startingIndex)
{
	cl_kernel kernel = clCreateKernel(program, funcName, &errNum);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&unParsedBuffers[unParsedBuffers.size() - 1]);
	checkErr(errNum, "createKernelArg");
	errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&csvFile.tweet[csvFile.tweet.size() - 1]);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&csvFile.location[csvFile.location.size() - 1]);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&numThreadsInBatch);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)lineBreaks);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 5, sizeof(cl_int), (void*)&startingIndex);
	checkErr(errNum, "setup kernel");
	kernels.push_back(kernel);
	enqeueKernel(textCommandQueue, kernels.size() - 1, numThreadsInBatch, preferredDevice,&events[events.size()-1]);
}



void ClParser::parseTimeVars(cl_uint numThreadsInBatch, char * funcName, cl_mem * lineBreaks,cl_int startingIndex)
{
	if (csvFile.year.size() <2)
	{
		createBuffer(sizeof(cl_int), &csvFile.year, csvFile.numRecords, CL_MEM_READ_WRITE);
		createBuffer(sizeof(cl_int), &csvFile.month, csvFile.numRecords, CL_MEM_READ_WRITE);
		createBuffer(sizeof(cl_int), &csvFile.day, csvFile.numRecords, CL_MEM_READ_WRITE);
		createBuffer(sizeof(cl_int), &csvFile.hour, csvFile.numRecords, CL_MEM_READ_WRITE);
		createBuffer(sizeof(cl_int), &csvFile.minute, csvFile.numRecords, CL_MEM_READ_WRITE);
	}
	else 
	{
		cl_buffer_region region = { csvFile.batchSize*csvFile.year.size(),numThreadsInBatch*sizeof(cl_int) };

		cl_mem buffer = clCreateSubBuffer(csvFile.year[0], CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region,&errNum);
		checkErr(errNum, "create time sub buffer");
		csvFile.year.push_back(buffer);
		buffer = clCreateSubBuffer(csvFile.month[0], CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &errNum);
		checkErr(errNum, "create time sub buffer");
		csvFile.month.push_back(buffer);
		buffer = clCreateSubBuffer(csvFile.day[0], CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &errNum);
		checkErr(errNum, "create time sub buffer");
		csvFile.day.push_back(buffer);
		buffer = clCreateSubBuffer(csvFile.hour[0], CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &errNum);
		checkErr(errNum, "create time sub buffer");
		csvFile.hour.push_back(buffer);
		buffer = clCreateSubBuffer(csvFile.minute[0], CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &errNum);
		checkErr(errNum, "create time sub buffer");
		csvFile.minute.push_back(buffer);
	}
	setupTimeKernel(funcName, numThreadsInBatch,lineBreaks, startingIndex);
}

void ClParser::parseScoreVars(cl_uint numThreadsInBatch, char * funcName,cl_mem * lineBreaks,cl_int startingIndex)
{
	if (csvFile.envScore.size() == 0)
	{
		createBuffer(sizeof(cl_int), &csvFile.envScore, csvFile.numRecords, CL_MEM_READ_WRITE);
		createBuffer(sizeof(cl_int), &csvFile.socialScore, csvFile.numRecords, CL_MEM_READ_WRITE);
		createBuffer(sizeof(cl_int), &csvFile.sentiment, csvFile.numRecords, CL_MEM_READ_WRITE);
	}
	else 
	{
		cl_buffer_region region = { csvFile.batchSize*csvFile.envScore.size(),numThreadsInBatch * sizeof(cl_int) };

		cl_mem buffer = clCreateSubBuffer(csvFile.envScore[0], CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &errNum);
		checkErr(errNum, "create score sub buffer");
		csvFile.envScore.push_back(buffer);
		buffer = clCreateSubBuffer(csvFile.socialScore[0], CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &errNum);
		checkErr(errNum, "create score sub buffer");
		csvFile.socialScore.push_back(buffer);
		buffer = clCreateSubBuffer(csvFile.sentiment[0], CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &errNum);
		checkErr(errNum, "create score sub buffer");
		csvFile.sentiment.push_back(buffer);
	}
	setupScoreKernel(funcName, numThreadsInBatch, lineBreaks,startingIndex);
}

void ClParser::parseTextVars(cl_uint numThreadsInBatch, char * funcName,cl_mem * lineBreaks,cl_int startingIndex)
{
	if (csvFile.tweet.size() == 0) 
	{
		createBuffer(sizeof(cl_char), &csvFile.tweet, csvFile.numRecords*TEXT_OFFSETS[0], CL_MEM_READ_WRITE);
		createBuffer(sizeof(cl_char), &csvFile.location, csvFile.numRecords*TEXT_OFFSETS[1], CL_MEM_READ_WRITE);

	}
	else 
	{
		cl_buffer_region twitterRegion = { csvFile.batchSize*csvFile.tweet.size()*TEXT_OFFSETS[0],numThreadsInBatch * sizeof(cl_char)*TEXT_OFFSETS[0] };
		cl_mem buffer = clCreateSubBuffer(csvFile.tweet[0], CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &twitterRegion, &errNum);
		checkErr(errNum, "create text sub buffer");
		csvFile.tweet.push_back(buffer);

		cl_buffer_region locationRegion = { csvFile.batchSize*csvFile.location.size()*TEXT_OFFSETS[1],numThreadsInBatch * sizeof(cl_char)*TEXT_OFFSETS[1] };
		buffer = clCreateSubBuffer(csvFile.location[0], CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &locationRegion, &errNum);
		checkErr(errNum, "create text sub buffer");
		csvFile.location.push_back(buffer);

	}
	setupTextKernel(funcName, numThreadsInBatch, lineBreaks, startingIndex);
}




void ClParser::parseVars(cl_uint numThreadsInBatch, cl_mem * lineBreaks, cl_int startingIndex)
{
	parseTimeVars(numThreadsInBatch, "parse_timestamp", lineBreaks, startingIndex);
	parseScoreVars(numThreadsInBatch, "parse_scores",lineBreaks,startingIndex);
	parseTextVars(numThreadsInBatch, "parse_text",lineBreaks,startingIndex);

}

void ClParser::findLineBreaks(cl_mem * lineBreaks, cl_int batchSize, cl_int * newIndex) {
	
	cl_int test = batchSize*CSV_ROW_LENGTH;

	cl_kernel kernel = clCreateKernel(program, "new_line", &errNum);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&unParsedBuffers[unParsedBuffers.size() - 1]);
	checkErr(errNum, "createKernelArg");
	errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)lineBreaks);
	checkErr(errNum, "createKernelArg");
	errNum = clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&test);
	checkErr(errNum, "setup kernel");
	kernels.push_back(kernel);
	enqeueKernel(textCommandQueue, kernels.size() - 1, test, 250, preferredDevice, &events[events.size()-1]);
	clWaitForEvents(events.size(), events.data());
	clFinish(queues[textCommandQueue]);
	*newIndex = 0;
	cl_uint offset = 250;
	cl_uint groupSize = 25;
	cl_uint numRecords = 0;
	cl_uint numThreads = batchSize*10;
	clFinish(queues[textCommandQueue]);
	
	
	cl_uint maxValNum = batchSize*CSV_ROW_LENGTH;
	cl_kernel globalKerne3l = clCreateKernel(program, "collapseValsGlobal", &errNum);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(globalKerne3l, 0, sizeof(cl_mem), (void *)lineBreaks);
	checkErr(errNum, "createKernelArg");
	errNum = clSetKernelArg(globalKerne3l, 2, sizeof(cl_int), (void*)&maxValNum);
	checkErr(errNum, "setup kernel");
	kernels.push_back(globalKerne3l);
	while (*newIndex < batchSize - 1) // & offset < csvFile.numRecords*CSV_ROW_LENGTH + 1 / 2)
	{
		errNum = clSetKernelArg(globalKerne3l, 1, sizeof(cl_int), (void*)&offset);
		checkErr(errNum, "setup kernel");
		errNum = clSetKernelArg(globalKerne3l, 3, sizeof(cl_int), (void*)&groupSize);
		checkErr(errNum, "setup kernel");
		kernels.push_back(globalKerne3l);
		enqeueKernel(textCommandQueue, kernels.size() - 1, numThreads, 256, preferredDevice, &events[events.size() - 1]);
	
		offset = offset * 2;
		groupSize = groupSize * 2;
		clWaitForEvents(events.size(), events.data());
	//	clFinish(queues[textCommandQueue]);
		copyDataToHost(textCommandQueue, *lineBreaks, newIndex, 1);
//		clFinish(queues[textCommandQueue]);
//		cout << "numRecords: " << *newIndex << endl;
	}
	//copyDataToHost(textCommandQueue, *lineBreaks, tempResults, 1);
	clEnqueueReadBuffer(queues[textCommandQueue], *lineBreaks, CL_TRUE, sizeof(cl_int)*batchSize, sizeof(cl_int), newIndex, 1, &events[events.size() - 1], NULL);
	//cout << " new index " <<  *newIndex << endl;
	clFinish(queues[textCommandQueue]);
	clWaitForEvents(events.size(), events.data());
	//releaseKernels();
	if (debug) {
	//	printLineSearchDebug(lineBreaks);
	}

}

void ClParser::printLineSearchDebug(cl_mem * lineBreaks)
{


	int * debugPrint = (int*)malloc(sizeof(int)*csvFile.batchSize);

	clFinish(queues[textCommandQueue]);



	copyDataToHost(textCommandQueue, *lineBreaks, &(debugPrint[0]), csvFile.batchSize);


	clFinish(queues[textCommandQueue]);

	
	for (int i = 0; i < 100; i++) {
			std::cout << "tempResult " << debugPrint[i] << "  ";
			std::cout << "index: " << i << endl;
		
		}
		
	
	free(debugPrint);
}

int ClParser::twice(ifstream *inFile, char * unParsedData, cl_uint batchSize) {

	inFile->read(unParsedData, batchSize*CSV_ROW_LENGTH);
	return inFile->gcount();
}


void ClParser::processCSVFile(ifstream *inFile, char * unParsedRecords)
{
	cl_uint batchSize = csvFile.batchSize;
	int unParsedLocation = 0;
	cl_int newIndex = 0;
	cl_int copyIndex = 0;
	future <int> futureVal;
	bool stillReading = true;
	futureVal = std::async(&ClParser::twice, this, inFile, &unParsedRecords[unParsedLocation], batchSize);
	
	for (cl_uint batchNum = 0; batchNum < csvFile.numBatches; batchNum++) {


	
			createBuffer(sizeof(cl_int), &lineBreaks, batchSize*CSV_ROW_LENGTH, CL_MEM_READ_WRITE);
			createBuffer(sizeof(cl_char), &unParsedBuffers, batchSize*CSV_ROW_LENGTH, CL_MEM_READ_ONLY);


		
		if (stillReading) {
			unParsedLocation += futureVal.get();
			

	

		}


				cl_event newEvent;
				copyDataToBuffer(memcpyCommandQueue, &(unParsedBuffers[unParsedBuffers.size() - 1]), &unParsedRecords[copyIndex], batchSize*CSV_ROW_LENGTH, &newEvent);  // copy input data to device on a dedicated queue

				clWaitForEvents(1, &newEvent);

				if (!inFile->eof()) {
					futureVal = std::async(&ClParser::twice, this, inFile, &unParsedRecords[unParsedLocation], batchSize);
				}
				else {
					stillReading = false;
				}


		
			
		
			

				findLineBreaks(&lineBreaks[lineBreaks.size() - 1], batchSize, &newIndex);

				/*
				int * debugPrint = (int*)malloc(sizeof(int)*csvFile.batchSize*CSV_ROW_LENGTH);




				copyDataToHost(textCommandQueue, lineBreaks[lineBreaks.size() - 1], &(debugPrint[0]), csvFile.batchSize*CSV_ROW_LENGTH);
				clWaitForEvents(1, &events[events.size() - 1]);
				clFinish(queues[textCommandQueue]);
		/*		for (int i = 0; i < batchSize; i++) {
					if (unParsedRecords[debugPrint[i] + copyIndex] != '\n') {
						int a = debugPrint[i];
						cout << "mismatch, index " << i + batchNum*batchSize << ", value " << debugPrint[i] + copyIndex << "," << endl << unParsedRecords[debugPrint[i] - 1 + copyIndex] << unParsedRecords[debugPrint[i] + copyIndex] << unParsedRecords[debugPrint[i] + 1 + copyIndex] << endl;
					}
					if (debugPrint[i] == 0) {
						cout << "zero warning" << i << endl;
					}
				}
				for (int i = 0; i < 10; i++) {
					cout << debugPrint[i] << "why";
					cout << debugPrint[i] - 10 << endl;



				}


				
				free(debugPrint);




				*/


			
			
			copyIndex += newIndex + 1;
	

		parseVars(csvFile.batchSize, &lineBreaks[lineBreaks.size() - 1], copyIndex);


	
	
		}
}

void ClParser::allocateMemory()
{
	unParsedRecords = (char*)malloc(sizeof(char)*csvFile.numRecords*CSV_ROW_LENGTH);

}

void ClParser::releaseMemory()
{
	free(unParsedRecords);
}


void ClParser::BuffersToHost(cl_int * inputPtr, std::vector<cl_mem> * buffers, cl_uint queueNum)
{
	
	for (cl_uint batchNum = 0; batchNum < csvFile.numBatches - 1; batchNum++)
	{
		copyDataToHost(queueNum, (*buffers)[batchNum], &(inputPtr[csvFile.batchSize*batchNum]), csvFile.batchSize);
	}
	// for final batch that may be truncated
	int tempBatchSize = csvFile.numRecords - (csvFile.batchSize)*(csvFile.numBatches - 1);
	cout << tempBatchSize << endl;
	int batchNum = csvFile.numBatches - 1;
	copyDataToHost(queueNum, (*buffers)[batchNum], &(inputPtr[csvFile.batchSize*batchNum]), tempBatchSize);
}

void ClParser::BuffersToHost(cl_char * inputPtr, std::vector<cl_mem> * buffers, cl_uint queueNum, cl_uint textOffest)
{
	for (cl_uint batchNum = 0; batchNum < csvFile.numBatches - 1; batchNum++)
	{
		copyDataToHost(queueNum, (*buffers)[batchNum], &(inputPtr[csvFile.batchSize*batchNum*textOffest]), csvFile.batchSize*textOffest);
	}
	// for final batch that may be truncated
	int tempBatchSize = csvFile.numRecords - (csvFile.batchSize)*(csvFile.numBatches - 1);
	cout << tempBatchSize << endl;
	int batchNum = csvFile.numBatches - 1;
	copyDataToHost(queueNum, (*buffers)[batchNum], &(inputPtr[csvFile.batchSize*batchNum*textOffest]), tempBatchSize*textOffest);
}



void ClParser::printOutput()
{
	clWaitForEvents(events.size(), events.data());
	
	cl_int * timePtrs[5];
	for (int i = 0; i < 5; i++) 
	{
		timePtrs[i] = (cl_int*)malloc(sizeof(cl_int)*csvFile.numRecords);
	}

	BuffersToHost(timePtrs[0], &csvFile.year, timeCommmandQueue);
	BuffersToHost(timePtrs[1], &csvFile.month, timeCommmandQueue);

	BuffersToHost(timePtrs[2], &csvFile.day, timeCommmandQueue);
   BuffersToHost(timePtrs[3], &csvFile.hour, timeCommmandQueue);
	BuffersToHost(timePtrs[4], &csvFile.minute, timeCommmandQueue);
	
	cl_int * sentPtr = (cl_int*)malloc(sizeof(cl_int)*csvFile.numRecords);
	BuffersToHost(sentPtr, &csvFile.sentiment, scoreCommandQueue);
	cl_int * envPtr = (cl_int*)malloc(sizeof(cl_int)*csvFile.numRecords);
	BuffersToHost(envPtr, &csvFile.envScore, scoreCommandQueue);
	cl_int * socialPtr = (cl_int*)malloc(sizeof(cl_int)*csvFile.numRecords);
	BuffersToHost(socialPtr, &csvFile.socialScore, scoreCommandQueue);
		clFinish(queues[timeCommmandQueue]);
	cl_char * textPtr = (cl_char*)malloc(sizeof(cl_char)*csvFile.numRecords*TEXT_OFFSETS[1]);
	BuffersToHost(textPtr, &csvFile.location, textCommandQueue, TEXT_OFFSETS[1]);
	cl_char * tweetPtr = (cl_char*)malloc(sizeof(cl_char)*csvFile.numRecords*TEXT_OFFSETS[0]);
	BuffersToHost(tweetPtr, &csvFile.tweet, textCommandQueue, TEXT_OFFSETS[0]);
	
	

	for (int i = 0; i < csvFile.numRecords; i++) {
		if (timePtrs[0][i] != 2016) {
			cout << "err at index: " << i << ", value equals: " << timePtrs[0][i] << endl;
			if (timePtrs[1][i] != 12) {
				cout << "err at index: " << i << ", value equals: " << timePtrs[1][i] << endl;


			}

		}


	}


	//for (int i = csvFile.numRecords - 8; i < csvFile.numRecords; i++)
	for(int i = 0; i < 10; i++)
	{
		cout << "year " << timePtrs[0][i] << ", month " << timePtrs[1][i] << ", day " << timePtrs[2][i] << ", hour "
			<< timePtrs[3][i] << " ,minute " << timePtrs[4][i] << endl;
		cout << "sentiment " << sentPtr[i] << ", envScore " << envPtr[i] << ", socialScore " << socialPtr[i] << endl;
		
		cout << "twitter text: ";
		for (int j = 0; j < 280; j++)
		{
			cout << tweetPtr[j+i*TEXT_OFFSETS[0]];
		}
		cout << endl;

		cout << "location: ";
		for (int j = 0; j < 20; j++)
		{
			cout << textPtr[j+i*20];
		}
		cout << endl;
		cout << endl;
		cout << endl;
	}
	
	for (int timeIndex = 0; timeIndex < 5; timeIndex++) 
	{
		free(timePtrs[timeIndex]);
	}

	
	free(sentPtr);
	free(envPtr);
	free(socialPtr);
	free(textPtr);
	free(tweetPtr);
}
void  ClParser::parseFile(char *inputFile)
{
	this->inputFile = inputFile;
	ifstream inFile(inputFile);
	loadMetaData(&inFile);
	createProgram(1, deviceIDs, preferredDevice);
	allocateMemory();

	for (int queueNum = 0; queueNum < 4; queueNum++) 
	{
		createCommandQueue(preferredDevice);
	}

	//for (cl_uint batchNum = 0; batchNum < csvFile.numBatches; batchNum++)
	//{
	//	if (batchNum == 0) {
			processCSVFile(&inFile, unParsedRecords);
		//}
	//}

	inFile.close();
	
//	if (debug) { printOutput(); }


	releaseMemory();

	
}




char * ClParser::getInputFile()
{
	return inputFile;

}