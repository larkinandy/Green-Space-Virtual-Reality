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




void ClParser::setupTimeKernel(const char * funcName,cl_uint numThreadsInBatch)
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
	kernels.push_back(kernel);
	enqeueKernel(timeCommmandQueue, kernels.size() - 1, numThreadsInBatch, preferredDevice, &events[newEventNum]);
}


void ClParser::setupScoreKernel(const char * funcName, cl_uint numThreadsInBatch)
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
	kernels.push_back(kernel);
	enqeueKernel(scoreCommandQueue, kernels.size() - 1, numThreadsInBatch, preferredDevice,&events[newEventNum]);
}



void ClParser::setupTextKernel(const char * funcName, cl_uint numThreadsInBatch)
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
	kernels.push_back(kernel);
	enqeueKernel(textCommandQueue, kernels.size() - 1, numThreadsInBatch, preferredDevice,&events[newEventNum]);
}



void ClParser::parseTimeVars(cl_uint numThreadsInBatch, char * funcName)
{
	if (csvFile.year.size() == 0)
	{
		createBuffer(sizeof(cl_int), &csvFile.year, csvFile.numRecords, CL_MEM_WRITE_ONLY);
		createBuffer(sizeof(cl_int), &csvFile.month, csvFile.numRecords, CL_MEM_WRITE_ONLY);
		createBuffer(sizeof(cl_int), &csvFile.day, csvFile.numRecords, CL_MEM_WRITE_ONLY);
		createBuffer(sizeof(cl_int), &csvFile.hour, csvFile.numRecords, CL_MEM_WRITE_ONLY);
		createBuffer(sizeof(cl_int), &csvFile.minute, csvFile.numRecords, CL_MEM_WRITE_ONLY);
	}
	else 
	{
		cl_buffer_region region = { csvFile.batchSize*csvFile.year.size(),numThreadsInBatch*sizeof(cl_int) };

		cl_mem buffer = clCreateSubBuffer(csvFile.year[0], CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region,&errNum);
		checkErr(errNum, "create time sub buffer");
		csvFile.year.push_back(buffer);
		buffer = clCreateSubBuffer(csvFile.month[0], CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &errNum);
		checkErr(errNum, "create time sub buffer");
		csvFile.month.push_back(buffer);
		buffer = clCreateSubBuffer(csvFile.day[0], CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &errNum);
		checkErr(errNum, "create time sub buffer");
		csvFile.day.push_back(buffer);
		buffer = clCreateSubBuffer(csvFile.hour[0], CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &errNum);
		checkErr(errNum, "create time sub buffer");
		csvFile.hour.push_back(buffer);
		buffer = clCreateSubBuffer(csvFile.minute[0], CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &errNum);
		checkErr(errNum, "create time sub buffer");
		csvFile.minute.push_back(buffer);
	}
	setupTimeKernel(funcName, numThreadsInBatch);
}

void ClParser::parseScoreVars(cl_uint numThreadsInBatch, char * funcName)
{
	if (csvFile.envScore.size() == 0)
	{
		createBuffer(sizeof(cl_int), &csvFile.envScore, csvFile.numRecords, CL_MEM_WRITE_ONLY);
		createBuffer(sizeof(cl_int), &csvFile.socialScore, csvFile.numRecords, CL_MEM_WRITE_ONLY);
		createBuffer(sizeof(cl_int), &csvFile.sentiment, csvFile.numRecords, CL_MEM_WRITE_ONLY);
	}
	else 
	{
		cl_buffer_region region = { csvFile.batchSize*csvFile.envScore.size(),numThreadsInBatch * sizeof(cl_int) };

		cl_mem buffer = clCreateSubBuffer(csvFile.envScore[0], CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &errNum);
		checkErr(errNum, "create score sub buffer");
		csvFile.envScore.push_back(buffer);
		buffer = clCreateSubBuffer(csvFile.socialScore[0], CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &errNum);
		checkErr(errNum, "create score sub buffer");
		csvFile.socialScore.push_back(buffer);
		buffer = clCreateSubBuffer(csvFile.sentiment[0], CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &errNum);
		checkErr(errNum, "create score sub buffer");
		csvFile.sentiment.push_back(buffer);
	}
	setupScoreKernel(funcName, numThreadsInBatch);
}

void ClParser::parseTextVars(cl_uint numThreadsInBatch, char * funcName)
{
	if (csvFile.tweet.size() == 0) 
	{
		createBuffer(sizeof(cl_char), &csvFile.tweet, csvFile.numRecords*TEXT_OFFSETS[0], CL_MEM_WRITE_ONLY);
		createBuffer(sizeof(cl_char), &csvFile.location, csvFile.numRecords*TEXT_OFFSETS[1], CL_MEM_WRITE_ONLY);

	}
	else 
	{
		cl_buffer_region twitterRegion = { csvFile.batchSize*csvFile.tweet.size()*TEXT_OFFSETS[0],numThreadsInBatch * sizeof(cl_char)*TEXT_OFFSETS[0] };
		cl_mem buffer = clCreateSubBuffer(csvFile.tweet[0], CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &twitterRegion, &errNum);
		checkErr(errNum, "create text sub buffer");
		csvFile.tweet.push_back(buffer);

		cl_buffer_region locationRegion = { csvFile.batchSize*csvFile.location.size()*TEXT_OFFSETS[1],numThreadsInBatch * sizeof(cl_char)*TEXT_OFFSETS[1] };
		buffer = clCreateSubBuffer(csvFile.location[0], CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &locationRegion, &errNum);
		checkErr(errNum, "create text sub buffer");
		csvFile.location.push_back(buffer);

	}
	setupTextKernel(funcName, numThreadsInBatch);
}




void ClParser::parseVars(cl_uint numThreadsInBatch)
{
	parseTimeVars(numThreadsInBatch, "parse_timestamp");
	parseScoreVars(numThreadsInBatch, "parse_scores");
	parseTextVars(numThreadsInBatch, "parse_text");

}

void ClParser::findLineBreaks(char * rawData, cl_mem * lineBreaks) {
	createBuffer(sizeof(cl_char), &unParsedBuffers, csvFile.numRecords*CSV_ROW_LENGTH, CL_MEM_READ_ONLY);

	cl_event newEvent;
	copyDataToBuffer(memcpyCommandQueue, &(unParsedBuffers[unParsedBuffers.size() - 1]), &(rawData[0]), csvFile.numRecords*CSV_ROW_LENGTH, &newEvent);  // copy input data to device on a dedicated queue

	cl_int test = csvFile.numRecords*CSV_ROW_LENGTH;

	cl_kernel kernel = clCreateKernel(program, "new_line", &errNum);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&unParsedBuffers[unParsedBuffers.size() - 1]);
	checkErr(errNum, "createKernelArg");
	errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)lineBreaks);
	checkErr(errNum, "createKernelArg");
	errNum = clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&test);
	checkErr(errNum, "setup kernel");
	kernels.push_back(kernel);
	enqeueKernel(textCommandQueue, kernels.size() - 1, test, 250, preferredDevice, &events[0]);

	clFinish(queues[textCommandQueue]);

	cl_int * tempResults = (cl_int*)malloc(sizeof(cl_int) * 1);
	cl_uint offset = 250;
	cl_uint groupSize = 25;
	cl_uint numRecords = 0;
	cl_uint numThreads = 100000;
	cl_uint maxValNum = csvFile.numRecords*CSV_ROW_LENGTH;
	cl_kernel globalKerne3l = clCreateKernel(program, "collapseValsGlobal", &errNum);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(globalKerne3l, 0, sizeof(cl_mem), (void *)lineBreaks);
	checkErr(errNum, "createKernelArg");
	errNum = clSetKernelArg(globalKerne3l, 2, sizeof(cl_int), (void*)&maxValNum);
	checkErr(errNum, "setup kernel");
	while (numRecords < csvFile.numRecords - 1 & offset < csvFile.numRecords*CSV_ROW_LENGTH + 1 / 2)
	{
		errNum = clSetKernelArg(globalKerne3l, 1, sizeof(cl_int), (void*)&offset);
		checkErr(errNum, "setup kernel");
		errNum = clSetKernelArg(globalKerne3l, 3, sizeof(cl_int), (void*)&groupSize);
		checkErr(errNum, "setup kernel");
		kernels.push_back(globalKerne3l);
		enqeueKernel(textCommandQueue, kernels.size() - 1, numThreads, 256, preferredDevice, &events[events.size() - 1]);
		copyDataToHost(textCommandQueue, *lineBreaks, tempResults, 1);
		offset = offset * 2;
		groupSize = groupSize * 2;
	}

	cout << tempResults[0] << endl;

	free(tempResults);

	if (debug) {
	//	printLineSearchDebug(lineBreaks);
	}
	

}

void ClParser::printLineSearchDebug(cl_mem * lineBreaks)
{


	int * debugPrint = (int*)malloc(sizeof(int)*csvFile.numRecords*CSV_ROW_LENGTH);

	clFinish(queues[textCommandQueue]);



	copyDataToHost(textCommandQueue, *lineBreaks, &(debugPrint[0]), csvFile.numRecords*CSV_ROW_LENGTH);


	clFinish(queues[textCommandQueue]);


	for (int i = 1; i < csvFile.numRecords; i++) {
		if (debugPrint[i] < debugPrint[i - 1] & debugPrint[i] != 0 | (debugPrint[i] > 100 & debugPrint[i] < i)) {
			for (int j = i - 10; j < i + 10; j++) {
				cout << "index: " << j << " value: " << debugPrint[j] << endl;
			}
			cout << "tempResult " << debugPrint[i] << "  ";
			cout << "index: " << i << endl;
			if (debugPrint[i] - debugPrint[i - 1] > 350) {
				cout << "error: " << debugPrint[i] << " , " << debugPrint[i - 1] << endl;
			}
		}
		

	}
	for (int i = 0; i < 10; i++) {
		cout << "index: " << i << "value: " << debugPrint[i] << endl;

	}

	free(debugPrint);
}


void ClParser::processCSVBatch(ifstream *inFile, char * unParsedRecords, cl_uint batchNum)
{
	cl_uint minRecordNum = batchNum*csvFile.batchSize;
	cl_uint maxRecordNum = min((batchNum + 1)*csvFile.batchSize, csvFile.numRecords);

	/*
	for (int recordNum = minRecordNum; recordNum < maxRecordNum; recordNum++)
	{
		inFile->getline(&(unParsedRecords[recordNum*CSV_ROW_LENGTH]), CSV_ROW_LENGTH, '\n');
	}*/
	
	inFile->read(unParsedRecords, csvFile.numRecords*CSV_ROW_LENGTH);
	
	createBuffer(sizeof(cl_int), &csvFile.year, csvFile.numRecords*CSV_ROW_LENGTH, CL_MEM_READ_WRITE);

	findLineBreaks(unParsedRecords, &csvFile.year[0]);

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

	cl_char * textPtr = (cl_char*)malloc(sizeof(cl_char)*csvFile.numRecords*TEXT_OFFSETS[1]);
	BuffersToHost(textPtr, &csvFile.location, textCommandQueue, TEXT_OFFSETS[1]);
	cl_char * tweetPtr = (cl_char*)malloc(sizeof(cl_char)*csvFile.numRecords*TEXT_OFFSETS[0]);
	BuffersToHost(tweetPtr, &csvFile.tweet, textCommandQueue, TEXT_OFFSETS[0]);




	for (int i = csvFile.numRecords - 8; i < csvFile.numRecords; i++)
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

	for (cl_uint batchNum = 0; batchNum < csvFile.numBatches; batchNum++)
	{
		if (batchNum == 0) {
			processCSVBatch(&inFile, unParsedRecords, batchNum);
		}
	}

	inFile.close();
	
	//if (debug) { printOutput(); }


	releaseMemory();

	
}




char * ClParser::getInputFile()
{
	return inputFile;

}