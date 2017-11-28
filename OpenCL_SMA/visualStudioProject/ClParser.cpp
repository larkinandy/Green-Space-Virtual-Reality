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




void ClParser::setupTimeKernel(const char * funcName, cl_uint numVars,cl_uint numThreadsInBatch)
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
	kernels.push_back(kernel);
	enqeueKernel(timeCommmandQueue, kernels.size() - 1, numThreadsInBatch, preferredDevice, &events[newEventNum]);
}


void ClParser::setupScoreKernel(const char * funcName, cl_uint numVars, cl_uint numThreadsInBatch)
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
	kernels.push_back(kernel);
	enqeueKernel(scoreCommandQueue, kernels.size() - 1, numThreadsInBatch, preferredDevice);
}



void ClParser::setupTextKernel(const char * funcName, cl_uint numVars, cl_uint numThreadsInBatch)
{
	cl_kernel kernel = clCreateKernel(program, funcName, &errNum);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&unParsedBuffers[unParsedBuffers.size() - 1]);
	checkErr(errNum, "createKernelArg");
	errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&csvFile.tweet[csvFile.tweet.size() - 1]);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&csvFile.location[csvFile.location.size() - 1]);
	checkErr(errNum, "setup kernel");
	kernels.push_back(kernel);
	enqeueKernel(textCommandQueue, kernels.size() - 1, numThreadsInBatch, preferredDevice);
}



void ClParser::parseTimeVars(cl_uint numThreadsInBatch, char * funcName)
{
	createBuffer(sizeof(cl_int), &csvFile.year, numThreadsInBatch, CL_MEM_WRITE_ONLY);
	createBuffer(sizeof(cl_int), &csvFile.month, numThreadsInBatch, CL_MEM_WRITE_ONLY);
	createBuffer(sizeof(cl_int), &csvFile.day, numThreadsInBatch, CL_MEM_WRITE_ONLY);
	createBuffer(sizeof(cl_int), &csvFile.hour, numThreadsInBatch, CL_MEM_WRITE_ONLY);
	createBuffer(sizeof(cl_int), &csvFile.minute, numThreadsInBatch, CL_MEM_WRITE_ONLY);
	setupTimeKernel(funcName, 5,numThreadsInBatch);
}

void ClParser::parseScoreVars(cl_uint numThreadsInBatch, char * funcName)
{
	createBuffer(sizeof(cl_int), &csvFile.envScore, numThreadsInBatch, CL_MEM_WRITE_ONLY);
	createBuffer(sizeof(cl_int), &csvFile.socialScore, numThreadsInBatch, CL_MEM_WRITE_ONLY);
	createBuffer(sizeof(cl_int), &csvFile.sentiment, numThreadsInBatch, CL_MEM_WRITE_ONLY);
	setupScoreKernel(funcName, 3, numThreadsInBatch);
}

void ClParser::parseTextVars(cl_uint numThreadsInBatch, char * funcName)
{
	createBuffer(sizeof(cl_char), &csvFile.tweet, numThreadsInBatch*TEXT_OFFSETS[0], CL_MEM_WRITE_ONLY);
	createBuffer(sizeof(cl_char), &csvFile.location, numThreadsInBatch*TEXT_OFFSETS[1], CL_MEM_WRITE_ONLY);
	setupTextKernel(funcName, 2, numThreadsInBatch);
}




void ClParser::parseVars(cl_uint numThreadsInBatch)
{
	parseTimeVars(numThreadsInBatch, "parse_timestamp");
	parseScoreVars(numThreadsInBatch, "parse_scores");
	parseTextVars(numThreadsInBatch, "parse_text");

}


void ClParser::processCSVBatch(ifstream *inFile, char * unParsedRecords, cl_uint batchNum)
{
	cl_uint minRecordNum = batchNum*csvFile.batchSize;
	cl_uint maxRecordNum = min((batchNum + 1)*csvFile.batchSize, csvFile.numRecords);

	for (int recordNum = minRecordNum; recordNum < maxRecordNum; recordNum++)
	{
		inFile->getline(&(unParsedRecords[recordNum*CSV_ROW_LENGTH]), CSV_ROW_LENGTH, '\n');
	}
	
	createBuffer(sizeof(cl_char), &unParsedBuffers, csvFile.batchSize*CSV_ROW_LENGTH, CL_MEM_READ_ONLY);
	
	cl_event newEvent;
	copyDataToBuffer(memcpyCommandQueue, &(unParsedBuffers[unParsedBuffers.size()-1]), &(unParsedRecords[minRecordNum*CSV_ROW_LENGTH]), (maxRecordNum - minRecordNum)*CSV_ROW_LENGTH, &newEvent);  // copy input data to device on a dedicated queue
	cl_uint numThreadsInBatch = maxRecordNum - minRecordNum;
	newEventNum = events.size() - 1;
	parseVars(numThreadsInBatch);
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



	
cl_int * yearPtr = (cl_int*)malloc(sizeof(cl_int)*csvFile.numRecords);
BuffersToHost(yearPtr, &csvFile.year,timeCommmandQueue);
cl_int * monthPtr = (cl_int*)malloc(sizeof(cl_int)*csvFile.numRecords);
BuffersToHost(monthPtr, &csvFile.month, timeCommmandQueue);
cl_int * dayPtr = (cl_int*)malloc(sizeof(cl_int)*csvFile.numRecords);
BuffersToHost(dayPtr, &csvFile.day, timeCommmandQueue);
cl_int * hourPtr = (cl_int*)malloc(sizeof(cl_int)*csvFile.numRecords);
BuffersToHost(hourPtr, &csvFile.hour, timeCommmandQueue);
cl_int * minutePtr = (cl_int*)malloc(sizeof(cl_int)*csvFile.numRecords);
BuffersToHost(minutePtr, &csvFile.minute, timeCommmandQueue);


cl_int * sentPtr = (cl_int*)malloc(sizeof(cl_int)*csvFile.numRecords);
BuffersToHost(sentPtr, &csvFile.sentiment,scoreCommandQueue);
cl_int * envPtr = (cl_int*)malloc(sizeof(cl_int)*csvFile.numRecords);
BuffersToHost(envPtr, &csvFile.envScore, scoreCommandQueue);
cl_int * socialPtr = (cl_int*)malloc(sizeof(cl_int)*csvFile.numRecords);
BuffersToHost(socialPtr, &csvFile.socialScore, scoreCommandQueue);

cl_char * textPtr = (cl_char*)malloc(sizeof(cl_char)*csvFile.numRecords*TEXT_OFFSETS[1]);
BuffersToHost(textPtr, &csvFile.location, textCommandQueue,TEXT_OFFSETS[1]);
cl_char * tweetPtr = (cl_char*)malloc(sizeof(cl_char)*csvFile.numRecords*TEXT_OFFSETS[0]);
BuffersToHost(tweetPtr, &csvFile.tweet, textCommandQueue, TEXT_OFFSETS[0]);


	/*
	for (int i = 0; i < queues.size() - 1; i++)
	{
		errNum = clFinish(queues[i]);
		checkErr(errNum, "finish queue");
	*/
	for (int i = csvFile.numRecords - 8; i < csvFile.numRecords; i++)
//		for (int i = 0; i < 5; i++)
	{
		cout << "year " << yearPtr[i] << ", month " << monthPtr[i] << ", day " << dayPtr[i] << ", hour " << hourPtr[i] << " ,minute " << minutePtr[i] << endl;
		cout << "sentiment " << sentPtr[i] << ", envScore " << envPtr[i] << ", socialScore " << socialPtr[i] << endl;

				cout << "twitter text: ";
				for (int j = 0; j < 100; j++)
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

	free(yearPtr);
	free(monthPtr);
	free(dayPtr);
	free(hourPtr);
	free(minutePtr);
	free(sentPtr);
	free(envPtr);
	free(socialPtr);
	free(textPtr);
	free(tweetPtr);

		
}
void  ClParser::parseFile(char *inputFile)
{
	ifstream inFile(inputFile);
	loadMetaData(&inFile);
	createProgram(1, deviceIDs, preferredDevice);
	allocateMemory();

	createCommandQueue(preferredDevice);
	createCommandQueue(preferredDevice);
	createCommandQueue(preferredDevice);
	createCommandQueue(preferredDevice);

	for (cl_uint batchNum = 0; batchNum < csvFile.numBatches; batchNum++)
	{
		processCSVBatch(&inFile, unParsedRecords, batchNum);
	}

	inFile.close();
	//clWaitForEvents(events.size(), events.data());

	/*
	for (int i = 0; i < queues.size() ; i++)
	{
		errNum = clFinish(queues[i]);
		cout << "queue num " << i;
		checkErr(errNum, "finish queue");
	}*/
	
	releaseMemory();
	//if(debug) {printOutput();}
	
}




char * ClParser::getInputFile()
{
	return inputFile;

}