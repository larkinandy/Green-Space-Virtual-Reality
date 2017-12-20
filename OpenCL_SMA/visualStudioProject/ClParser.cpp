/* ClParser.hpp
* Header file for class that parses csv files on GPU devices using OpenCL
* Author: Andrew Larkin
* December 5, 2017
*/


#include "ClParser.hpp"


ClParser::ClParser(cl_context * contextPtr, cl_device_id * deviceIDs, cl_uint numDevices, cl_uint preferredDevice) : 
	DeviceBaseClass(contextPtr, deviceIDs, numDevices, preferredDevice)
{
	deviceFunctionFile = "parserKernels.cl";
}

ClParser::~ClParser() 
{
	cleanupCheck.get();
}


ClParser::ClParser() :DeviceBaseClass() 
{
}


void ClParser::allocateMemory()
{
	unParsedRecords = (char*)malloc(sizeof(char)*csvFile.numRecords*csvFile.CSV_ROW_LENGTH);
	// todo: allocation for 1 million records takes almost 100ms.  Look into smaller size allocation, 
	// repeatedly using a smaller set
}



// release OpenCL objects and memory
void ClParser::cleanup()
{
	// todo: look into asynchronously releasing buffers, as they are no longer influencing any
	// other OpenCL operation after they return their results
	releaseResults();
	/*releaseProgram();
	releaseCommandQueues();
	releaseKernels();
	releaseEvents();
	*/
}

// delete results.  Only run when results are no longer needed by any classes in the OpenCL program
void ClParser::releaseResults()
{
	releaseBuffers(&csvFile.year);
	releaseBuffers(&csvFile.hour);
	releaseBuffers(&csvFile.day);
	releaseBuffers(&csvFile.month);
	releaseBuffers(&csvFile.minute);
	releaseBuffers(&csvFile.envScore);
	releaseBuffers(&csvFile.socialScore);
	releaseBuffers(&csvFile.sentiment);
	releaseBuffers(&csvFile.location);
	releaseBuffers(&csvFile.tweet);
}

// delete intermediate products.  Call after completing parsing
void ClParser::releaseIntermediates() 
{
	free(unParsedRecords);
	releaseBuffers(&lineBreaks);
	releaseBuffers(&unParsedBuffers);
}



char * ClParser::getInputFile()
{
	return inputFile;
}

// end of clParser.cpp


// get number of records, columns, and column header from top of CSV
// note  that this metadata format is not common in csv files
cl_uint ClParser::loadMetaData(ifstream * inFile)
{
	char * metaData = new char[csvFile.CSV_ROW_LENGTH];
	
	//get number of records 
	inFile->getline(metaData, csvFile.CSV_ROW_LENGTH, ',');
	inFile->getline(metaData, csvFile.CSV_ROW_LENGTH, ',');
	csvFile.numRecords = stoi(metaData);

	//get number of columns
	inFile->getline(metaData, csvFile.CSV_ROW_LENGTH, ',');
	inFile->getline(metaData, csvFile.CSV_ROW_LENGTH, ',');
	csvFile.numVars = stoi(metaData);
	
	//get colunn headers
	inFile->getline(metaData, csvFile.CSV_ROW_LENGTH, '\n');
	delete metaData;

	// if the number of records if less than the batch size, then the batch size should be 
	// reduced
	if (csvFile.numRecords < csvFile.batchSize) { csvFile.batchSize = csvFile.numRecords; }
	csvFile.numBatches = (csvFile.numRecords + csvFile.batchSize - 1) / csvFile.batchSize;

	return 0;
}

// create a kernel for parsing time variables and set kernel args
void ClParser::setupTimeKernel(const char * funcName,cl_uint numThreadsInBatch)
{
	int bufferIndex = csvFile.year.size() - 1;
	cl_kernel kernel = clCreateKernel(program, funcName, &errNum);
	errNum += clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&unParsedBuffers[bufferIndex]);
	errNum += clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&csvFile.year[bufferIndex]);
	errNum += clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&csvFile.month[bufferIndex]);
	errNum += clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&csvFile.day[bufferIndex]);
	errNum += clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&csvFile.hour[bufferIndex]);
	errNum += clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&csvFile.minute[bufferIndex]);
	errNum += clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&lineBreaks[bufferIndex]);
	errNum += clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&numThreadsInBatch);
	checkErr(errNum, "setup time kernel");
	kernels.push_back(kernel);
	enqeueKernel(timeCommmandQueue, kernels.size() - 1, numThreadsInBatch, preferredDevice);
}

// create a kernel for parsing numerical, or score-related variables, and set args
void ClParser::setupScoreKernel(const char * funcName, cl_uint numThreadsInBatch)
{
	int bufferIndex = csvFile.envScore.size() - 1;
	cl_kernel kernel = clCreateKernel(program, funcName, &errNum);
	checkErr(errNum, "setup kernel");
	errNum += clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&unParsedBuffers[bufferIndex]);
	errNum += clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&csvFile.envScore[bufferIndex]);
	errNum += clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&csvFile.socialScore[bufferIndex]);
	errNum += clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&csvFile.sentiment[bufferIndex]);
	errNum += clSetKernelArg(kernel, 4, sizeof(cl_int),(void*)&numThreadsInBatch);
	errNum += clSetKernelArg(kernel, 5, sizeof(cl_int), (void*)&lineBreaks[bufferIndex]);
	kernels.push_back(kernel);
	enqeueKernel(scoreCommandQueue, kernels.size() - 1, numThreadsInBatch,preferredDevice);
}

// create a kernel for parsing text variables in csv, and set args
void ClParser::setupTextKernel(const char * funcName, cl_uint numThreadsInBatch)
{
	int bufferIndex = csvFile.tweet.size() - 1;
	cl_kernel kernel = clCreateKernel(program, funcName, &errNum);
	checkErr(errNum, "setup kernel");
	errNum += clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&unParsedBuffers[bufferIndex]);
	errNum += clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&csvFile.tweet[bufferIndex]);
	errNum += clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&csvFile.location[bufferIndex]);
	errNum += clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&numThreadsInBatch);
	errNum += clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)&lineBreaks[bufferIndex]);
	checkErr(errNum, "setup kernel");
	kernels.push_back(kernel);
	enqeueKernel(textCommandQueue, kernels.size() - 1, numThreadsInBatch, preferredDevice);

}

// create buffer and setup a kernel for parsing time variables in the csv file
void ClParser::parseTimeVars(cl_uint numThreadsInBatch, char * funcName)
{
	// buffers are created for the first batch in multi-batch processing.  For follow up batches, regions in the first bufffer 
	// are accessed using subbuffers.  At the end of the batch processing, all vals will be in the buffers created in the first 
	// batch
	if (csvFile.year.size() ==0)
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
	setupTimeKernel(funcName, numThreadsInBatch);
}

// create buffers and setup a kernel for parsing score related variables in a csv file
void ClParser::parseScoreVars(cl_uint numThreadsInBatch, char * funcName)
{
	// buffers are created for the first batch in multi-batch processing.  For follow up batches, regions in the first bufffer 
	// are accessed using subbuffers.  At the end of the batch processing, all vals will be in the buffers created in the first 
	// batch
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
	setupScoreKernel(funcName, numThreadsInBatch);
}

// create buffers and setup kernel args for parsing text vars in a csv file
void ClParser::parseTextVars(cl_uint numThreadsInBatch, char * funcName)
{
	// buffers are created for the first batch in multi-batch processing.  For follow up batches, regions in the first bufffer 
	// are accessed using subbuffers.  At the end of the batch processing, all vals will be in the buffers created in the first 
	// batch
	if (csvFile.tweet.size() == 0) 
	{
		createBuffer(sizeof(cl_char), &csvFile.tweet, csvFile.numRecords*csvFile.TEXT_OFFSETS[0], CL_MEM_READ_WRITE);
		createBuffer(sizeof(cl_char), &csvFile.location, csvFile.numRecords*csvFile.TEXT_OFFSETS[1], CL_MEM_READ_WRITE);
	}
	else 
	{
		cl_buffer_region twitterRegion = { csvFile.batchSize*csvFile.tweet.size()*csvFile.TEXT_OFFSETS[0],numThreadsInBatch * sizeof(cl_char)*csvFile.TEXT_OFFSETS[0] };
		cl_mem buffer = clCreateSubBuffer(csvFile.tweet[0], CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &twitterRegion, &errNum);
		checkErr(errNum, "create text sub buffer");
		csvFile.tweet.push_back(buffer);

		cl_buffer_region locationRegion = { csvFile.batchSize*csvFile.location.size()*csvFile.TEXT_OFFSETS[1],numThreadsInBatch * sizeof(cl_char)*csvFile.TEXT_OFFSETS[1] };
		buffer = clCreateSubBuffer(csvFile.location[0], CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &locationRegion, &errNum);
		checkErr(errNum, "create text sub buffer");
		csvFile.location.push_back(buffer);
	}
	setupTextKernel(funcName, numThreadsInBatch);
}

// parse all vars in a batch, one category at a time
void ClParser::parseVars(cl_uint numThreadsInBatch)
{
	parseTimeVars(numThreadsInBatch, "parse_timestamp");
	parseScoreVars(numThreadsInBatch, "parse_scores");
	parseTextVars(numThreadsInBatch, "parse_text");
}

// find all of the indices for newline characters in the csv file
void ClParser::findLineBreaks(cl_mem * lineBreaks, cl_int batchSize, cl_int * newIndex) 
{	
	cl_uint maxValNum = batchSize*csvFile.CSV_ROW_LENGTH;
	cl_uint offset = 250;					// starting stride between groups where indices 
											// of positively identified newline characters will be stored
	cl_uint groupSize = 25;					// number of threads to allocate for each stride group
	cl_uint numRecords = 0;					// number of contiguous sorted records below index 0.
	cl_uint numThreads = batchSize * 10;	// number of threads to allocate for entire dataset
	*newIndex = 0;							// index of the  last new line element to process

	// identify all newline characters, and send to the lowest unoccupied group index.
	cl_kernel kernel = clCreateKernel(program, "new_line", &errNum);
	errNum += clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&unParsedBuffers[unParsedBuffers.size() - 1]);
	errNum += clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)lineBreaks);
	kernels.push_back(kernel);
	checkErr(errNum, "setup new_line kernel");
	enqeueKernel(opQueue, kernels.size() - 1, maxValNum, 256, preferredDevice, &events[events.size()-1]);

	// combine the two closest clusters of new line indices.  Repeat until the batchsize number of indices
	// are contiguously below the 0 index
	kernel = clCreateKernel(program, "collapse_vals_global", &errNum);
	errNum += clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)lineBreaks);
	kernels.push_back(kernel);
	while (*newIndex < batchSize - 1)
	{
		errNum += clSetKernelArg(kernel, 1, sizeof(cl_int), (void*)&offset);
		errNum += clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&groupSize);
		checkErr(errNum, "setup collapse_vals_global kernel");
		enqeueKernel(opQueue, kernels.size() - 1, numThreads, 256, preferredDevice, &events[events.size() - 1]);
		copyDataToHost(opQueue, *lineBreaks, newIndex, 1);
		offset *= 2;
		groupSize *= 2;
	}
	// get the index where the last new line character to process in this batch is located
	clEnqueueReadBuffer(queues[opQueue], *lineBreaks, CL_TRUE, sizeof(cl_int)*batchSize, sizeof(cl_int), newIndex, 1, &events[events.size() - 1], NULL);
}

// asynchronously read input data into host memory.  Allows for parallel processing of OpenCL-related host code while 
// waiting for data reads
int ClParser::asyncFileRead(ifstream *inFile, char * unParsedData, cl_uint batchSize) 
{
	inFile->read(unParsedData, batchSize*csvFile.CSV_ROW_LENGTH);
	return inFile->gcount();
}

// Where most of the logic flow management occurs.  Responsible for 
// handling async operations and parsing data in batches
void ClParser::processCSVFile(ifstream *inFile, char * unParsedRecords)
{
	cl_uint batchSize = csvFile.batchSize;
	int unParsedLocation = 0;
	cl_int newIndex = 0;
	cl_int copyIndex = 0;
	future <int> futureVal;
	bool stillReading = true;

	// asynchronously read first batch of data while OpenCL creates a program
	futureVal = std::async(&ClParser::asyncFileRead, this, inFile, &unParsedRecords[unParsedLocation], batchSize);
	createProgram(1, deviceIDs, preferredDevice);
	
	for (cl_uint batchNum = 0; batchNum < csvFile.numBatches; batchNum++)
	{
		createBuffer(sizeof(cl_int), &lineBreaks, batchSize*csvFile.CSV_ROW_LENGTH, CL_MEM_READ_WRITE);
		createBuffer(sizeof(cl_char), &unParsedBuffers, batchSize*csvFile.CSV_ROW_LENGTH, CL_MEM_READ_ONLY);

		// wait until async read is complete and access results.  Start new async read if there is still data 
		// in the input dataset
		if (stillReading) { unParsedLocation += futureVal.get(); }
		if (!inFile->eof())
		{
			futureVal = async(&ClParser::asyncFileRead, this, inFile, &unParsedRecords[unParsedLocation], batchSize);
		}
		else { stillReading = false; }

		// copy input data to device on a dedicated queue
		cl_event loadDataEvent;
		copyDataToBuffer(opQueue, &(unParsedBuffers[unParsedBuffers.size() - 1]),
			&unParsedRecords[copyIndex], batchSize*csvFile.CSV_ROW_LENGTH, &loadDataEvent);
		clWaitForEvents(1, &loadDataEvent);

		// important debug step! Test for unsorted output and missing values
		findLineBreaks(&lineBreaks[lineBreaks.size() - 1], batchSize, &newIndex);
		if (debug) { checkLineBreakConsistency(&lineBreaks[lineBreaks.size() - 1], batchSize, copyIndex, batchNum); }
		copyIndex += newIndex + 1;

		parseVars(csvFile.batchSize);
	
	}
	
	releaseKernels();
}


// import debug operation.  Check for errors in finding new line segments and sorting data.  Thread race conditions
// led to numerous debugs in previous builds.  Leads to signifciant slow down, so only use in debug mode.  NOTE:
// if working correctly, the 0 index for each batch should fail all three tests.
void ClParser::checkLineBreakConsistency(cl_mem * lineBreaks, cl_uint batchSize, cl_uint copyIndex, cl_uint batchNum)
{
	int * debugPrint = (int*)malloc(sizeof(int)*csvFile.batchSize*csvFile.CSV_ROW_LENGTH);
	copyDataToHost(opQueue, *lineBreaks, &(debugPrint[0]), csvFile.batchSize*csvFile.CSV_ROW_LENGTH);
	clWaitForEvents(1, &events[events.size() - 1]);
	clFinish(queues[opQueue]);
	for (int i = 1; i < 10; i++) { cout << (debugPrint[i]) << endl; }
	for (int i = 1; i < batchSize; i++) 
	{
		// check all indices matches the new line character
		if (unParsedRecords[debugPrint[i] + copyIndex] != '\n') 
		{
			cout << "mismatch, index " << i + batchNum*batchSize << ", value " << debugPrint[i] + copyIndex << "," << endl;
			cout << unParsedRecords[debugPrint[i] - 1 + copyIndex]  << unParsedRecords[debugPrint[i] + copyIndex] 
				<< unParsedRecords[debugPrint[i] + 1 + copyIndex] << endl;
		}
		// check for zeros (no data)
		if (debugPrint[i] == 0) { cout << "zero warning" << i << endl; }
		// check for sorted order
		if (debugPrint[i] < debugPrint[i - 1]) { cout << "warning out of order element at index " << i << " upper val: " << debugPrint[i] << " Lower val: " << debugPrint[i-1] << endl; }
	}
	free(debugPrint);
}

// copy integer data from buffer to host device.  Abstracts away many of the details
void ClParser::BuffersToHost(cl_int * inputPtr, std::vector<cl_mem> * buffers, cl_uint queueNum)
{
	for (cl_uint batchNum = 0; batchNum < csvFile.numBatches - 1; batchNum++)
	{
		copyDataToHost(queueNum, (*buffers)[batchNum], &(inputPtr[csvFile.batchSize*batchNum]), csvFile.batchSize);
	}
	// for final batch that may be truncated
	int tempBatchSize = csvFile.numRecords - (csvFile.batchSize)*(csvFile.numBatches - 1);
	int batchNum = csvFile.numBatches - 1;
	copyDataToHost(queueNum, (*buffers)[batchNum], &(inputPtr[csvFile.batchSize*batchNum]), tempBatchSize);
}


// copy character data from buffer to host device.  Abstracts away many of the details
void ClParser::BuffersToHost(cl_char * inputPtr, std::vector<cl_mem> * buffers, cl_uint queueNum, cl_uint textOffest)
{
	for (cl_uint batchNum = 0; batchNum < csvFile.numBatches - 1; batchNum++)
	{
		copyDataToHost(queueNum, (*buffers)[batchNum], &(inputPtr[csvFile.batchSize*batchNum*textOffest]), csvFile.batchSize*textOffest);
	}
	// for final batch that may be truncated
	int tempBatchSize = csvFile.numRecords - (csvFile.batchSize)*(csvFile.numBatches - 1);
	int batchNum = csvFile.numBatches - 1;
	copyDataToHost(queueNum, (*buffers)[batchNum], &(inputPtr[csvFile.batchSize*batchNum*textOffest]), tempBatchSize*textOffest);
}

// print sample from parsed csv file and test values for consistency
void ClParser::printOutput()
{	
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
	clFinish(queues[timeCommmandQueue]);

	cl_int * sentPtr = (cl_int*)malloc(sizeof(cl_int)*csvFile.numRecords);
	BuffersToHost(sentPtr, &csvFile.sentiment, scoreCommandQueue);
	cl_int * envPtr = (cl_int*)malloc(sizeof(cl_int)*csvFile.numRecords);
	BuffersToHost(envPtr, &csvFile.envScore, scoreCommandQueue);
	cl_int * socialPtr = (cl_int*)malloc(sizeof(cl_int)*csvFile.numRecords);
	BuffersToHost(socialPtr, &csvFile.socialScore, scoreCommandQueue);
	clFinish(queues[scoreCommandQueue]);

	cl_char * textPtr = (cl_char*)malloc(sizeof(cl_char)*csvFile.numRecords*csvFile.TEXT_OFFSETS[1]);
	BuffersToHost(textPtr, &csvFile.location, textCommandQueue, csvFile.TEXT_OFFSETS[1]);
	cl_char * tweetPtr = (cl_char*)malloc(sizeof(cl_char)*csvFile.numRecords*csvFile.TEXT_OFFSETS[0]);
	BuffersToHost(tweetPtr, &csvFile.tweet, textCommandQueue, csvFile.TEXT_OFFSETS[0]);
	clFinish(queues[textCommandQueue]);

	// print out parsed values for the final 8 records in the dataset.  
	for (int i = csvFile.numRecords - 8; i < csvFile.numRecords; i++)
	{
		cout << "year " << timePtrs[0][i] << ", month " << timePtrs[1][i] << ", day " << timePtrs[2][i] << ", hour "
			<< timePtrs[3][i] << " ,minute " << timePtrs[4][i] << endl;
		cout << "sentiment " << sentPtr[i] << ", envScore " << envPtr[i] << ", socialScore " << socialPtr[i] << endl;
		
		cout << "twitter text: ";
		for (int j = 0; j < 280; j++)
		{
			cout << tweetPtr[j+i*csvFile.TEXT_OFFSETS[0]];
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
	
	// free locally allocated memories

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

// primary function called externally by other classes.
void  ClParser::parseFile(char *inputFile)
{
	this->debug = false;
	this->inputFile = inputFile;
	ifstream inFile(inputFile);
	loadMetaData(&inFile);
	allocateMemory();

	for (int queueNum = 0; queueNum < 4; queueNum++) 
	{
		createCommandQueue(preferredDevice);
	}

	processCSVFile(&inFile, unParsedRecords);
	inFile.close();

	clWaitForEvents(events.size(), events.data());

	if (debug) { printOutput(); }
	cleanupCheck = std::async(&ClParser::releaseIntermediates,this);
}
