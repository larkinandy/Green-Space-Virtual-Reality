#include "ClParser.hpp"


ClParser::ClParser(cl_context * contextPtr, cl_device_id * deviceIDs, cl_uint numDevices, cl_uint preferredDevice) : DeviceBaseClass(contextPtr, deviceIDs, numDevices, preferredDevice)
{
	deviceFunctionFile = "parserKernels.cl";
}


ClParser::~ClParser() {


}


ClParser::ClParser() :DeviceBaseClass() {


}

cl_uint ClParser::loadMetaData(ifstream * inFile)
{
	char * metaData = new char[CSV_ROW_LENGTH];
	
	//get number of records 
	inFile->getline(metaData, CSV_ROW_LENGTH, ',');
	inFile->getline(metaData, CSV_ROW_LENGTH, ',');
	numRecords = stoi(metaData);
	//get number of columns
	inFile->getline(metaData, CSV_ROW_LENGTH, ',');
	inFile->getline(metaData, CSV_ROW_LENGTH, ',');
	numCols = stoi(metaData);
	
	//get colunn headers
	inFile->getline(metaData, CSV_ROW_LENGTH, '\n');

	delete metaData;

	if (numRecords < batchSize) { batchSize = numRecords; }
	numBatches = (numRecords + batchSize - 1) / batchSize;

	return 0;
}

void ClParser::setupKernel(const char * funcName, int numVars, std::vector<cl_mem> varBuffers) {

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


void ClParser::parseVarCategory(int numVars, int queueNum, std::vector<cl_mem> * varBuffers, const char * funcName, cl_uint numThreadsInBatch, cl_uint * textOffsets) {
	for (cl_uint bufferIndex = 0; bufferIndex < numVars; bufferIndex++) {
		createBuffer(sizeof(cl_int), varBuffers, numThreadsInBatch*textOffsets[bufferIndex], CL_MEM_WRITE_ONLY);
	}
	setupKernel(funcName, numVars, *varBuffers);
	//enqeueKernel(queueNum, kernels.size() - 1, numThreadsInBatch, preferredDevice, &events[events.size() - 1]);
	enqeueKernel(queueNum, kernels.size() - 1, numThreadsInBatch, preferredDevice);
	//enqeueKernel(0, kernels.size() - 1, numThreadsInBatch, preferredDevice);
}



void ClParser::parseVarCategory(int numVars, int queueNum, std::vector<cl_mem> * varBuffers, const char * funcName, cl_uint numThreadsInBatch) {
	for (cl_uint bufferIndex = 0; bufferIndex < numVars; bufferIndex++) {
		createBuffer(sizeof(cl_int), varBuffers,numThreadsInBatch, CL_MEM_WRITE_ONLY);
	}
	setupKernel(funcName, numVars, *varBuffers);
	//enqeueKernel(queueNum, kernels.size() - 1, numThreadsInBatch, preferredDevice, &events[events.size() - 1]);
	enqeueKernel(queueNum, kernels.size() - 1, numThreadsInBatch, preferredDevice);
	//enqeueKernel(0, kernels.size() - 1, numThreadsInBatch, preferredDevice);
}




void ClParser::parseVars(cl_uint numThreadsInBatch)
{
	parseVarCategory(NUM_TIME_VARS, timeCommmandQueue, &timeBuffers, "parse_timestamp", numThreadsInBatch);
	parseVarCategory(NUM_SCORE_VARS,scoreCommandQueue, &scoreBuffers, "parse_scores", numThreadsInBatch);
	parseVarCategory(NUM_TEXT_VARS, textCommandQueue, &textBuffers, "parse_text", numThreadsInBatch,TEXT_OFFSETS);
}


void ClParser::processCSVBatch(ifstream *inFile, char * unParsedRecords, cl_uint batchNum)
{
	cl_uint minRecordNum = batchNum*batchSize;
	cl_uint maxRecordNum = min((batchNum + 1)*batchSize, numRecords);

	for (int recordNum = minRecordNum; recordNum < maxRecordNum; recordNum++) {
		inFile->getline(&(unParsedRecords[recordNum*CSV_ROW_LENGTH]), CSV_ROW_LENGTH, '\n');
	}
	

	createBuffer(sizeof(cl_char), &unParsedBuffers, batchSize*CSV_ROW_LENGTH, CL_MEM_READ_ONLY);


	copyDataToBuffer(memcpyCommandQueue, &(unParsedBuffers[unParsedBuffers.size()-1]), &(unParsedRecords[minRecordNum*CSV_ROW_LENGTH]), (maxRecordNum - minRecordNum)*CSV_ROW_LENGTH);  // copy input data to device on a dedicated queue


	cl_uint numThreadsInBatch = maxRecordNum - minRecordNum;
	parseVars(numThreadsInBatch);



	

}

void ClParser::allocateMemory()
{
	unParsedRecords = (char*)malloc(sizeof(char)*numRecords*CSV_ROW_LENGTH);

}

void ClParser::releaseMemory()
{
	free(unParsedRecords);


}
void ClParser::BuffersToHost(std::vector<cl_int*> ptrs, int commandQueue, std::vector<cl_mem> * varBuffers)
{
	for (cl_uint batchNum = 0; batchNum < numBatches - 1; batchNum++)

	{
		for (cl_uint ptrNum = 0; ptrNum < ptrs.size(); ptrNum++)
		{
			copyDataToHost(commandQueue, (*varBuffers)[ptrs.size()*batchNum + ptrNum], &(ptrs[ptrNum][batchSize*batchNum]), batchSize);
		}
	}
	// for final batch that may be truncated
	int tempBatchSize = numRecords - (batchSize)*(numBatches - 1);
	int batchNum = numBatches - 1;
	for (cl_uint ptrNum = 0; ptrNum < ptrs.size(); ptrNum++)
	{
		copyDataToHost(commandQueue, (*varBuffers)[ptrs.size()*batchNum + ptrNum], &(ptrs[ptrNum][batchSize*batchNum]), tempBatchSize);
	}
}



void ClParser::BuffersToHost(std::vector<cl_char*> ptrs, int queueNum, std::vector<cl_mem> * varBuffers)
{
	for (cl_uint batchNum = 0; batchNum < numBatches - 1; batchNum++)
	{
		for (cl_uint ptrNum = 0; ptrNum < ptrs.size(); ptrNum++)
		{
			copyDataToHost(queueNum, (*varBuffers)[ptrs.size()*batchNum + ptrNum], &(ptrs[ptrNum][batchSize*batchNum*TEXT_OFFSETS[ptrNum]]), batchSize*TEXT_OFFSETS[ptrNum]);
		}
	}
	// for final batch that may be truncated
	int tempBatchSize = numRecords - (batchSize)*(numBatches - 1);
	int batchNum = numBatches - 1;
	for (cl_uint ptrNum = 0; ptrNum < ptrs.size(); ptrNum++)
	{
		copyDataToHost(queueNum, (*varBuffers)[ptrs.size()*batchNum + ptrNum], &(ptrs[ptrNum][batchSize*batchNum*TEXT_OFFSETS[ptrNum]]), tempBatchSize*TEXT_OFFSETS[ptrNum]);
	}


}

void ClParser::printOutput()
{

	clWaitForEvents(events.size(), events.data());

	


std::vector<cl_char*> textPtrs;
std::vector<cl_int *> timePtrs;
std::vector<cl_int*> scorePtrs;


for (cl_int ptrIndex = 0; ptrIndex < 5; ptrIndex++) {
cl_int * tmpPtr = (cl_int*)malloc(sizeof(cl_int)*numRecords);
timePtrs.push_back(tmpPtr);
}
for (cl_uint ptrIndex = 0; ptrIndex < 2; ptrIndex++) {
cl_char * tmpPtr = (cl_char*)malloc(sizeof(cl_char)*TEXT_OFFSETS[ptrIndex] * numRecords);
textPtrs.push_back(tmpPtr);
}
for (cl_uint ptrIndex = 0; ptrIndex < 3; ptrIndex++) {
cl_int * tmpPtr = (cl_int*)malloc(sizeof(cl_int)*numRecords);
scorePtrs.push_back(tmpPtr);
}

	BuffersToHost(timePtrs, timeCommmandQueue, &timeBuffers);
	BuffersToHost(scorePtrs, scoreCommandQueue, &scoreBuffers);
	BuffersToHost(textPtrs, textCommandQueue, &textBuffers); 


	/*
	for (int i = 0; i < queues.size() - 1; i++)
	{
		errNum = clFinish(queues[i]);
		checkErr(errNum, "finish queue");
	}
	*/
	for (int i = numRecords-8; i < numRecords; i++)
	{
		cout << "year " << timePtrs[0][i];
		cout << " month " << timePtrs[1][i];
		cout << "day " << timePtrs[2][i];
		cout << " hour " << timePtrs[3][i];
		cout << " minute " << timePtrs[4][i];
		cout << " score1 " << scorePtrs[0][i];
		cout << " score2 " << scorePtrs[1][i];
		cout << " score3 " << scorePtrs[2][i];
		cout << endl;
		cout << endl;


		cout << "twitter text: ";
		for (int j = 0; j < 100; j++)
		{
			cout << textPtrs[0][j+i*TEXT_OFFSETS[0]];
		}
		cout << endl;
		cout << endl;

		cout << "location: ";
		for (int j = 0; j < 20; j++)
		{
			cout << textPtrs[1][j+i*TEXT_OFFSETS[1]];

		}
		cout << endl;
		cout << endl;
		cout << endl;

	



	}




	for (int ptrIndex = 0; ptrIndex < textPtrs.size(); ptrIndex++)
	{
		free(textPtrs[ptrIndex]);
	}
	for (int ptrIndex = 0; ptrIndex < timePtrs.size(); ptrIndex++)
	{
		free(timePtrs[ptrIndex]);
	}
	for (int ptrIndex = 0; ptrIndex < scorePtrs.size(); ptrIndex++)
	{
		free(scorePtrs[ptrIndex]);
	}


}
void  ClParser::parseFile(char *inputFile)
{
	
	batchSize =2048;
	ifstream inFile(inputFile);
	loadMetaData(&inFile);
	createProgram(1, deviceIDs, preferredDevice);
	allocateMemory();

	createCommandQueue(preferredDevice);
	createCommandQueue(preferredDevice);
	createCommandQueue(preferredDevice);
	createCommandQueue(preferredDevice);

	for (cl_uint batchNum = 0; batchNum < numBatches; batchNum++)
	{
		processCSVBatch(&inFile, unParsedRecords, batchNum);
		
		
	}

	inFile.close();


//	if (debug) {printOutput();}

	releaseMemory();
	
}




char * ClParser::getInputFile()
{
	return inputFile;

}