#include "ClParser.hpp"



ClParser::ClParser(cl_context * contextPtr, cl_device_id * deviceIDs, cl_uint numDevices, cl_uint preferredDevice) : DeviceBaseClass(contextPtr, deviceIDs, numDevices, preferredDevice)
{
	deviceFunctionFile = "parserKernels.cl";
}


ClParser::~ClParser() {


}


ClParser::ClParser() :DeviceBaseClass() {


}

cl_int ClParser::getMetaData(ifstream * inFile, char *metaData)
{

	if (!inFile->is_open()) { return -1; }
	//discard first line

	inFile->getline(metaData, CSV_ROW_LENGTH, ',');
	inFile->getline(metaData, CSV_ROW_LENGTH, ',');
	numRecords = stoi(metaData);
	inFile->getline(metaData, CSV_ROW_LENGTH, ',');
	inFile->getline(metaData, CSV_ROW_LENGTH, ',');
	numCols = stoi(metaData);
	inFile->getline(metaData, CSV_ROW_LENGTH, '\n');
	return 0;
}

int ClParser::getHeaderInfo(char *metaData)
{
	return 1;
}


void ClParser::setupTimeKernel() {

	cl_kernel kernel = clCreateKernel(program, "parse_timestamp", &errNum);
	checkErr(errNum, "create time parse kernel");
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffers[unParsedBuffers[unParsedBuffers.size() - 1]]);
	for (int ptrIndex = 0; ptrIndex < timePtrs.size(); ptrIndex++)
	{
		errNum = clSetKernelArg(kernel, timePtrs.size() - ptrIndex, sizeof(cl_mem), (void *)&buffers[timeBuffers[timeBuffers.size() - 1 - ptrIndex]]);
	}
	kernels.push_back(kernel);
}

void ClParser::setupKernel(const char * funcName, std::vector<cl_int*> ptrs, std::vector<cl_int> varBuffers) {

	cl_kernel kernel = clCreateKernel(program, funcName, &errNum);
	checkErr(errNum, "setup kernel");
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffers[unParsedBuffers[unParsedBuffers.size() - 1]]);
	for (int ptrIndex = 0; ptrIndex < ptrs.size(); ptrIndex++)
	{
		errNum = clSetKernelArg(kernel, ptrs.size() - ptrIndex, sizeof(cl_mem), (void *)&buffers[varBuffers[varBuffers.size() - 1 - ptrIndex]]);
	}
	kernels.push_back(kernel);
}


void ClParser::parseVarCategory(std::vector<cl_int*> ptrs, std::vector<cl_int> * varQueues, std::vector<cl_int> * varBuffers, const char * funcName) {
	createCommandQueue(preferredDevice);
	varQueues->push_back(queues.size() - 1);
	for (cl_uint bufferIndex = 0; bufferIndex < ptrs.size(); bufferIndex++) {
		createBuffer(sizeof(cl_int), batchSize, CL_MEM_WRITE_ONLY);
		varBuffers->push_back(buffers.size() - 1);
	}
	setupKernel(funcName, ptrs, *varBuffers);
	enqeueKernel((*varQueues)[varQueues->size() - 1], kernels.size() - 1, batchSize, preferredDevice);
}

cl_int ClParser::parseTimeVars()
{
	parseVarCategory(timePtrs, &timeQueues, &timeBuffers, "parse_timestamp");
	parseVarCategory(scorePtrs, &scoreQueues, &scoreBuffers, "parse_scores");
	return 0;
}


cl_int ClParser::processCSVBatch(ifstream *inFile, char * unParsedRecords, cl_uint batchNum)
{

	cl_uint minRecordNum = batchNum*batchSize;
	cl_uint maxRecordNum = min((batchNum + 1)*batchSize, numRecords);
	cl_uint queueNum = batchNum;



	cl_uint startBuffer = buffers.size();
	for (int recordNum = minRecordNum; recordNum < maxRecordNum; recordNum++)
	{
		inFile->getline(&(unParsedRecords[recordNum*CSV_ROW_LENGTH]), CSV_ROW_LENGTH, '\n');
	}
	createBuffer(sizeof(cl_char), batchSize*CSV_ROW_LENGTH, CL_MEM_READ_ONLY);


	copyDataToBuffer(0, buffers.size() - 1, &(unParsedRecords[minRecordNum*CSV_ROW_LENGTH]), (maxRecordNum - minRecordNum)*CSV_ROW_LENGTH);  // copy input data to device on a dedicated queue
	unParsedBuffers.push_back(buffers.size() - 1);
	parseTimeVars();


	return 1;
}

void ClParser::allocateMemory(cl_uint numRecords)
{
	unParsedRecords = (char*)malloc(sizeof(char)*numRecords*CSV_ROW_LENGTH);

	for (cl_int ptrIndex = 0; ptrIndex < 5; ptrIndex++) {
		cl_int * tmpPtr = (cl_int*)malloc(sizeof(cl_int)*numRecords);
		timePtrs.push_back(tmpPtr);
	}

	for (cl_uint ptrIndex = 0; ptrIndex < 2; ptrIndex++) {
		cl_char * tmpPtr = (cl_char*)malloc(sizeof(cl_char)*TEXT_OFFSETS[ptrIndex] * numRecords);
		textPtrs.push_back(tmpPtr);
	}

	for (cl_uint ptrIndex = 0; ptrIndex < 3; ptrIndex++) {
		cl_int * tmpPtr = (cl_int*)malloc(sizeof(cl_int)*TEXT_OFFSETS[ptrIndex] * numRecords);
		scorePtrs.push_back(tmpPtr);
	}


}

void ClParser::releaseMemory()
{
	free(unParsedRecords);
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

void ClParser::BuffersToHost(std::vector<cl_int*> ptrs, std::vector<cl_int> varQueues, std::vector<cl_int> varBuffers)
{
	cl_uint numBatches = numRecords  / batchSize;
	for (cl_uint batchNum = 0; batchNum < numBatches; batchNum++)
	{
		for (cl_uint ptrNum = 0; ptrNum < ptrs.size(); ptrNum++)
		{
			copyDataToHost(varQueues[batchNum], varBuffers[ptrs.size()*batchNum + ptrNum], &(ptrs[ptrNum][batchSize*batchNum]), batchSize);
		}
	}
}



cl_int ClParser::LoadFile(char *inputFile)
{
	ifstream inFile(inputFile);
	char * metaData = new char[CSV_ROW_LENGTH];
	getMetaData(&inFile, metaData);
	getHeaderInfo(metaData);
	createProgram(1, deviceIDs, preferredDevice);
	delete metaData;
	if (debug)
	{
		cout << numRecords;
		cout << " loading csv " << endl;
	}

	if (numRecords < batchSize) { batchSize = numRecords; }

	cl_uint numElements = numRecords*CSV_ROW_LENGTH;
	cl_uint numBatches = (numRecords + batchSize - 1) / batchSize;
	allocateMemory(numRecords);
	createCommandQueue(preferredDevice);
	for (cl_uint batchNum = 0; batchNum < numBatches; batchNum++)
	{
		processCSVBatch(&inFile, unParsedRecords, batchNum);
	}

	inFile.close();


	for (int i = 0; i < queues.size(); i++) {
		clFinish(queues[i]);
	}

	BuffersToHost(timePtrs,timeQueues,timeBuffers);
	BuffersToHost(scorePtrs, scoreQueues, scoreBuffers);

	for (int i = 0; i < queues.size(); i++) {
		clFinish(queues[i]);
	}


/*
	for (int i = 0; i < 90; i++) {
		//cout << "unparsed" << unParsedRecords[i];
		cout << "year " << timePtrs[0][i];
		cout << " month " << timePtrs[1][i];
		cout << "day " << timePtrs[2][i];
		cout << " hour " << timePtrs[3][i];
		cout << " minute " << timePtrs[4][i];
		cout << endl;
	}

	*/

	for (int i = 0; i < 90; i++) {
		cout << " score1 " << scorePtrs[0][i];
		cout << " score2 " << scorePtrs[1][i];
		cout << " score3 " << scorePtrs[2][i];
		cout << endl;
	}


	releaseMemory();
	return 0;
}




char * ClParser::getInputFile()
{
	return inputFile;

}