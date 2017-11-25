#pragma once
#include "DeviceBaseClass.hpp"

#include <iostream>
#include <fstream>

using namespace std;

class ClParser : DeviceBaseClass {


public:
	ClParser();
	ClParser(cl_context * contextPtr, cl_device_id * deviceIDs, cl_uint numDevices, cl_uint preferredDevice);
	~ClParser();
	cl_int getMetaData(ifstream * inFile, char *metaData);
	cl_int LoadFile(char *inputFile);
	char * getInputFile();


protected:



private:
	char * inputFile;
	const cl_uint CSV_ROW_LENGTH = 600;
	cl_uint TEXT_OFFSETS[2] = { 280 ,20 };	// number of characters in the location and tweet text variables
	cl_uint numRecords = 0;
	cl_uint numCols = 10;
	int batchSize = 512;

	std::vector<cl_int> timeQueues;
	std::vector<cl_int> timeBuffers;
	std::vector<cl_int> scoreQueues;
	std::vector<cl_int> scoreBuffers;
	std::vector<cl_int> textQueues;
	std::vector<cl_int> textBuffers;

	std::vector<cl_char*> textPtrs;
	std::vector<cl_int *> timePtrs;
	std::vector<cl_int*> scorePtrs;

	std::vector<cl_int> unParsedBuffers;

	char * unParsedRecords = NULL;
	const cl_uint unParsedQueueNum = 0;

	int getHeaderInfo(char *metaData);
	int processCSVBatch(ifstream *inFile, char * unParsedRecords, cl_uint batchNum);
	void setupTimeKernel();
	void allocateMemory(cl_uint numElements);
	cl_int parseTimeVars();
	void releaseMemory();
	void BuffersToHost(std::vector<cl_int*> ptrs, std::vector<cl_int> varQueues, std::vector<cl_int> varBuffers);
	void parseVarCategory(std::vector<cl_int*> ptrs, std::vector<cl_int>*  varQueues, std::vector<cl_int>*  varBuffers, const char * funcName);
	void setupKernel(const char * funcName, std::vector<cl_int*> ptrs, std::vector<cl_int> varBuffers);

};