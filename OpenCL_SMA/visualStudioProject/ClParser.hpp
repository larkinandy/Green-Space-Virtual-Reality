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
	void parseFile(char *inputFile);
	char * getInputFile();
	void printOutput();

protected:



private:
	char * inputFile;
	char * unParsedRecords = NULL;
	const cl_uint unParsedQueueNum = 0;
	const cl_uint CSV_ROW_LENGTH = 600;
	cl_uint TEXT_OFFSETS[2] = { 280 ,20 };	// number of characters in the location and tweet text variables
	cl_uint numRecords, numCols, numBatches = 0;
	cl_uint batchSize = 33;
	const int NUM_TIME_VARS = 5;
	const int NUM_SCORE_VARS = 3;
	const int NUM_TEXT_VARS = 2;

	std::vector<cl_mem> timeBuffers;
	std::vector<cl_mem> scoreBuffers;
	std::vector<cl_mem> envBuffers;
	std::vector<cl_mem> textBuffers;
	std::vector<cl_mem> unParsedBuffers;

	const int memcpyCommandQueue = 0;
	const int timeCommmandQueue = 1;
	const int scoreCommandQueue = 2;
	const int textCommandQueue = 3;

	cl_uint loadMetaData(ifstream * inFile);
	void processCSVBatch(ifstream *inFile, char * unParsedRecords, cl_uint batchNum);
	void allocateMemory();
	void parseVars(cl_uint numThreadsInBatch);
	void releaseMemory();
	void parseVarCategory(int numVars, int commandQueue, std::vector<cl_mem>*  varBuffers, const char * funcName, cl_uint numThreadsInBatch, cl_uint * textOffsets);
	void parseVarCategory(int numVars, int queueNum, std::vector<cl_mem> * varBuffers, const char * funcName, cl_uint numThreadsInBatch);
	void BuffersToHost(std::vector<cl_int*> ptrs, int commandQueue, std::vector<cl_mem> * varBuffers);
	void BuffersToHost(std::vector<cl_char*> ptrs, int commandQueue, std::vector<cl_mem> * varBuffers);

	void setupKernel(const char * funcName, int numVars, std::vector<cl_mem> varBuffers);
};