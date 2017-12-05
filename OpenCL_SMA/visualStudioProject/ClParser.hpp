#pragma once
#include "DeviceBaseClass.hpp"

#include <iostream>
#include <fstream>

#include <future>

using namespace std;

class ClParser : DeviceBaseClass 
{


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
	const int NUM_TIME_VARS = 5;
	const int NUM_SCORE_VARS = 3;
	const int NUM_TEXT_VARS = 2;
	
	std::vector<cl_mem> unParsedBuffers;
	std::vector<cl_mem> lineBreaks;
	
	int newEventNum;

	parsedCSV csvFile;

	const int memcpyCommandQueue = 0;
	const int timeCommmandQueue = 1;
	const int scoreCommandQueue = 2;
	const int textCommandQueue = 3;

	//cl_mem lineBreaks;

	cl_uint readEventNum;

	cl_uint loadMetaData(ifstream * inFile);
	void processCSVFile(ifstream * inFile, char * unParsedRecords);
	void allocateMemory();
	void parseVars(cl_uint numThreadsInBatch, cl_mem * lineBreaks,cl_int offsetIndex);
	void releaseMemory();
	
	void parseTimeVars(cl_uint numThreadsInBatch, char * funcName, cl_mem * lineBreaks,cl_int startingIndex);
	void parseScoreVars(cl_uint numThreadsInBatch, char * funcName,cl_mem * lineBreaks,cl_int startingIndex);
	void parseTextVars(cl_uint numThreadsInBatch, char * funcName, cl_mem * lineBreaks,cl_int startingIndex);

	void setupKernel(const char * funcName, int numVars, std::vector<cl_mem> varBuffers);
	void setupTimeKernel(const char * funcName, cl_uint numThreadsInBatch, cl_mem * lineBreaks,cl_int startingIndex);
	void setupScoreKernel(const char * funcName, cl_uint numThreadsInBatch,cl_mem * lineBreaks,cl_int startingIndex);
	void setupTextKernel(const char * funcName, cl_uint numThreadsInBatch, cl_mem * lineBreaks, cl_int startingIndex);

	void BuffersToHost(cl_int * inputPtr, std::vector<cl_mem> *buffers, cl_uint queueNum);
	void BuffersToHost(cl_char * inputPtr, std::vector<cl_mem> *buffers, cl_uint queueNum, const cl_uint offest);

	void printLineSearchDebug(cl_mem * lineBreaks);
	void findLineBreaks(cl_mem * lineBreaks, cl_int batchSize, cl_int * newIndex);


	int twice(ifstream *inFile, char * unParsedData, cl_uint batchSize);
};