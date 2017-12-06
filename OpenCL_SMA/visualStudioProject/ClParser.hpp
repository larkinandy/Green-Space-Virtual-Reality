/* ClParser.hpp
* Header file for class that parses csv files using OpenCL
* Author: Andrew Larkin
* December 5, 2017 
*/

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
		int newEventNum;
	
		std::vector<cl_mem> unParsedBuffers;
		std::vector<cl_mem> lineBreaks;

		parsedCSV csvFile; // custom struct to store results

		// one queue for each data type/major operation
		const int opQueue = 0;
		const int timeCommmandQueue = 1;
		const int scoreCommandQueue = 2;
		const int textCommandQueue = 3;

		//
		cl_uint loadMetaData(ifstream * inFile);
		void processCSVFile(ifstream * inFile, char * unParsedRecords);
		void allocateMemory();
		int asyncFileRead(ifstream *inFile, char * unParsedData, cl_uint batchSize);
		void releaseMemory();
	
		void parseVars(cl_uint numThreadsInBatch);
		void parseTimeVars(cl_uint numThreadsInBatch, char * funcName);
		void parseScoreVars(cl_uint numThreadsInBatch, char * funcName);
		void parseTextVars(cl_uint numThreadsInBatch, char * funcName);

		void setupTimeKernel(const char * funcName, cl_uint numThreadsInBatch);
		void setupScoreKernel(const char * funcName, cl_uint numThreadsInBatch);
		void setupTextKernel(const char * funcName, cl_uint numThreadsInBatch);

		void BuffersToHost(cl_int * inputPtr, std::vector<cl_mem> *buffers, cl_uint queueNum);
		void BuffersToHost(cl_char * inputPtr, std::vector<cl_mem> *buffers, cl_uint queueNum, const cl_uint offest);

		void findLineBreaks(cl_mem * lineBreaks, cl_int batchSize, cl_int * newIndex);
		void checkLineBreakConsistency(cl_mem * lineBreaks, cl_uint batchSize, cl_uint copyIndex, cl_uint batchNum);

};

/* end of ClParser.hpp*/