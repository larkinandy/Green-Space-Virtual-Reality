#pragma once

#include <vector>
#include<chrono>
#include <algorithm>
#include <CL/cl.h>

struct parsedCSV
{
	cl_uint numRecords;
	cl_uint numVars;
	cl_uint numBatches;
	cl_uint batchSize = 2048;
	std::vector<cl_mem> year;
	std::vector<cl_mem> month;
	std::vector<cl_mem> day;
	std::vector<cl_mem> hour;
	std::vector<cl_mem> minute;
	std::vector<cl_mem> sentiment;
	std::vector<cl_mem> envScore;
	std::vector<cl_mem> socialScore;
	std::vector<cl_mem> location;
	std::vector<cl_mem> tweet;
};


struct summaryStats
{





};