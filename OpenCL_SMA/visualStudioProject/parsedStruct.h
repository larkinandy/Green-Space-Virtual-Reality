#pragma once

#include <vector>
#include <CL/cl.h>

// struct for holding parsed CSV data.  one vector for each variable
struct parsedCSV
{
	cl_uint numRecords;
	cl_uint numVars;
	cl_uint numBatches;
	cl_uint batchSize = 4096*2;
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

// struct for holding derived statistics.  Once vector for each summary statistic
struct summaryStats
{





};