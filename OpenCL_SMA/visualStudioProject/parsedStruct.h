/* parsedStruct.h
* structs for holding CSV data and summary statistics
* Author: Andrew Larkin
* December 5, 2017 */

#pragma once

#include <vector>
#include <CL/cl.h>

// struct for holding parsed CSV data.  one vector for each variable
struct parsedCSV
{
	const cl_uint CSV_ROW_LENGTH = 500;				// largely dependent on tweet length.				
	const cl_uint TEXT_OFFSETS[2] = { 280 ,20 };	// number of characters in the location and tweet text variables
	const int NUM_TIME_VARS = 5;
	const int NUM_SCORE_VARS = 3;
	const int NUM_TEXT_VARS = 2;
	cl_uint batchSize = 50000;
	cl_uint numRecords;
	cl_uint numVars;
	cl_uint numBatches;
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
	// todo: determine which statistics to derive.  Should this be a separate file?  It will be used by a separate class(es) than the
	// parsedCSV struct
};

// end of parsedStruct.h