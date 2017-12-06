/* testDriver.cpp
* Implementation file used to test and debug OpenCL_SMA
* Author: Andrew Larkin
* December 5, 2017 */

#include "testDriver.hpp"

//	main() function
int main(int argc, char** argv)
{
	SMA_Analyzer analyzer = SMA_Analyzer();

	// for measuring the elapsed time to completion
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::milliseconds milliseconds;
	Clock::time_point t0 = Clock::now();

	// call csv parser block 1 of the OpenCL_SMA project
	analyzer.parseCSV("TestData_OpenCL_SMA_100000.csv");

	Clock::time_point t1 = Clock::now();
	milliseconds ms = std::chrono::duration_cast<milliseconds>(t1 - t0);
	std::cout << "Time to completion: " << ms.count() << "ms\n" << std::endl;
}

// end of testDriver.cpp