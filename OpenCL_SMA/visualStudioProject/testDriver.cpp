#include "testDriver.hpp"
///
//	main() function
//

// print results of kernel operation
void printOutput(int numElements, float * dataset)
{
	std::cout << "averages of sequential 4 elements for an input array with " << numElements << " elements" << std::endl;
	// Display output in rows
	for (int index = 0; index < numElements; index++)
	{
		std::cout << " " << dataset[index];
	}
	std::cout << std::endl;
}

void createInput(int numElements, int **dataset) 
{
	*dataset = (int*)malloc(sizeof(int)* numElements);
	for (int index = 0; index < numElements; index++) {
		(*dataset)[index] = index;
	}
}


int main(int argc, char** argv)
{
	
	
	SMA_Analyzer analyzer = SMA_Analyzer();

	// for measuring the elapsed time to completion
	typedef std::chrono::high_resolution_clock Clock;
	typedef std::chrono::milliseconds milliseconds;
	Clock::time_point t0 = Clock::now();

	analyzer.parseCSV("TestData_OpenCL_SMA_100.csv");

	Clock::time_point t1 = Clock::now();
	milliseconds ms = std::chrono::duration_cast<milliseconds>(t1 - t0);
	std::cout << "Time to completion: " << ms.count() << "ms\n" << std::endl;



}
