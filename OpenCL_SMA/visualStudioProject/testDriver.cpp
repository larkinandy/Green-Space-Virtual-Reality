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


void testAverageTest() {

	//_CrtSetDbgFlag(_CRTDBG_LEAK_CHECK_DF);

	int * testData1 = NULL;
	int *testData2 = NULL;
	float * output1 = NULL;
	float * output2 = NULL;

	SMA_Analyzer analyzer = SMA_Analyzer();
		createInput(16, &testData1);


		analyzer.getAverage(16, testData1, &output1);

		createInput(16, &testData2);
		analyzer.getAverage(16, testData2, &output2);

	printOutput(4, output2);

	free(testData1);
	free(testData2);
	free(output1);
	free(output2);

	std::cout << "Program completed successfully" << std::endl;


}

int main(int argc, char** argv)
{
	SMA_Analyzer analyzer = SMA_Analyzer();
	analyzer.parseCSV("TestData_OpenCL_SMA2.csv");
	//testAverageTest();



}
