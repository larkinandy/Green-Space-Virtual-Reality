#include "testDriver.hpp"
///
//	main() function
//

int main(int argc, char** argv)
{
	SMA_Analyzer analyzer = SMA_Analyzer();
	
	int * testData1 = NULL;
	analyzer.createInput(&testData1, 16);
	analyzer.getAverage(16, testData1);

	analyzer.createInput(&testData1, 16);
	analyzer.getAverage(16, testData1);

	std::cout << "Program completed successfully" << std::endl;
	return 0;
	

}
