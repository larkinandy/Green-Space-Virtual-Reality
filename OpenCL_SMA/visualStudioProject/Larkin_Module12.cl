//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Dan Ginsburg, Timothy Mattson
// ISBN-10:   ??????????
// ISBN-13:   ?????????????
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/??????????
//            http://www.????????.com
//

// simple.cl
//
//    This is a simple example demonstrating buffers and sub-buffer usage

__kernel void rect_based_avg(__global int * buffer, __global float * output, const int kernelArg)
{
	size_t id = get_global_id(0);
	float sum = (buffer[0] + buffer[1] + buffer[2] + buffer[3])/4.0f;
	output[kernelArg] = sum;
}