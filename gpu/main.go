package main

import (
	"fmt"
	"github.com/Inkeliz/go-opencl/opencl"
	"unsafe"
)

func main() {
	const (
		dataSize = 32

		programCode = `
kernel void kern(global float* out)
{
        size_t i = get_global_id(0);
        out[i] = i * i;
}
`
	)

	var device = getFirstDevice(opencl.DeviceTypeGPU)

	context, err := device.CreateContext()
	if err != nil {
		panic(err)
	}
	defer context.Release()

	var commandQueue opencl.CommandQueue
	commandQueue, err = context.CreateCommandQueue(device)
	if err != nil {
		panic(err)
	}
	defer commandQueue.Release()

	var program opencl.Program
	program, err = context.CreateProgramWithSource(programCode)
	if err != nil {
		panic(err)
	}
	defer program.Release()

	var log string
	err = program.Build(device, &log)
	if err != nil {
		fmt.Println(log)
		panic(err)
	}

	kernel, err := program.CreateKernel("kern")
	if err != nil {
		panic(err)
	}
	defer kernel.Release()

	buffer, err := context.CreateBuffer([]opencl.MemFlags{opencl.MemWriteOnly}, dataSize*4)
	if err != nil {
		panic(err)
	}
	defer buffer.Release()

	err = kernel.SetArg(0, buffer.Size(), &buffer)
	if err != nil {
		panic(err)
	}

	err = commandQueue.EnqueueNDRangeKernel(kernel, 1, []uint64{dataSize})
	if err != nil {
		panic(err)
	}

	commandQueue.Flush()
	commandQueue.Finish()

	data := make([]float32, dataSize)

	err = commandQueue.EnqueueReadBuffer(buffer, true, dataSize*4, unsafe.Pointer(&data[0]))
	if err != nil {
		panic(err)
	}

	for _, item := range data {
		fmt.Printf("%v ", item)
	}
	fmt.Println()
	// Output:
	// 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
}

func getFirstDevice(deviceType opencl.DeviceType) opencl.Device {
	platforms, err := opencl.GetPlatforms()
	if err != nil {
		panic(err)
	}

	for _, platform := range platforms {
		var devices []opencl.Device
		devices, err = platform.GetDevices(deviceType)
		if err != nil {
			panic(err)
		}

		for _, device := range devices {
			var available bool
			err = device.GetInfo(opencl.DeviceAvailable, &available)
			if err == nil && available {
				return device
			}
		}
	}

	panic("No device found")
	return opencl.Device{}
}
