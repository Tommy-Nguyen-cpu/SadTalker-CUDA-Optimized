import tensorrt as trt # Import tensorrt to be used to optimize deep learning
import pycuda.driver as cuda # Used to initialize stream and allocate memory for tensors in CUDA.
import pycuda.autoinit # Automatically initialize CUDA drivers so CUDA can be used.

class TensorRTWrapper():
    def __init__(self):
        self.LOGGER = trt.Logger(trt.Logger.WARNING) # Logs any warning messages output by tensorrt.
        self.builder = trt.Builder(self.LOGGER) # Used to make our network and engine.
        self.engine = None

    def create_engine(self, max_workspace_size=1<<30, onnx_file_path = '../SadTalker/kp_detector.onnx'):
        # Creates a computational graph that will be used by our builder to make our engine. Think of this like a blueprint for our network.
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) # We are telling the network that programmers can explicitly state the type of every tensor and the input and output.

        self.parser = trt.OnnxParser(self.network, self.LOGGER) # Used to parse info from our ONNX file.

        # Open our ONNX file and parse information that can be used to build our engine.
        with open(onnx_file_path, "rb") as f:
            if not self.parser.parse(f.read()): # If parsing failed, print out errors.
                for error in range(self.parser.num_errors):
                    print(f"ERROR: {self.parser.get_error(error)}")
                return None
        
        # Configure settings for builder so that it knows how we want it to setup the engine.
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size) # Sets the memory size of our workspace so it doesn't exceed it.
        self.config.set_flag(trt.BuilderFlag.FP16) # Use FP16 if GPU supports it.

        self.engine = self.builder.build_serialized_network(self.network, self.config) # Build our engine based on our computational graph and engine configuration settings.

    # Straightforward, write the engine to the file.
    def save_engine(self, file_path = "kp_detector.engine"):
        with open(file_path, "wb") as f:
            f.write(self.engine)
    
    def load_engine(self, file_path = "kp_detector.engine"):
        runtime = trt.Runtime(self.LOGGER) # Loads in serialized deep learning model so we can deserialize it (i.e. convert from bytes/nums to ICudaEngine).
        with open(file_path, "rb") as f:
            engine_data = f.read()
        self.engine = runtime.deserialize_cuda_engine(engine_data) # Converts bytes/ints back into ICudaEngine.

    def __call__(self, input, outputs):
        stream = cuda.Stream() # Initialize CUDA data stream.
        d_input = cuda.mem_alloc(1 * input.nbytes) # Allocates memory in GPU for input data. Grabs the size of the input data in bytes.
        
        if type(outputs) is list:
            d_outputs = [cuda.mem_alloc(1 * output.nbytes) for output in outputs] # Allocates memory in GPU for output data.
        else:
            d_outputs = cuda.mem_alloc(1 * outputs.nbytes)

        tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)] # Grabs the name of the input/output tensors.
        
        context = self.engine.create_execution_context() # Context used for inference. Might be worth exploring having multiple contexts so multiple batches can be inferred simultaneously.
        context.set_tensor_address(tensor_names[0], int(d_input)) # Sets where in the GPU memory the input tensor will be stored and used.
        
        if type(outputs) is list:
            [context.set_tensor_address(tensor_names[i], int(d_outputs[i-1])) for i in range(1, len(tensor_names))] # Sets in the GPU memory where the output tensor will be stored and used.
        else:
            context.set_tensor_address(tensor_names[1], int(d_outputs))
        
        cuda.memcpy_htod_async(d_input, input, stream) # Copies the input "input" to the GPU memory address "d_input". Serializes the input through the stream "stream".
        
        context.execute_async_v3(stream.handle) # Performs inference on the GPU asynchronously.

        if type(outputs) is list:
            [cuda.memcpy_dtoh_async(outputs[i], d_outputs[i], stream) for i in range(len(outputs))]
        else:
            cuda.memcpy_dtoh_async(outputs, d_outputs, stream) # Copies output tensor back into our parameter "output" from the GPU memory address "d_output". Serializes output through the stream "stream".

        stream.synchronize() # Waits for all activity on the stream to finish before continuing on in the method.