import tensorrt as trt # Import tensorrt to be used to optimize deep learning
import torch

class TensorRTWrapper():
    def __init__(self):
        self.LOGGER = trt.Logger(trt.Logger.WARNING) # Logs any warning messages output by tensorrt.
        self.builder = trt.Builder(self.LOGGER) # Used to make our network and engine.
        self.runtime = trt.Runtime(self.LOGGER) # Loads in serialized deep learning model so we can deserialize it (i.e. convert from bytes/nums to ICudaEngine).
        self.engine = None
        self.context = None

    def create_engine(self, max_workspace_size=1<<30, onnx_file_path = '../SadTalker/kp_detector.onnx'):
        # Creates a computational graph that will be used by our builder to make our engine. Think of this like a blueprint for our network.
        self.network = self.builder.create_network() # We are telling the network that programmers can explicitly state the type of every tensor and the input and output.

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

        engine_memory_data = self.builder.build_serialized_network(self.network, self.config) # Build our engine based on our computational graph and engine configuration settings.
        self.engine = self.runtime.deserialize_cuda_engine(engine_memory_data)
        self.context = self.engine.create_execution_context() # Context used for inference. Might be worth exploring having multiple contexts so multiple batches can be inferred simultaneously.

    # Straightforward, write the engine to the file.
    def save_engine(self, file_path = "kp_detector.engine"):
        with open(file_path, "wb") as f:
            f.write(self.engine.serialize())
    
    def load_engine(self, file_path = "kp_detector.engine"):
        engine_data = None
        with open(file_path, "rb") as f:
            engine_data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_data) # Converts bytes/ints back into ICudaEngine.
        self.context = self.engine.create_execution_context() # Context used for inference. Might be worth exploring having multiple contexts so multiple batches can be inferred simultaneously.

        tensor_names = [self.engine.get_tensor_name(i)
                for i in range(self.engine.num_io_tensors)]
        print(f">>> {file_path} valid I/O tensor names:", tensor_names)
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            print(f"{i:2d} {name:20s} {shape}")

    def __call__(self, input, outputs, device="cuda"):
        # --- prepare inputs ---
        stream = torch.cuda.Stream()

        # keep references so tensors aren't freed
        input_tensors = []
        d_inputs = []
        
        # support list, dict, or single
        if isinstance(input, list):
            items = input
        else:
            items = [input]

        for inp in items:
            t_inp = torch.from_numpy(inp).to(device)
            input_tensors.append(t_inp)
            d_inputs.append(int(t_inp.data_ptr()))

        # --- prepare outputs ---
        output_tensors = []
        d_outputs = []
        if isinstance(outputs, list):
            outs = outputs
        else:
            outs = [outputs]

        for out in outs:
            t_out = torch.from_numpy(out).to(device)
            output_tensors.append(t_out)
            d_outputs.append(int(t_out.data_ptr()))

        # fetch binding names
        tensor_names = [self.engine.get_tensor_name(i)
                        for i in range(self.engine.num_io_tensors)]

        # --- bind inputs ---
        next_pos = 0
        for ptr in d_inputs:
            self.context.set_tensor_address(tensor_names[next_pos], ptr)
            next_pos += 1

        # --- bind outputs ---
        for ptr in d_outputs:
            self.context.set_tensor_address(tensor_names[next_pos], ptr)
            next_pos += 1

        # --- run ---
        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

        # --- return ---
        results = [t.cpu() for t in output_tensors]
        return results if len(results) > 1 else results[0]
