import tensorrt as trt # Import tensorrt to be used to optimize deep learning
import torch
import os
import numpy as np

class TensorRTWrapper():
    def __init__(self):
        self.LOGGER = trt.Logger(trt.Logger.WARNING) # Logs any warning messages output by tensorrt.
        self.builder = trt.Builder(self.LOGGER) # Used to make our network and engine.
        self.runtime = trt.Runtime(self.LOGGER) # Loads in serialized deep learning model so we can deserialize it (i.e. convert from bytes/nums to ICudaEngine).
        self.engine = None
        self.context = None

    def create_engine(self, max_workspace_size=1<<35, onnx_file_path = '../SadTalker/kp_detector.onnx'):
        # Creates a computational graph that will be used by our builder to make our engine. Think of this like a blueprint for our network.
        # self.network = self.builder.create_network() # We are telling the network that programmers can explicitly state the type of every tensor and the input and output.
        EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(EXPLICIT_BATCH)

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


    # =========================
    # LOAD ENGINE (fast)
    # =========================
    def load_engine(self, path):
        with open(path, "rb") as f:
            engine_data = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        self._setup_io()

    # =========================
    # INTERNAL: setup I/O names
    # =========================
    def _setup_io(self):
        self.input_name = None
        self.output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)

        print("Input:", self.input_name)
        print("Outputs:", self.output_names)

    def allocate_buffers(self, input_shapes, output_shapes, dtype=torch.float16):
        """
        input_shapes: list of shapes (one per input binding)
        output_shapes: list of shapes (one per output binding)
        """

        if len(input_shapes) != len(self.input_names):
            raise ValueError(
                f"Expected {len(self.input_names)} inputs, got {len(input_shapes)}"
            )

        self.input_tensors = []
        for name, shape in zip(self.input_names, input_shapes):
            tensor = torch.empty(shape, device=self.device, dtype=dtype)
            self.input_tensors.append(tensor)

            # IMPORTANT: set shape per input (dynamic shape support)
            self.context.set_input_shape(name, tuple(shape))

        self.output_tensors = [
            torch.empty(shape, device=self.device, dtype=dtype)
            for shape in output_shapes
        ]

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
            if inp is not None:
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

        input_idx = 0
        output_idx = 0

        for name in tensor_names:
            mode = self.engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.INPUT:
                self.context.set_tensor_address(name, d_inputs[input_idx])
                input_idx += 1
            else:
                self.context.set_tensor_address(name, d_outputs[output_idx])
                output_idx += 1

        # --- run ---
        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()

        # --- return ---
        results = [t.cpu() for t in output_tensors]
        return results if len(results) > 1 else results[0]

class TensorRTWrapper2:
    def __init__(self, device="cuda"):
        self.device = device
        self.LOGGER = trt.Logger(trt.Logger.WARNING)

        self.builder = trt.Builder(self.LOGGER)
        self.runtime = trt.Runtime(self.LOGGER)

        self.engine = None
        self.context = None

        self.input_names = []
        self.output_names = []

        self.input_tensors = []
        self.output_tensors = []

        self.stream = torch.cuda.Stream()

        self.buffers_ready = False

    # =========================
    # BUILD ENGINE
    # =========================
    def create_engine(self, onnx_file_path, workspace=1 << 33, timing_cache_path=None):
        EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        network = self.builder.create_network(EXPLICIT_BATCH)
        parser = trt.OnnxParser(network, self.LOGGER)

        with open(onnx_file_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("ONNX parse failed")

        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)

        # This is the precision switch that is actually worth gating.
        if self.builder.platform_has_fast_fp16:
            self.config.set_flag(trt.BuilderFlag.FP16)

        for layer in network:
            if layer.name in {"final_conv", "to_rgb", "output"}:
                layer.precision = trt.DataType.FLOAT
                for i in range(layer.num_outputs):
                    layer.set_output_type(i, trt.DataType.FLOAT)
        
        # Spend more time searching for better tactics.
        self.config.builder_optimization_level = 5
        # Reuse tactic timings across builds.
        if timing_cache_path is None:
            timing_cache_path = onnx_file_path + ".timing_cache"

        if os.path.exists(timing_cache_path):
            with open(timing_cache_path, "rb") as f:
                cache = self.config.create_timing_cache(f.read())
        else:
            cache = self.config.create_timing_cache(b"")

        self.config.set_timing_cache(cache, False)

        engine_bytes = self.builder.build_serialized_network(network, self.config)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()

        self._setup_io()

    # =========================
    # LOAD ENGINE
    # =========================
    def load_engine(self, path):
        with open(path, "rb") as f:
            engine_data = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        self._setup_io()

    # Straightforward, write the engine to the file.
    def save_engine(self, file_path = "kp_detector.engine"):
        with open(file_path, "wb") as f:
            f.write(self.engine.serialize())
    # =========================
    # SETUP IO (FIXED)
    # =========================
    def _setup_io(self):
        self.input_names = []
        self.output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        print("Inputs:", self.input_names)
        print("Outputs:", self.output_names)

    # =========================
    # ALLOCATE BUFFERS
    # =========================
    def allocate_buffers(self, input_shapes, output_shapes, dtype=torch.float16):
        if len(input_shapes) != len(self.input_names):
            raise ValueError(
                f"Expected {len(self.input_names)} inputs, got {len(input_shapes)}"
            )

        self.input_tensors = []
        for name, shape in zip(self.input_names, input_shapes):
            t = torch.empty(shape, device=self.device, dtype=dtype)
            self.input_tensors.append(t)

            self.context.set_input_shape(name, tuple(shape))

        self.output_tensors = [
            torch.empty(shape, device=self.device, dtype=dtype)
            for shape in output_shapes
        ]

        self.buffers_ready = True

    # =========================
    # COPY INPUTS
    # =========================
    def copy_inputs(self, inputs):
        if not self.buffers_ready:
            raise RuntimeError("Call allocate_buffers first")

        if len(inputs) != len(self.input_tensors):
            raise ValueError(
                f"Expected {len(self.input_tensors)} inputs, got {len(inputs)}"
            )

        for src, dst in zip(inputs, self.input_tensors):
            if src is None:
                raise ValueError("TensorRT inputs cannot be None")

            if isinstance(src, np.ndarray):
                src = torch.from_numpy(src)

            # Keep dtype consistent with the engine buffers.
            src = src.to(device=self.device, dtype=dst.dtype)
            dst.copy_(src, non_blocking=True)

    # =========================
    # RUN INFERENCE
    # =========================
    def __call__(self):
        if not self.buffers_ready:
            raise RuntimeError("Call allocate_buffers first")

        # bind inputs
        for name, tensor in zip(self.input_names, self.input_tensors):
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        # bind outputs
        for name, tensor in zip(self.output_names, self.output_tensors):
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        return [t.cpu() for t in self.output_tensors]