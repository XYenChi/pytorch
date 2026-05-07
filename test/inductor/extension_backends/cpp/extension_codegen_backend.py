from torch._inductor.codegen import cpp, cpp_wrapper_cpu, wrapper
from torch._inductor.scheduler import BaseScheduling
from torch._inductor.virtualized import V


class ExtensionWrapperCodegen(wrapper.PythonWrapperCodegen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def create(
        is_subgraph,
        subgraph_name,
        parent_wrapper,
        partition_signatures=None,
    ):
        return ExtensionWrapperCodegen()

    def _generate_kernel_call_helper(
        self,
        kernel_name,
        call_args,
        *,
        device=None,
        triton=True,
        **kwargs,
    ):
        # The base PythonWrapperCodegen only knows about "cpu" and "mps" for
        # non-triton kernels. Treat the dummy extension_device as a CPU kernel
        # so calls into it can be generated.
        if not triton and device is not None and device.type == "extension_device":
            self.writeline(self.wrap_kernel_call(kernel_name, call_args))
            return
        return super()._generate_kernel_call_helper(
            kernel_name,
            call_args,
            device=device,
            triton=triton,
            **kwargs,
        )


class ExtensionCppWrapperCodegen(cpp_wrapper_cpu.CppWrapperCpu):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def create(
        is_subgraph,
        subgraph_name,
        parent_wrapper,
        partition_signatures=None,
    ):
        # CppWrapperCpu.create() hard-codes its own type, so override here so that
        # this subclass's get_device_include_path is actually used.
        return ExtensionCppWrapperCodegen()

    @staticmethod
    def get_device_include_path(device: str) -> str:
        # The extension backend reuses the CPU wrapper headers; there is no
        # bespoke <torch/csrc/inductor/cpp_wrapper/extension_device.h>.
        return cpp_wrapper_cpu.CppWrapperCpu.get_device_include_path("cpu")


class ExtensionScheduling(BaseScheduling):
    def __init__(self, scheduler):
        super().__init__(scheduler)
        self._scheduling = cpp.CppScheduling(scheduler)

    def can_fuse_vertical(self, node1, node2):
        return True

    def can_fuse_horizontal(self, node1, node2):
        return True

    def group_fn(self, sizes):
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    def codegen_template(self, template_node, epilogue_nodes):
        pass

    def codegen_node(self, node):
        self._scheduling.codegen_node(node)

    def codegen_sync(self):
        pass

    def flush(self):
        self._scheduling.flush()
