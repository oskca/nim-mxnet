type
  mx_uint* = cuint
  mx_float* = cfloat
  PredictorHandle* = pointer
  NDListHandle* = pointer

proc MXGetLastError*(): cstring {.importc: "MXGetLastError", dynlib: mxnet.}
proc MXPredCreate*(symbol_json_str: cstring; param_bytes: pointer; param_size: cint;
                  dev_type: cint; dev_id: cint; num_input_nodes: mx_uint;
                  input_keys: cstringArray; input_shape_indptr: ptr mx_uint;
                  input_shape_data: ptr mx_uint; `out`: ptr PredictorHandle): cint {.
    importc: "MXPredCreate", dynlib: mxnet.}
proc MXPredCreatePartialOut*(symbol_json_str: cstring; param_bytes: pointer;
                            param_size: cint; dev_type: cint; dev_id: cint;
                            num_input_nodes: mx_uint; input_keys: cstringArray;
                            input_shape_indptr: ptr mx_uint;
                            input_shape_data: ptr mx_uint;
                            num_output_nodes: mx_uint; output_keys: cstringArray;
                            `out`: ptr PredictorHandle): cint {.
    importc: "MXPredCreatePartialOut", dynlib: mxnet.}
proc MXPredReshape*(num_input_nodes: mx_uint; input_keys: cstringArray;
                   input_shape_indptr: ptr mx_uint; input_shape_data: ptr mx_uint;
                   handle: PredictorHandle; `out`: ptr PredictorHandle): cint {.
    importc: "MXPredReshape", dynlib: mxnet.}
proc MXPredGetOutputShape*(handle: PredictorHandle; index: mx_uint;
                          shape_data: ptr ptr mx_uint; shape_ndim: ptr mx_uint): cint {.
    importc: "MXPredGetOutputShape", dynlib: mxnet.}
proc MXPredSetInput*(handle: PredictorHandle; key: cstring; data: ptr mx_float;
                    size: mx_uint): cint {.importc: "MXPredSetInput", dynlib: mxnet.}
proc MXPredForward*(handle: PredictorHandle): cint {.importc: "MXPredForward",
    dynlib: mxnet.}
proc MXPredPartialForward*(handle: PredictorHandle; step: cint; step_left: ptr cint): cint {.
    importc: "MXPredPartialForward", dynlib: mxnet.}
proc MXPredGetOutput*(handle: PredictorHandle; index: mx_uint; data: ptr mx_float;
                     size: mx_uint): cint {.importc: "MXPredGetOutput", dynlib: mxnet.}
proc MXPredFree*(handle: PredictorHandle): cint {.importc: "MXPredFree", dynlib: mxnet.}
proc MXNDListCreate*(nd_file_bytes: cstring; nd_file_size: cint;
                    `out`: ptr NDListHandle; out_length: ptr mx_uint): cint {.
    importc: "MXNDListCreate", dynlib: mxnet.}
proc MXNDListGet*(handle: NDListHandle; index: mx_uint; out_key: cstringArray;
                 out_data: ptr ptr mx_float; out_shape: ptr ptr mx_uint;
                 out_ndim: ptr mx_uint): cint {.importc: "MXNDListGet", dynlib: mxnet.}
proc MXNDListFree*(handle: NDListHandle): cint {.importc: "MXNDListFree",
    dynlib: mxnet.}