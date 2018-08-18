# nim_mxnet
# Copyright oskca
# Nim binding for MxNet predictor API
{.passL:"-lpthread".}
const
  mxnet = "libmxnet.so"

include mxnet/raw

import strformat

type
  DeviceType* = enum
    CPU = 1
    GPU = 2
  Device* = ref object
    Type*: DeviceType
    Id*: int
  InputNode* = ref object
    Key*: string
    Shape*: seq[int]
  Predictor* = ref object
    h*: PredictorHandle
    keys: seq[string]
    shapeIdx: seq[mx_uint]
    shapeData: seq[mx_uint]

# proc newPredictor*(symbol:cstring, 
#   params: seq[byte], 
#   dev: Device,
  # nodes: seq[InputNode]) =
  

proc newPredictor*(symbol, params: string, dev: Device, nodes:seq[InputNode]): Predictor = 
  new(result)
  result.keys = @[]
  result.shapeIdx = @[0.mx_uint]
  result.shapeData = @[]
  var j:mx_uint = 0
  for n in nodes:
    result.keys.add(n.Key)
    j+=len(n.Shape).mx_uint
    result.shapeIdx.add(j)
    # result.shapeData.add n.Shape
    for s in n.Shape:
      result.shapeData.add mx_uint(s)
  let ret =  MXPredCreate(
      symbol.cstring,
      pointer(params[0].unsafeAddr),
      cint(params.len),
      cint(dev.Type),
      cint(dev.Id),
      mx_uint(nodes.len),
      allocCStringArray(result.keys),
      result.shapeIdx[0].addr,
      result.shapeData[0].addr,
      result.h.addr
    )
  if ret < 0:
    raise newException(AssertionError, fmt"MXPredCreate failed: {$MXGetLastError()}") 

proc free*(p: Predictor) = 
  let ret = p.h.MXPredFree()
  if ret < 0 :
    raise newException(AssertionError, fmt"MXPredFree failed: {$MXGetLastError()}") 

proc setInput*(p: Predictor, key: string, data: seq[float32]) = 
  ## set the input data of predictor
  ## param key The name of input node to set
  ## param data The float data to be set
  let ret = MxPredSetInput(
    p.h, 
    key.cstring(),
    data[0].unsafeAddr(),
    mx_uint(len(data)))
  if ret < 0 :
    raise newException(AssertionError, fmt"MxPredSetInput failed: {$MXGetLastError()}") 

proc forward*(p: Predictor) =
  ## run a forward pass after SetInput
  let ret = MXPredForward(p.h)
  if ret < 0 : raise newException(AssertionError, fmt"MXPredGetOutputShape failed: {$MXGetLastError()}")

proc getOutputShape(p: Predictor, index: int): seq[int] = 
  ## get the shape of output node
  ## param index The index of output node, set to 0 if there is only one output
  var shapeData: ptr mx_uint
  var dim: mx_uint
  let ret = MXPredGetOutputShape(
    p.h,
    index.mx_uint,
    shapeData.addr,
    dim.addr
    )
  if ret < 0 : raise newException(AssertionError, fmt"MXPredGetOutputShape failed: {$MXGetLastError()}") 
  result = newSeq[int]()
  var buf = cast[ptr array[0xffffffff, mx_uint]](shapeData)
  for i in 0..<dim:
    result.add int(buf[][i])

proc getOutput*(p: Predictor, index: int): seq[float32] =
  ## get the output value of prediction
  ## param index The index of output node, set to 0 if there is only one output
  let shape = p.getOutputShape(index)
  var size = 1
  for v in shape:
    size *= v
  result = newSeq[float32](size)
  let ret = MXPredGetOutput(
    p.h,
    mx_uint(index),
    cast[ptr mx_float](result[0].addr),
    mx_uint(size)
    )
  if ret < 0 : raise newException(AssertionError, fmt"MXPredGetOutputShape failed: {$MXGetLastError()}") 
