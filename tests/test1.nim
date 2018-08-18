import mxnet

let symbol = readFile("./go_test-symbol.json")
let params = readFile("./go_test-0001.params")
let p = newPredictor(symbol, params,
    Device(Type:CPU, Id:0),
    @[InputNode(Key: "data", Shape: @[1, 10]),
      InputNode(Key: "softmax_label", Shape: @[1, 10])]
    )

p.free()
