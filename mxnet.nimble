# Package

version       = "0.1.0"
author        = "oskca"
description   = "Nim binding for MxNet predictor API"
license       = "Apache2"
srcDir        = "src"

# Dependencies

requires "nim >= 0.18.0"

import ospaths

proc genRaw(fpath, ofn: string) =
  let precompiled = staticExec("gcc -E "& fpath)
  var fp = getTempDir() / "out.E.h"
  writeFile(fp, precompiled)
  exec "c2nim --dynlib:mxnet -o:src/" & ofn & " " & fp
  rmFile(fp)


task gen, "Generate raw nim bindings":
  genRaw("/usr/local/include/mxnet/c_predict_api.h", "private/raw.nim")
