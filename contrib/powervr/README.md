# Vendored subset of the PowerVR Native SDK (and glm)

These files are used by the POD importer (`code/AssetLib/POD/`) to parse
PowerVR `.pod` models. They are copied **verbatim and unmodified** from
upstream; the directory layout mirrors the SDK's so that a real SDK checkout (or
a symlink to one) can be used instead:

```sh
cmake -DASSIMP_BUILD_POD_IMPORTER=ON -DPVR_SDK_DIR=/path/to/Native_SDK ...
```

If `PVR_SDK_DIR` is not set, this directory is used. A real SDK checkout must
have its glm submodule initialized (`external/glm/source`).

## Upstream

| Component | Source | Version |
|---|---|---|
| PowerVR Native SDK (`framework/PVRAssets`, `framework/PVRCore`) | https://github.com/powervr-graphics/Native_SDK | tag `R26.1-v5.16` |
| glm (`external/glm/source/glm`), a submodule of the SDK | https://github.com/g-truc/glm | commit `658d8960d081e0c9c312d49758c7ef919371b428` (0.9.9.4) |

Only the files transitively included by the six sources compiled in
`code/CMakeLists.txt` (`PODReader.cpp`, `Model.cpp`, `Mesh.cpp`, `Animation.cpp`,
`Camera.cpp`, `Light.cpp`) are present: 30 files from the SDK and the 205 glm
headers they pull in. If the importer starts including more, copy the extra files
from the same upstream versions.

## Licenses

### PowerVR Native SDK

The MIT License (MIT)
Copyright (c) Imagination Technologies Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.


### glm

glm is dual-licensed under the Happy Bunny License (Modified MIT) or the MIT
License, quoted from glm's `manual.md` at the commit above. The MIT License
applies here.

#### The MIT License

Copyright (c) 2005 - G-Truc Creation

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
