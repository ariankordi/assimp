/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2025, assimp team

All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the
following conditions are met:

* Redistributions of source code must retain the above
  copyright notice, this list of conditions and the
  following disclaimer.

* Redistributions in binary form must reproduce the above
  copyright notice, this list of conditions and the
  following disclaimer in the documentation and/or other
  materials provided with the distribution.

* Neither the name of the assimp team, nor the names of its
  contributors may be used to endorse or promote products
  derived from this software without specific prior
  written permission of the assimp team.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

----------------------------------------------------------------------
*/

/** @file IQMExporter.h
 *  @brief Inter-Quake Model (IQM v2) exporter.
 *
 *  Writes meshes, the skeleton (only the nodes that skin the meshes), and node animations
 *  sampled at a fixed frame rate. Options: #AI_CONFIG_EXPORT_IQM_Y_UP and
 *  #AI_CONFIG_EXPORT_IQM_FRAMERATE.
 */
#pragma once
#ifndef AI_IQMEXPORTER_H_INC
#define AI_IQMEXPORTER_H_INC

#include <assimp/types.h>

struct aiScene;

namespace Assimp {

class IOSystem;
class ExportProperties;

/// Exports @p pScene to the IQM file @p pFile. Registered as format id "iqm".
void ExportSceneIQM(const char* pFile, IOSystem* pIOSystem, const aiScene* pScene,
        const ExportProperties* pProperties);

} // namespace Assimp

#endif // AI_IQMEXPORTER_H_INC
