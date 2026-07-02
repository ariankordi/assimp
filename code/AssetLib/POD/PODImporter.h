/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2026, assimp team

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

/** @file PODImporter.h
 *  @brief Importer for PowerVR POD (.pod) files, built on top of the
 *  PVRAssets loader from the PowerVR Native SDK.
 */
#pragma once
#ifndef AI_PODIMPORTER_H_INC
#define AI_PODIMPORTER_H_INC

#ifndef ASSIMP_BUILD_NO_POD_IMPORTER

#include <assimp/BaseImporter.h>

#include <vector>

struct aiNode;

namespace pvr {
namespace assets {
class Model;
} // namespace assets
} // namespace pvr

namespace Assimp {

/// Importer for PowerVR POD mesh, skeleton and animation data.
class PODImporter : public BaseImporter {
public:
    /// BaseImporter override.
    bool CanRead(const std::string &pFile, IOSystem *pIOHandler, bool checkSig) const override;

protected:
    /// BaseImporter override.
    void InternReadFile(const std::string &pFile, aiScene *pScene, IOSystem *pIOHandler) override;

    /// BaseImporter override.
    const aiImporterDesc *GetInfo() const override;

private:
    void BuildNodes(pvr::assets::Model &model, aiScene *pScene);
    void BuildMeshesAndMaterials(pvr::assets::Model &model, aiScene *pScene);
    void BuildAnimations(pvr::assets::Model &model, aiScene *pScene);

    /// Maps a POD node index to the aiNode created for it; populated by BuildNodes
    /// and consumed by BuildMeshesAndMaterials (bone nodes) and BuildAnimations.
    std::vector<aiNode *> mNodesByIndex;
};

} // namespace Assimp

#endif // ASSIMP_BUILD_NO_POD_IMPORTER
#endif // AI_PODIMPORTER_H_INC
