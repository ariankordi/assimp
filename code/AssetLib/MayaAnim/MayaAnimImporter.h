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

/** @file MayaAnimImporter.h
 *  @brief Importer for Autodesk Maya ASCII animation curve files (.anim).
 */
#pragma once
#ifndef AI_MAYAANIMIMPORTER_H_INC
#define AI_MAYAANIMIMPORTER_H_INC

#ifndef ASSIMP_BUILD_NO_MAYAANIM_IMPORTER

#include <assimp/BaseImporter.h>

#include <string>

struct aiScene;

namespace Assimp {

/// Importer for Maya's ASCII ".anim" animation-curve format (animImportExport
/// plug-in output, also written by BrawlCrate and other game-modding tools).
///
/// The format carries only per-attribute keyframe curves (translateX,
/// rotateY, ...) addressed by node name; it has no meshes and no usable
/// hierarchy. The importer therefore produces a flat node list under a dummy
/// root plus one aiAnimation, with the curves baked into per-frame keys.
///
/// Attributes the file leaves unkeyed keep their bind value in Maya. Since
/// that value is not in the file, a caller can supply a scene to read it from
/// via AI_CONFIG_IMPORT_MAYA_ANIM_REFERENCE_SCENE; otherwise 0 (translate,
/// rotate) or 1 (scale) is assumed.
class MayaAnimImporter : public BaseImporter {
public:
    /// BaseImporter override.
    bool CanRead(const std::string &pFile, IOSystem *pIOHandler, bool checkSig) const override;

    /// BaseImporter override.
    void SetupProperties(const Importer *pImp) override;

    /// BaseImporter override.
    const aiImporterDesc *GetInfo() const override;

protected:
    /// BaseImporter override.
    void InternReadFile(const std::string &pFile, aiScene *pScene, IOSystem *pIOHandler) override;

private:
    /// Optional bind-pose scene for unkeyed attributes; not owned.
    const aiScene *mReferenceScene = nullptr;
};

} // namespace Assimp

#endif // ASSIMP_BUILD_NO_MAYAANIM_IMPORTER
#endif // AI_MAYAANIMIMPORTER_H_INC
