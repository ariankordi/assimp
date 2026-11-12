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

#include "PVRAssets/model/Animation.h"
#include "PVRCore/types/Types.h"
#include "assimp/Exceptional.h"
#include "assimp/anim.h"
#include "assimp/matrix4x4.h"
#include "assimp/mesh.h"
#include "assimp/quaternion.h"
#include "assimp/types.h"
#include "assimp/vector3.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>
#ifndef ASSIMP_BUILD_NO_POD_IMPORTER

#include "PODImporter.h"

#include <assimp/DefaultLogger.hpp>
#include <assimp/IOSystem.hpp>
#include <assimp/importerdesc.h>
#include <assimp/scene.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <unordered_map>

#include "PVRAssets/Model.h"
#include "PVRAssets/fileio/PODDefines.h"
#include "PVRAssets/fileio/PODReader.h"
#include "PVRCore/stream/BufferStream.h"

#include <assimp/Importer.hpp>
#include <assimp/config.h>

#include <cstring>
#include <optional>

using namespace pvr;
using namespace pvr::assets;

namespace Assimp {

static constexpr aiImporterDesc desc = {
    "PowerVR POD Importer",
    "",
    "",
    "Built on the PVRAssets loader from the PowerVR Native SDK.",
    aiImporterFlags_SupportBinaryFlavour,
    0,
    0,
    0,
    0,
    "pod"
};

const aiImporterDesc *PODImporter::GetInfo() const {
    return &desc;
}

bool PODImporter::CanRead(const std::string &pFile, IOSystem *pIOHandler, bool checkSig) const {
    if (!checkSig) {
        return SimpleExtensionCheck(pFile, "pod");
    }
    static const char *tokens[] = { "AB.POD.2.0" };
    return SearchFileHeaderForToken(pIOHandler, pFile, tokens, AI_COUNT_OF(tokens));
}

namespace
{

// Generic vertex component decoder, ported from PVRAssets' Helper::VertexRead
// (PVRAssets/Helper.cpp) so we don't have to link the rest of Helper.cpp, which
// pulls in the (unused, for POD) GltfReader/tinygltf dependency.
void ReadVertexComponent(const uint8_t *data, pvr::DataType type, uint32_t count, float *out) {
    out[0] = 0;
    if (count > 1) out[1] = 0;
    if (count > 2) out[2] = 0;
    if (count > 3) out[3] = 1;

    switch (type) {
    default:
        break;
    case pvr::DataType::Float32:
        for (uint32_t i = 0; i < count; ++i) out[i] = reinterpret_cast<const float *>(data)[i];
        break;
    case pvr::DataType::Fixed16_16:
        for (uint32_t i = 0; i < count; ++i) out[i] = reinterpret_cast<const int32_t *>(data)[i] * 1.0f / static_cast<float>(1 << 16);
        break;
    case pvr::DataType::Int32:
        for (uint32_t i = 0; i < count; ++i) out[i] = static_cast<float>(reinterpret_cast<const int32_t *>(data)[i]);
        break;
    case pvr::DataType::UInt32:
        for (uint32_t i = 0; i < count; ++i) out[i] = static_cast<float>(reinterpret_cast<const uint32_t *>(data)[i]);
        break;
    case pvr::DataType::Int8:
        for (uint32_t i = 0; i < count; ++i) out[i] = static_cast<float>(reinterpret_cast<const int8_t *>(data)[i]);
        break;
    case pvr::DataType::Int8Norm:
        for (uint32_t i = 0; i < count; ++i) out[i] = static_cast<float>(reinterpret_cast<const int8_t *>(data)[i]) / static_cast<float>((1 << 7) - 1);
        break;
    case pvr::DataType::UInt8:
        for (uint32_t i = 0; i < count; ++i) out[i] = static_cast<float>(reinterpret_cast<const uint8_t *>(data)[i]);
        break;
    case pvr::DataType::UInt8Norm:
        for (uint32_t i = 0; i < count; ++i) out[i] = static_cast<float>(reinterpret_cast<const uint8_t *>(data)[i]) / static_cast<float>((1 << 8) - 1);
        break;
    case pvr::DataType::Int16:
        for (uint32_t i = 0; i < count; ++i) out[i] = static_cast<float>(reinterpret_cast<const int16_t *>(data)[i]);
        break;
    case pvr::DataType::Int16Norm:
        for (uint32_t i = 0; i < count; ++i) out[i] = static_cast<float>(reinterpret_cast<const int16_t *>(data)[i]) / static_cast<float>((1 << 15) - 1);
        break;
    case pvr::DataType::UInt16:
        for (uint32_t i = 0; i < count; ++i) out[i] = static_cast<float>(reinterpret_cast<const uint16_t *>(data)[i]);
        break;
    }
}

uint32_t ReadFaceIndex(const uint8_t *data, IndexType type, uint32_t i) {
    if (type == IndexType::IndexType32Bit) {
        return reinterpret_cast<const uint32_t *>(data)[i];
    }
    return reinterpret_cast<const uint16_t *>(data)[i];
}

// ---------------------------------------------------------------------------
// Raw node transform scan and evaluation.
//
// PVRAssets' PODReader (v5.x) is not used for node transforms or animation:
// it stores every node's animation flags and index arrays into one shared,
// model-wide AnimationData (so they are effectively discarded), treats index-
// compressed key arrays as one key per frame, never builds a rest transform
// for matrix-animated nodes, and bakes its own quaternion sign convention into
// what it returns. Instead, node blocks are re-read here and evaluated with
// the semantics of the PowerVR SDK 3.5 runtime, CPVRTModelPOD::SetFrame /
// GetScalingMatrix / GetRotationMatrix / GetTranslationMatrix /
// GetTransformationMatrix / GetWorldMatrixNoCache in Tools/PVRTModelPOD.cpp,
// tag R15.1-v3.5 of https://github.com/powervr-graphics/Native_SDK
// (Copyright (c) Imagination Technologies Limited). Only the behaviour is
// reproduced; no SDK code is copied. Key points of that behaviour:
//  * a component's per-frame data is consulted only if its nAnimFlags bit
//    is set; otherwise element 0 is used even if more data exists;
//  * index arrays hold *float offsets* into the key array, not key numbers;
//  * scale keys are 7 floats wide (3 used);
//  * a node with matrix data uses it exclusively, ignoring S/R/T;
//  * matrices are stored in the SDK's row-vector layout (translation in
//    elements 12..14);
//  * frame 0 is the bind pose used for skinning (GetBoneWorldMatrix).
// Verified numerically against that runtime with the scratch tools in
// scratch/pod-roundtrip (pod35_dump vs assimp_anim_dump).

/// Node animation flag bits (SDK 3.5 EPODAnimationData).
constexpr uint32_t cPodHasPositionAni = 0x01;
constexpr uint32_t cPodHasRotationAni = 0x02;
constexpr uint32_t cPodHasScaleAni = 0x04;
constexpr uint32_t cPodHasMatrixAni = 0x08;

/// Floats per key for each node transform array.
constexpr size_t cPodPositionStride = 3;
constexpr size_t cPodRotationStride = 4;
constexpr size_t cPodScaleStride = 7;
constexpr size_t cPodMatrixStride = 16;

/// Bit set on a tag id to mark the end of the block it opened.
constexpr uint32_t cPodEndTagBit = pod::c_endTagMask;

/// Read-only view of a byte range (assimp builds as C++17, so no std::span).
struct ByteView {
    const uint8_t *data = nullptr; ///< First byte.
    size_t size = 0;               ///< Number of bytes.

    /// Returns the sub-range [offset, offset + count); caller validates bounds.
    ByteView Sub(size_t offset, size_t count) const { return { data + offset, count }; }
};

/// Little-endian 32-bit load from an unaligned position (POD is little-endian).
uint32_t LoadU32(const uint8_t *p) {
    uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

/// Copies a tag payload of 32-bit values into a vector (whole elements only).
template <typename T>
std::vector<T> LoadArray32(ByteView payload) {
    static_assert(sizeof(T) == 4);
    std::vector<T> out(payload.size / 4);
    if (!out.empty()) std::memcpy(out.data(), payload.data, out.size() * 4);
    return out;
}

/// Converts a tag payload to a string, dropping trailing NULs.
std::string LoadString(ByteView payload) {
    std::string s(reinterpret_cast<const char *>(payload.data), payload.size);
    while (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

/// Scans a POD byte stream for file-level history/options and the raw
/// transform data of every node block, in file order. The tag stream is
/// flat: every tag (container or data) is (id, length, payload) and each is
/// closed by (id | end bit, 0); containers simply have a zero length, so
/// skipping each payload by its length walks the whole file. Every length
/// is validated against the remaining buffer.
PODRawInfo ScanRawPod(ByteView file) {
    PODRawInfo info;
    PODRawNode *node = nullptr;
    size_t pos = 0;
    while (file.size - pos >= 8) {
        const uint32_t id = LoadU32(file.data + pos);
        const uint32_t length = LoadU32(file.data + pos + 4);
        pos += 8;
        if (length > file.size - pos) {
            throw DeadlyImportError("POD: tag ", id & ~cPodEndTagBit, " overruns the file");
        }
        const ByteView payload = file.Sub(pos, length);
        pos += length;

        if (id == (static_cast<uint32_t>(pod::e_sceneNode) | cPodEndTagBit)) {
            node = nullptr;
            continue;
        }
        if (id & cPodEndTagBit) continue;

        switch (id) {
        case pod::ExportOptions: info.exportOptions = LoadString(payload); break;
        case pod::FileHistory: info.history = LoadString(payload); break;
        case pod::e_sceneNumFrames:
            if (payload.size >= 4) info.numFrames = LoadU32(payload.data);
            break;
        case pod::e_sceneNode: node = &info.nodes.emplace_back(); break;
        default:
            if (!node) break;
            switch (id) {
            case pod::e_nodeName: node->name = LoadString(payload); break;
            case pod::e_nodeAnimationFlags:
                if (payload.size >= 4) node->animFlags = LoadU32(payload.data);
                break;
            // The deprecated single-value tags share storage with the animated
            // ones; with no flag bit set only element 0 is ever read.
            case pod::e_nodePosition:
            case pod::e_nodeAnimationPosition: node->position = LoadArray32<float>(payload); break;
            case pod::e_nodeRotation:
            case pod::e_nodeAnimationRotation: node->rotation = LoadArray32<float>(payload); break;
            case pod::e_nodeScale:
            case pod::e_nodeAnimationScale: node->scale = LoadArray32<float>(payload); break;
            case pod::e_nodeMatrix:
            case pod::e_nodeAnimationMatrix: node->matrix = LoadArray32<float>(payload); break;
            case pod::e_nodeAnimationPositionIndex: node->positionIdx = LoadArray32<uint32_t>(payload); break;
            case pod::e_nodeAnimationRotationIndex: node->rotationIdx = LoadArray32<uint32_t>(payload); break;
            case pod::e_nodeAnimationScaleIndex: node->scaleIdx = LoadArray32<uint32_t>(payload); break;
            case pod::e_nodeAnimationMatrixIndex: node->matrixIdx = LoadArray32<uint32_t>(payload); break;
            default: break;
            }
            break;
        }
    }
    return info;
}

/// Returns the float offset of @p frame's key in one transform array, as the
/// SDK 3.5 runtime selects it, or std::nullopt when the array is too short.
std::optional<size_t> KeyOffset(const std::vector<float> &data, const std::vector<uint32_t> &indices,
        bool animated, uint32_t frame, size_t stride, size_t width) {
    size_t offset = 0;
    if (animated) {
        if (!indices.empty()) {
            // Index arrays are float offsets, one per frame.
            offset = indices[std::min<size_t>(frame, indices.size() - 1)];
        } else {
            offset = static_cast<size_t>(frame) * stride;
        }
    }
    if (offset > data.size() || data.size() - offset < width) return std::nullopt;
    return offset;
}

/// One node's local transform at one frame, in assimp terms.
struct PODLocalTransform {
    aiVector3D scaling{ 1.0f, 1.0f, 1.0f };   ///< Per-axis scale.
    aiQuaternion rotation;                    ///< Rotation (assimp convention).
    aiVector3D position;                      ///< Translation.
    std::optional<aiMatrix4x4> matrix;        ///< Set for matrix nodes (S/R/T then hold its decomposition).
};

/// Converts a stored POD quaternion (x, y, z, w floats) to the rotation it
/// represents under @p convention.
aiQuaternion ToAiQuaternion(const float *q, PODQuaternionConvention convention) {
    if (convention == PODQuaternionConvention::Sdk) {
        // SDK 3.5 PVRTMatrixRotationQuaternionF (non-DX11 path) fills a row-
        // vector matrix whose effective rotation is that of the conjugate.
        return { q[3], -q[0], -q[1], -q[2] };
    }
    return { q[3], q[0], q[1], q[2] };
}

/// Converts 16 floats of a POD (row-vector layout) matrix to an aiMatrix4x4
/// (column-vector convention): element (row, col) = f[col * 4 + row].
aiMatrix4x4 ToAiMatrix(const float *f) {
    return aiMatrix4x4(f[0], f[4], f[8], f[12],
            f[1], f[5], f[9], f[13],
            f[2], f[6], f[10], f[14],
            f[3], f[7], f[11], f[15]);
}

/// Evaluates a node's local transform at an integer frame with SDK 3.5
/// semantics. @p warned is set when an out-of-range key had to be replaced
/// by element 0, so callers can log once per node.
PODLocalTransform EvaluateLocal(const PODRawNode &node, uint32_t frame, PODQuaternionConvention convention, bool &warned) {
    PODLocalTransform out;
    auto offsetOrFirst = [&](const std::vector<float> &data, const std::vector<uint32_t> &idx,
                                 uint32_t flag, size_t stride, size_t width) -> std::optional<size_t> {
        if (data.size() < width) return std::nullopt;
        std::optional<size_t> off = KeyOffset(data, idx, (node.animFlags & flag) != 0, frame, stride, width);
        if (!off) {
            warned = true;
            off = 0;
        }
        return off;
    };

    if (!node.matrix.empty()) {
        if (const auto off = offsetOrFirst(node.matrix, node.matrixIdx, cPodHasMatrixAni, cPodMatrixStride, cPodMatrixStride)) {
            out.matrix = ToAiMatrix(node.matrix.data() + *off);
            out.matrix->Decompose(out.scaling, out.rotation, out.position);
        } else {
            out.matrix = aiMatrix4x4();
        }
        return out;
    }
    if (const auto off = offsetOrFirst(node.scale, node.scaleIdx, cPodHasScaleAni, cPodScaleStride, 3)) {
        const float *s = node.scale.data() + *off;
        out.scaling = aiVector3D(s[0], s[1], s[2]);
    }
    if (const auto off = offsetOrFirst(node.rotation, node.rotationIdx, cPodHasRotationAni, cPodRotationStride, 4)) {
        out.rotation = ToAiQuaternion(node.rotation.data() + *off, convention);
    }
    if (const auto off = offsetOrFirst(node.position, node.positionIdx, cPodHasPositionAni, cPodPositionStride, 3)) {
        const float *p = node.position.data() + *off;
        out.position = aiVector3D(p[0], p[1], p[2]);
    }
    return out;
}

/// Composes a local transform into a matrix (T * R * S for SRT nodes, as the
/// SDK's GetWorldMatrixNoCache does). Deliberately not
/// aiMatrix4x4(scaling, rotation, position): that constructor scales the
/// rotation's *rows* (S * R), which disagrees with aiMatrix4x4::Decompose
/// (column scale, R * S) whenever the scale is non-uniform.
aiMatrix4x4 ToMatrix(const PODLocalTransform &t) {
    if (t.matrix) return *t.matrix;
    aiMatrix4x4 translation;
    aiMatrix4x4 scaling;
    aiMatrix4x4::Translation(t.position, translation);
    aiMatrix4x4::Scaling(t.scaling, scaling);
    return translation * aiMatrix4x4(t.rotation.GetMatrix()) * scaling;
}

/// True if any transform data is stored for the node.
bool HasTransformData(const PODRawNode &node) {
    return !node.matrix.empty() || !node.position.empty() || !node.rotation.empty() || !node.scale.empty();
}

/// Picks the quaternion convention for a file: Imagination's exporters
/// always write ExportOptions and FileHistory blocks.
PODQuaternionConvention DetectConvention(const PODRawInfo &raw) {
    if (!raw.history.empty() || !raw.exportOptions.empty()) return PODQuaternionConvention::Sdk;
    return PODQuaternionConvention::Raw;
}

// Composes a node's bind-pose (frame 0) world transform by walking its aiNode
// parent chain. Used instead of PVRAssets' precomputed
// Skeleton::invBindMatrices, which are baked with PVRAssets' own quaternion
// handling and ignore matrix-node rest transforms -- mixing them with this
// importer's node hierarchy would make bone world transform and inverse bind
// matrix mutually inconsistent, distorting the mesh even in bind pose.
aiMatrix4x4 ComputeWorldTransform(const aiNode *node) {
    aiMatrix4x4 world = node->mTransformation;
    for (const aiNode *parent = node->mParent; parent != nullptr; parent = parent->mParent) {
        world = parent->mTransformation * world;
    }
    return world;
}

unsigned int CountDescendants(const aiNode *n) {
    unsigned int count = n->mNumChildren;
    for (unsigned int i = 0; i < n->mNumChildren; ++i) {
        count += CountDescendants(n->mChildren[i]);
    }
    return count;
}

} // namespace

void PODImporter::BuildNodes(Model &model, const PODRawInfo &raw, aiScene *pScene) {
    const uint32_t numNodes = model.getNumNodes();
    mNodesByIndex.assign(numNodes, nullptr);
    if (raw.nodes.size() != numNodes) {
        throw DeadlyImportError("POD: raw node scan found ", raw.nodes.size(), " nodes, PVRAssets ", numNodes);
    }

    std::map<std::string, uint32_t> nameUsageCount;

    for (uint32_t i = 0; i < numNodes; ++i) {
        const Model::Node &node = model.getNode(i);
        std::string name = static_cast<const std::string &>(node.getName());
        if (name.empty()) {
            name = "_node" + std::to_string(i);
        }
        uint32_t &usageCount = nameUsageCount[name];
        if (usageCount > 0) {
            name += "_" + std::to_string(i);
        }
        ++usageCount;

        // Both readers walk node blocks in file order; a name mismatch would
        // mean the raw scan and PVRAssets disagree about which node is which.
        if (raw.nodes[i].name != static_cast<const std::string &>(node.getName())) {
            throw DeadlyImportError("POD: node ", i, " name mismatch between raw scan and PVRAssets");
        }

        // The rest transform is frame 0, which is also the pose the SDK uses
        // as the skinning bind pose.
        bool warned = false;
        aiNode *ainode = new aiNode(name);
        ainode->mTransformation = ToMatrix(EvaluateLocal(raw.nodes[i], 0, mConvention, warned));
        if (warned) {
            ASSIMP_LOG_WARN("POD: node \"", name, "\" has truncated transform data; using its first key");
        }
        mNodesByIndex[i] = ainode;
    }

    std::vector<aiNode *> rootLevelNodes;
    std::vector<uint32_t> rootLevelIndices; ///< POD node index of each rootLevelNodes entry.
    for (uint32_t i = 0; i < numNodes; ++i) {
        const uint32_t parentIdx = model.getNode(i).getParentID();
        if (parentIdx < numNodes) {
            aiNode *parent = mNodesByIndex[parentIdx];
            aiNode **newChildren = new aiNode *[parent->mNumChildren + 1];
            for (unsigned int c = 0; c < parent->mNumChildren; ++c) newChildren[c] = parent->mChildren[c];
            newChildren[parent->mNumChildren] = mNodesByIndex[i];
            delete[] parent->mChildren;
            parent->mChildren = newChildren;
            ++parent->mNumChildren;
            mNodesByIndex[i]->mParent = parent;
        } else {
            rootLevelNodes.push_back(mNodesByIndex[i]);
            rootLevelIndices.push_back(i);
        }
    }

    if (rootLevelNodes.size() == 1) {
        pScene->mRootNode = rootLevelNodes[0];
    } else {
        // POD's own top level can have several sibling nodes (e.g. mesh nodes
        // sitting alongside a joint hierarchy root), but assimp needs exactly
        // one aiScene::mRootNode. Rather than inventing a new node, reuse
        // whichever existing top-level node has the most descendants (in
        // practice, the joint hierarchy root) and reparent the rest under it
        // -- but only if that node's transform is identity and never
        // animated, so the reparenting is purely structural. Otherwise the
        // reparented siblings would inherit its transform: PVRGeoPOD output,
        // for example, puts the Z-up -> Y-up rotation on top-level nodes,
        // which turned every reparented mesh node by 90 degrees. Fabricate a
        // node only when no top-level node qualifies.
        size_t bestIdx = 0;
        unsigned int bestDescendants = 0;
        for (size_t i = 0; i < rootLevelNodes.size(); ++i) {
            const unsigned int count = CountDescendants(rootLevelNodes[i]);
            if (count > bestDescendants) {
                bestDescendants = count;
                bestIdx = i;
            }
        }

        aiNode *root = nullptr;
        if (bestDescendants > 0 && rootLevelNodes[bestIdx]->mTransformation.IsIdentity() &&
                raw.nodes[rootLevelIndices[bestIdx]].animFlags == 0) {
            root = rootLevelNodes[bestIdx];
        } else {
            root = new aiNode("POD_RootNode");
        }

        for (size_t i = 0; i < rootLevelNodes.size(); ++i) {
            if (rootLevelNodes[i] == root) continue;
            aiNode **newChildren = new aiNode *[root->mNumChildren + 1];
            for (unsigned int c = 0; c < root->mNumChildren; ++c) newChildren[c] = root->mChildren[c];
            newChildren[root->mNumChildren] = rootLevelNodes[i];
            delete[] root->mChildren;
            root->mChildren = newChildren;
            ++root->mNumChildren;
            rootLevelNodes[i]->mParent = root;
        }
        pScene->mRootNode = root;
    }
}

void PODImporter::BuildMeshesAndMaterials(Model &model, aiScene *pScene) {
    // Materials: minimal name-only translation.
    const uint32_t numMaterials = model.getNumMaterials();
    if (numMaterials > 0) {
        pScene->mNumMaterials = numMaterials;
        pScene->mMaterials = new aiMaterial *[numMaterials];
        for (uint32_t i = 0; i < numMaterials; ++i) {
            aiMaterial *mat = new aiMaterial();
            aiString name(static_cast<const std::string &>(model.getMaterial(i).getName()));
            mat->AddProperty(&name, AI_MATKEY_NAME);
            pScene->mMaterials[i] = mat;
        }
    } else {
        pScene->mNumMaterials = 1;
        pScene->mMaterials = new aiMaterial *[1];
        pScene->mMaterials[0] = new aiMaterial();
    }

    const uint32_t numMeshes = model.getNumMeshes();
    pScene->mNumMeshes = numMeshes;
    pScene->mMeshes = new aiMesh *[numMeshes];

    // Frame-0 world transform of the (first) node instancing each mesh. The
    // SDK's GetBoneWorldMatrix skins as boneWorld(f) * boneWorld(0)^-1 *
    // meshNodeWorld(0), so the mesh node's transform belongs in the offset
    // matrix too. Mesh nodes occupy indices [0, getNumMeshNodes()).
    std::vector<aiMatrix4x4> meshNodeWorld(numMeshes);
    {
        std::vector<bool> seen(numMeshes, false);
        for (uint32_t i = 0; i < model.getNumMeshNodes() && i < mNodesByIndex.size(); ++i) {
            const uint32_t meshIndex = model.getMeshNode(i).getObjectId();
            if (meshIndex < numMeshes && !seen[meshIndex]) {
                seen[meshIndex] = true;
                meshNodeWorld[meshIndex] = ComputeWorldTransform(mNodesByIndex[i]);
            }
        }
    }

    for (uint32_t m = 0; m < numMeshes; ++m) {
        Mesh &mesh = model.getMesh(m);

        if (mesh.getPrimitiveType() != pvr::PrimitiveTopology::TriangleList) {
            throw DeadlyImportError("POD: only TriangleList primitive topology is currently supported (mesh ", m, ")");
        }

        aiMesh *aimesh = new aiMesh();
        aimesh->mPrimitiveTypes = aiPrimitiveType_TRIANGLE;

        const uint32_t numVertices = mesh.getNumVertices();
        const uint32_t numFaces = mesh.getNumFaces();
        aimesh->mNumVertices = numVertices;
        aimesh->mVertices = new aiVector3D[numVertices];

        const Mesh::VertexAttributeData *posAttr = mesh.getVertexAttributeByName("POSITION");
        if (!posAttr) {
            throw DeadlyImportError("POD: mesh ", m, " has no POSITION vertex attribute");
        }
        const Mesh::VertexAttributeData *normalAttr = mesh.getVertexAttributeByName("NORMAL");
        const Mesh::VertexAttributeData *uv0Attr = mesh.getVertexAttributeByName("UV0");
        const Mesh::VertexAttributeData *boneIndexAttr = mesh.getVertexAttributeByName("BONEINDEX");
        const Mesh::VertexAttributeData *boneWeightAttr = mesh.getVertexAttributeByName("BONEWEIGHT");

        if (normalAttr) aimesh->mNormals = new aiVector3D[numVertices];
        if (uv0Attr) {
            aimesh->mTextureCoords[0] = new aiVector3D[numVertices];
            aimesh->mNumUVComponents[0] = 2;
        }

        auto readAttr = [&mesh](const Mesh::VertexAttributeData *attr, uint32_t vertexIndex, float *out) {
            const uint8_t *block = static_cast<const uint8_t *>(mesh.getData(attr->getDataIndex()));
            const uint32_t stride = mesh.getStride(attr->getDataIndex());
            const uint8_t *ptr = block + static_cast<size_t>(vertexIndex) * stride + attr->getOffset();
            ReadVertexComponent(ptr, attr->getVertexLayout().dataType, attr->getN(), out);
        };

        for (uint32_t v = 0; v < numVertices; ++v) {
            float comp[4];
            readAttr(posAttr, v, comp);
            aimesh->mVertices[v] = aiVector3D(comp[0], comp[1], comp[2]);

            if (normalAttr) {
                readAttr(normalAttr, v, comp);
                aimesh->mNormals[v] = aiVector3D(comp[0], comp[1], comp[2]);
            }
            if (uv0Attr) {
                readAttr(uv0Attr, v, comp);
                aimesh->mTextureCoords[0][v] = aiVector3D(comp[0], comp[1], 0.0f);
            }
        }

        const Mesh::FaceData &faces = mesh.getFaces();
        const uint8_t *faceData = faces.getData();
        const IndexType indexType = faces.getDataType();
        aimesh->mNumFaces = numFaces;
        aimesh->mFaces = new aiFace[numFaces];
        for (uint32_t f = 0; f < numFaces; ++f) {
            aiFace &face = aimesh->mFaces[f];
            face.mNumIndices = 3;
            face.mIndices = new unsigned int[3];
            for (uint32_t k = 0; k < 3; ++k) {
                face.mIndices[k] = ReadFaceIndex(faceData, indexType, f * 3 + k);
            }
        }

        // Skinning: PVRAssets' bone batches are already merged into a flat,
        // continuous per-vertex bone index list by PODReader (mergeBoneBatches),
        // so BONEINDEX values index directly into Skeleton::bones.
        const int32_t skeletonId = mesh.getSkeletonId();
        if (skeletonId >= 0 && boneIndexAttr && boneWeightAttr) {
            const Skeleton &skeleton = model.getSkeleton(static_cast<uint32_t>(skeletonId));
            // Number of bone influences stored per vertex (width of the
            // BONEWEIGHT/BONEINDEX attributes), NOT the skeleton's bone count.
            const uint32_t numInfluences = boneWeightAttr->getN();

            // Skeleton::bones/invBindMatrices are padded past the real bone
            // count with filler entries (all pointing at node index 0). We
            // don't need to compute the real count explicitly: any padding
            // slot simply receives no vertex weights below and is skipped when
            // emitting aiBones, so iterating the full (padded) array is safe.
            const size_t numBoneSlots = skeleton.bones.size();

            std::vector<std::vector<aiVertexWeight>> perBone(numBoneSlots);
            for (uint32_t v = 0; v < numVertices; ++v) {
                float indices[4];
                float weights[4];
                readAttr(boneIndexAttr, v, indices);
                readAttr(boneWeightAttr, v, weights);
                for (uint32_t j = 0; j < numInfluences && j < 4; ++j) {
                    if (weights[j] <= 0.0f) continue;
                    const auto boneLocal = static_cast<size_t>(std::lround(indices[j]));
                    if (boneLocal < numBoneSlots) {
                        perBone[boneLocal].push_back(aiVertexWeight(v, weights[j]));
                    }
                }
            }

            std::vector<aiBone *> bones;
            bones.reserve(numBoneSlots);
            for (size_t b = 0; b < numBoneSlots; ++b) {
                if (perBone[b].empty()) continue;
                aiBone *bone = new aiBone();
                const aiNode *boneNode = mNodesByIndex[skeleton.bones[b]];
                bone->mName = boneNode->mName;
                aiMatrix4x4 boneWorldInverse = ComputeWorldTransform(boneNode);
                boneWorldInverse.Inverse();
                bone->mOffsetMatrix = boneWorldInverse * meshNodeWorld[m];
                bone->mNumWeights = static_cast<unsigned int>(perBone[b].size());
                bone->mWeights = new aiVertexWeight[perBone[b].size()];
                std::copy(perBone[b].begin(), perBone[b].end(), bone->mWeights);
                bones.push_back(bone);
            }
            if (!bones.empty()) {
                aimesh->mNumBones = static_cast<unsigned int>(bones.size());
                aimesh->mBones = new aiBone *[bones.size()];
                std::copy(bones.begin(), bones.end(), aimesh->mBones);
            }
        }

        aimesh->mMaterialIndex = 0;
        pScene->mMeshes[m] = aimesh;
    }

    // Assign meshes (and their material) to their owning mesh node. Mesh nodes
    // occupy indices [0, getNumMeshNodes()) and objectIndex is the mesh index.
    const uint32_t numMeshNodes = model.getNumMeshNodes();
    for (uint32_t i = 0; i < numMeshNodes; ++i) {
        const Model::Node &node = model.getMeshNode(i);
        const uint32_t meshIndex = node.getObjectId();
        if (meshIndex >= numMeshes) continue;

        aiNode *ainode = mNodesByIndex[i];
        ainode->mNumMeshes = 1;
        ainode->mMeshes = new unsigned int[1];
        ainode->mMeshes[0] = meshIndex;

        const uint32_t materialIndex = node.getMaterialIndex();
        if (materialIndex < numMaterials) {
            pScene->mMeshes[meshIndex]->mMaterialIndex = materialIndex;
        }
    }
}

void PODImporter::BuildAnimations(Model &model, const PODRawInfo &raw, aiScene *pScene) {
    // A POD file holds at most one animation: the scene's frame range. Static
    // files (numFrames == 0, e.g. Miitomo's .Mdl.pod) produce none.
    const uint32_t numFrames = raw.numFrames;
    if (numFrames == 0) {
        return;
    }

    float fps = model.getFPS();
    if (fps <= 0.0f) fps = 30.0f;

    // One channel per node that stores any transform data -- including nodes
    // whose transform is constant. Animation-only POD files (e.g. Miitomo's
    // .Anim.pod) define the pose of every joint, static ones included; if
    // those got no channel, merging the clip onto a separately-loaded model
    // would leave them in the model's rest pose instead of the clip's.
    std::vector<aiNodeAnim *> channels;
    for (size_t i = 0; i < raw.nodes.size(); ++i) {
        const PODRawNode &node = raw.nodes[i];
        if (!HasTransformData(node)) continue;

        const bool isMatrix = !node.matrix.empty();
        // Number of keys to emit for one component: 1 if it isn't animated,
        // else numFrames capped by what the file can actually supply -- one
        // per index entry, or one per stride of key data. The scene frame
        // count is an untrusted 32-bit value; without the cap, a tiny file
        // declaring ~4 billion frames would force multi-gigabyte key arrays
        // per node. Frames past the cap would only repeat the last key,
        // which assimp consumers already hold past the final key.
        const auto keyCount = [&](uint32_t flag, const std::vector<float> &data,
                                      const std::vector<uint32_t> &indices, size_t stride) -> uint32_t {
            if ((node.animFlags & flag) == 0) return 1;
            const size_t available = indices.empty() ? data.size() / stride : indices.size();
            return static_cast<uint32_t>(std::max<size_t>(1, std::min<size_t>(numFrames, available)));
        };
        const uint32_t positionKeys = isMatrix ?
                keyCount(cPodHasMatrixAni, node.matrix, node.matrixIdx, cPodMatrixStride) :
                keyCount(cPodHasPositionAni, node.position, node.positionIdx, cPodPositionStride);
        const uint32_t rotationKeys = isMatrix ?
                positionKeys :
                keyCount(cPodHasRotationAni, node.rotation, node.rotationIdx, cPodRotationStride);
        const uint32_t scalingKeys = isMatrix ?
                positionKeys :
                keyCount(cPodHasScaleAni, node.scale, node.scaleIdx, cPodScaleStride);
        const uint32_t framesToEvaluate = std::max({ positionKeys, rotationKeys, scalingKeys });

        aiNodeAnim *channel = new aiNodeAnim();
        channel->mNodeName = mNodesByIndex[i]->mName;
        channel->mNumPositionKeys = positionKeys;
        channel->mPositionKeys = new aiVectorKey[positionKeys];
        channel->mNumRotationKeys = rotationKeys;
        channel->mRotationKeys = new aiQuatKey[rotationKeys];
        channel->mNumScalingKeys = scalingKeys;
        channel->mScalingKeys = new aiVectorKey[scalingKeys];

        bool warned = false;
        for (uint32_t f = 0; f < framesToEvaluate; ++f) {
            const PODLocalTransform local = EvaluateLocal(node, f, mConvention, warned);
            // Key times are frame numbers; mTicksPerSecond carries the FPS.
            const double time = static_cast<double>(f);
            if (f < positionKeys) channel->mPositionKeys[f] = aiVectorKey(time, local.position);
            if (f < scalingKeys) channel->mScalingKeys[f] = aiVectorKey(time, local.scaling);
            if (f < rotationKeys) {
                aiQuaternion q = local.rotation;
                // Keep consecutive keys in the same hemisphere. Matrix
                // decomposition picks an arbitrary sign per frame, and
                // consumers that interpolate component-wise (rather than
                // shortest-arc slerp) would otherwise spin through 360 deg.
                if (f > 0) {
                    const aiQuaternion &prev = channel->mRotationKeys[f - 1].mValue;
                    if (prev.x * q.x + prev.y * q.y + prev.z * q.z + prev.w * q.w < 0.0f) {
                        q = aiQuaternion(-q.w, -q.x, -q.y, -q.z);
                    }
                }
                channel->mRotationKeys[f] = aiQuatKey(time, q);
            }
        }
        if (warned) {
            ASSIMP_LOG_WARN("POD: node \"", node.name, "\" has animation keys beyond its data; using its first key");
        }
        channels.push_back(channel);
    }

    if (channels.empty()) {
        return;
    }

    aiAnimation *anim = new aiAnimation();
    std::string animName;
    if (!model.getInternalData().animationsData.empty()) {
        animName = model.getInternalData().animationsData[0].getInternalData().animationName;
    }
    anim->mName = aiString(animName.empty() ? std::string("Take") : animName);
    anim->mTicksPerSecond = fps;
    // The last key sits at frame numFrames - 1.
    anim->mDuration = static_cast<double>(numFrames - 1);
    anim->mNumChannels = static_cast<unsigned int>(channels.size());
    anim->mChannels = new aiNodeAnim *[channels.size()];
    std::copy(channels.begin(), channels.end(), anim->mChannels);

    pScene->mNumAnimations = 1;
    pScene->mAnimations = new aiAnimation *[1];
    pScene->mAnimations[0] = anim;
}

void PODImporter::SetupProperties(const Importer *pImp) {
    mConventionSetting = pImp->GetPropertyInteger(AI_CONFIG_IMPORT_POD_QUATERNION_CONVENTION, -1);
}

void PODImporter::InternReadFile(const std::string &pFile, aiScene *pScene, IOSystem *pIOHandler) {
    std::unique_ptr<IOStream> file(pIOHandler->Open(pFile, "rb"));
    if (!file.get()) {
        throw DeadlyImportError("POD: failed to open file ", pFile);
    }

    std::vector<uint8_t> buffer(file->FileSize());
    if (file->Read(buffer.data(), 1, buffer.size()) != buffer.size()) {
        throw DeadlyImportError("POD: failed to read file ", pFile);
    }

    const PODRawInfo raw = ScanRawPod({ buffer.data(), buffer.size() });
    switch (mConventionSetting) {
    case 0: mConvention = PODQuaternionConvention::Sdk; break;
    case 1: mConvention = PODQuaternionConvention::Raw; break;
    default: mConvention = DetectConvention(raw); break;
    }
    ASSIMP_LOG_INFO("POD: history \"", raw.history, "\"; using ",
            mConvention == PODQuaternionConvention::Sdk ? "SDK (conjugate)" : "raw", " quaternion convention");

    pvr::BufferStream stream(pFile, buffer.data(), buffer.size());

    Model model;
    try {
        pvr::assets::readPOD(stream, model);
    } catch (const std::exception &e) {
        throw DeadlyImportError("POD: failed to parse file ", pFile, ": ", e.what());
    }

    BuildNodes(model, raw, pScene);
    BuildMeshesAndMaterials(model, pScene);
    BuildAnimations(model, raw, pScene);
}

} // namespace Assimp

#endif // ASSIMP_BUILD_NO_POD_IMPORTER
