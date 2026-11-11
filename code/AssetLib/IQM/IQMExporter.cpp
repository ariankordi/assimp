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

/** @file IQMExporter.cpp
 *  @brief Inter-Quake Model (IQM v2) exporter.
 *
 *  Format reference and attribution: the IQM SDK by Lee Salzman (MIT license),
 *  https://github.com/lsalzman/iqm -- iqm.txt (format description) and iqm.cpp (the reference
 *  writer). This file is a new implementation against assimp's scene structures. Three rules
 *  are adapted from iqm.cpp so files match what the reference tool writes:
 *    - animation channels are quantized to 16 bits with one offset/scale per joint channel over
 *      every frame of every animation, and a channel whose range is below 1e-10 is stored only
 *      as its offset (iqm.cpp `calcanimdata`);
 *    - blend weights are rounded to bytes and then nudged so they sum to exactly 255
 *      (iqm.cpp `blendcombo::serialize`);
 *    - triangle adjacency pairs triangles over edges with equal positions (and equal blends),
 *      and an edge shared by more than two triangles has no neighbor (iqm.cpp `makeneighbors`).
 *
 *  Conventions written (the same ones assimp's IQM importer and iqm.cpp assume):
 *    - Z-up, unless #AI_CONFIG_EXPORT_IQM_Y_UP is set, which writes assimp's Y-up axes as-is
 *      (raylib, for example, does not rotate IQM files on load);
 *    - texture V runs top to bottom (v' = 1 - v of assimp's bottom-up V);
 *    - triangles are wound clockwise (assimp's counter-clockwise order, reversed).
 *
 *  What is exported:
 *    - one IQM mesh per (node, mesh) reference, in depth-first node order, so instanced meshes
 *      are written once per instance and the order matches node-order loaders;
 *    - joints: every node a mesh bone names, plus the nodes between such a bone and its nearest
 *      bone ancestor. Other nodes (scene roots, armature nodes, mesh nodes) are not added; their
 *      transforms are folded into the root joints and into the vertices;
 *    - vertices in bind pose, model space: skinned vertices are pre-transformed by their bones'
 *      rest transforms times their offset matrices, so files whose inverse bind matrices differ
 *      from the node rest pose still line up;
 *    - animations sampled at a fixed rate per animation (see ChooseFrameRate); an animation with a
 *      single key exports as a single frame;
 *    - per-frame bounds, triangle adjacency, positions, texture coordinates, normals, tangents,
 *      blend indices and weights, vertex colors, and extra UV sets as custom arrays "texcoord<n>".
 *
 *  Not representable in IQM, and skipped with a warning: morph targets, points and lines,
 *  extra vertex color sets, embedded textures (the material name is written instead).
 */

#if !defined(ASSIMP_BUILD_NO_EXPORT) && !defined(ASSIMP_BUILD_NO_IQM_EXPORTER)

#include "IQMExporter.h"
#include "iqm.h"

#include <assimp/DefaultLogger.hpp>
#include <assimp/Exceptional.h>
#include <assimp/Exporter.hpp>
#include <assimp/IOStream.hpp>
#include <assimp/IOSystem.hpp>
#include <assimp/config.h>
#include <assimp/material.h>
#include <assimp/scene.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace Assimp {

namespace {

constexpr unsigned int cMaxInfluences = 4;            ///< Blend indices/weights per vertex (IQM's ubyte4 layout).
constexpr unsigned int cMaxJoints = 256;              ///< Joint indices are stored as unsigned bytes.
constexpr unsigned int cMaxLoaderVertices = 0xFFFF;   ///< raylib (and others) index each mesh with 16-bit indices.
constexpr unsigned int cNoNeighbor = 0xFFFFFFFFu;     ///< Adjacency value for an edge without exactly one neighbor.
constexpr float cChannelRangeEpsilon = 1e-10f;        ///< Smaller channel ranges are constant (iqm.cpp's threshold).
constexpr double cQuantizeSteps = 65535.0;            ///< 16-bit channel values.
constexpr double cDefaultTicksPerSecond = 25.0;       ///< What assimp's own tools assume when a file gives none.
constexpr double cMaxNativeFrameRate = 240.0;         ///< Above this, ticks are a time unit (glTF: ms), not frames.
constexpr double cFrameGridTolerance = 1e-3;          ///< In frames: how close a key must sit to a frame boundary.
constexpr double cFallbackFrameRate = 60.0;           ///< Used when no standard rate hits every key time.
constexpr float cRestEpsilon = 1e-5f;                 ///< Key values this close to the rest pose do not move a node.
constexpr unsigned int cMaxExtraUvSets = AI_MAX_NUMBER_OF_TEXTURECOORDS - 1;

/// Frame rates tried, lowest first, when an animation's ticks are not frames.
constexpr std::array<double, 8> cStandardFrameRates = { 24.0, 25.0, 30.0, 48.0, 50.0, 60.0, 120.0, 240.0 };

/// A decomposed local transform (IQM's joint and pose representation).
struct JointTransform {
    aiVector3D translation;
    aiQuaternion rotation;
    aiVector3D scale = aiVector3D(1.0f, 1.0f, 1.0f);
};

/// One exported joint.
struct Joint {
    const aiNode* node = nullptr;
    uint32_t name = 0;        ///< Offset of the node name in the text section.
    int parent = -1;          ///< Index of the parent joint, or -1 for a root joint.
    bool hasPrefix = false;   ///< Whether @c prefix is not the identity.
    aiMatrix4x4 prefix;       ///< Transforms of the non-joint nodes above this one (and the axis change), applied before the node's own.
    JointTransform base;      ///< Rest pose relative to the parent joint.
};

/// One (node, mesh) reference, in the order the IQM meshes are written.
struct MeshInstance {
    const aiNode* node;
    const aiMesh* mesh;
};

/// Up to four joint influences of one vertex, as written to the file.
struct VertexBlend {
    std::array<uint8_t, cMaxInfluences> joints{};
    std::array<uint8_t, cMaxInfluences> weights{};   ///< Sums to 255 when the vertex has any influence.
};

/// All vertex attributes, concatenated over every mesh (IQM stores one array per attribute).
struct VertexData {
    std::vector<aiVector3D> positions;
    std::vector<aiVector2D> texcoords;
    std::vector<aiVector3D> normals;
    std::vector<aiColor4D> tangents;   ///< xyz = tangent, w = bitangent sign (as r, g, b, a).
    std::vector<VertexBlend> blends;
    std::vector<std::array<uint8_t, 4>> colors;
    std::array<std::vector<aiVector2D>, cMaxExtraUvSets> extraTexcoords;
};

/// The IQM mesh record plus its index data.
struct MeshRecord {
    uint32_t name = 0;
    uint32_t material = 0;
    uint32_t firstVertex = 0;
    uint32_t vertexCount = 0;
    uint32_t firstTriangle = 0;
    uint32_t triangleCount = 0;
};

/// Little-endian byte buffer, independent of the host byte order.
class ByteWriter {
public:
    void U8(uint8_t value) { mBytes.push_back(value); }
    void U16(uint16_t value) {
        U8(static_cast<uint8_t>(value));
        U8(static_cast<uint8_t>(value >> 8));
    }
    void U32(uint32_t value) {
        for (int shift = 0; shift < 32; shift += 8) U8(static_cast<uint8_t>(value >> shift));
    }
    void I32(int32_t value) { U32(static_cast<uint32_t>(value)); }
    void F32(float value) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        U32(bits);
    }
    void Bytes(const void* data, size_t size) {
        const auto* begin = static_cast<const uint8_t*>(data);
        mBytes.insert(mBytes.end(), begin, begin + size);
    }
    /// Pads with zero bytes up to a multiple of @p alignment.
    void Align(size_t alignment) {
        while (mBytes.size() % alignment != 0) U8(0);
    }
    void PatchU32(size_t offset, uint32_t value) {
        for (int i = 0; i < 4; ++i) mBytes[offset + i] = static_cast<uint8_t>(value >> (8 * i));
    }
    uint32_t Size() const { return static_cast<uint32_t>(mBytes.size()); }
    const std::vector<uint8_t>& Data() const { return mBytes; }

private:
    std::vector<uint8_t> mBytes;
};

/// The IQM text section: null-terminated strings, deduplicated, with "" at offset 0 so an
/// offset of 0 always reads as an empty name.
class StringTable {
public:
    StringTable() { Add(""); }

    uint32_t Add(const std::string& text) {
        const auto found = mOffsets.find(text);
        if (found != mOffsets.end()) return found->second;
        const auto offset = static_cast<uint32_t>(mData.size());
        mData.insert(mData.end(), text.begin(), text.end());
        mData.push_back('\0');
        mOffsets.emplace(text, offset);
        return offset;
    }

    const std::vector<char>& Data() const { return mData; }

private:
    std::vector<char> mData;
    std::unordered_map<std::string, uint32_t> mOffsets;
};

/// Rotation applied to everything when writing IQM's Z-up axes: +90 degrees about X,
/// the inverse of the importer's conversion (y' = -z, z' = y).
aiMatrix4x4 AxisConversion(bool yUp) {
    if (yUp) return aiMatrix4x4();
    return aiMatrix4x4(
            1.f, 0.f, 0.f, 0.f,
            0.f, 0.f, -1.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 1.f);
}

/// Global transform of @p node (the product of it and all its ancestors).
aiMatrix4x4 GlobalTransform(const aiNode* node) {
    aiMatrix4x4 global = node->mTransformation;
    for (const aiNode* parent = node->mParent; parent != nullptr; parent = parent->mParent) {
        global = parent->mTransformation * global;
    }
    return global;
}

aiMatrix4x4 ComposeTransform(const JointTransform& transform) {
    return aiMatrix4x4(transform.scale, transform.rotation, transform.translation);
}

JointTransform DecomposeTransform(const aiMatrix4x4& matrix) {
    JointTransform transform;
    matrix.Decompose(transform.scale, transform.rotation, transform.translation);
    transform.rotation.Normalize();
    return transform;
}

/// Calls @p visit for every node in depth-first, parent-before-child, child-index order.
template <class Visitor>
void VisitDepthFirst(const aiNode* root, Visitor&& visit) {
    std::vector<const aiNode*> stack(1, root);
    while (!stack.empty()) {
        const aiNode* node = stack.back();
        stack.pop_back();
        visit(node);
        for (unsigned int i = node->mNumChildren; i-- > 0;) stack.push_back(node->mChildren[i]);
    }
}

std::vector<MeshInstance> CollectMeshInstances(const aiScene* scene) {
    std::vector<MeshInstance> instances;
    VisitDepthFirst(scene->mRootNode, [&](const aiNode* node) {
        for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
            instances.push_back({ node, scene->mMeshes[node->mMeshes[i]] });
        }
    });
    return instances;
}

/// True if any key of @p channel moves @p node away from its rest transform. Exporters that bake
/// every node at every frame give unanimated nodes constant channels; those are not motion.
bool ChannelMovesNode(const aiNodeAnim* channel, const aiNode* node) {
    aiVector3D restScale, restTranslation;
    aiQuaternion restRotation;
    node->mTransformation.Decompose(restScale, restRotation, restTranslation);
    auto differs = [](const aiVector3D& a, const aiVector3D& b) {
        return std::fabs(a.x - b.x) > cRestEpsilon || std::fabs(a.y - b.y) > cRestEpsilon || std::fabs(a.z - b.z) > cRestEpsilon;
    };
    for (unsigned int k = 0; k < channel->mNumPositionKeys; ++k) {
        if (differs(channel->mPositionKeys[k].mValue, restTranslation)) return true;
    }
    for (unsigned int k = 0; k < channel->mNumScalingKeys; ++k) {
        if (differs(channel->mScalingKeys[k].mValue, restScale)) return true;
    }
    for (unsigned int k = 0; k < channel->mNumRotationKeys; ++k) {
        const aiQuaternion& q = channel->mRotationKeys[k].mValue;
        // q and -q are the same rotation.
        const float dot = q.x * restRotation.x + q.y * restRotation.y + q.z * restRotation.z + q.w * restRotation.w;
        if (1.0f - std::fabs(dot) > cRestEpsilon) return true;
    }
    return false;
}

/// Picks the joints, in depth-first order so every parent precedes its children:
///   - every node a mesh bone names;
///   - the nodes between such a bone and its nearest bone ancestor, so a chain is never broken;
///   - ancestors of those that some animation moves away from their rest pose (and the nodes
///     between), because such a node moves the skeleton and IQM can only animate joints.
///     Other ancestors are folded into the root joints instead.
std::vector<Joint> CollectJoints(const aiScene* scene, const std::vector<MeshInstance>& instances,
        const aiMatrix4x4& axis) {
    std::map<std::string, bool> boneNames;
    for (const MeshInstance& instance : instances) {
        for (unsigned int b = 0; b < instance.mesh->mNumBones; ++b) {
            boneNames[instance.mesh->mBones[b]->mName.C_Str()] = true;
        }
    }

    std::map<std::string, bool> animatedNames;
    for (unsigned int a = 0; a < scene->mNumAnimations; ++a) {
        const aiAnimation* animation = scene->mAnimations[a];
        for (unsigned int c = 0; c < animation->mNumChannels; ++c) {
            const aiNodeAnim* channel = animation->mChannels[c];
            const aiNode* node = scene->mRootNode->FindNode(channel->mNodeName);
            if (node != nullptr && ChannelMovesNode(channel, node)) animatedNames[channel->mNodeName.C_Str()] = true;
        }
    }

    std::map<const aiNode*, bool> isJoint;
    VisitDepthFirst(scene->mRootNode, [&](const aiNode* node) {
        if (boneNames.count(node->mName.C_Str()) == 0) return;
        isJoint[node] = true;
        // Walk up the ancestors. The topmost one that is a bone or animated becomes a joint,
        // and so does every node between it and this bone.
        std::vector<const aiNode*> path;
        size_t keep = 0; // How many entries of path (from the bottom) become joints.
        for (const aiNode* ancestor = node->mParent; ancestor != nullptr; ancestor = ancestor->mParent) {
            path.push_back(ancestor);
            const std::string name = ancestor->mName.C_Str();
            if (boneNames.count(name) != 0 || animatedNames.count(name) != 0) keep = path.size();
        }
        for (size_t i = 0; i < keep; ++i) isJoint[path[i]] = true;
    });

    std::vector<Joint> joints;
    std::map<const aiNode*, int> jointIndex;
    VisitDepthFirst(scene->mRootNode, [&](const aiNode* node) {
        if (isJoint.count(node) == 0) return;
        Joint joint;
        joint.node = node;
        // Multiply up the non-joint ancestors until the parent joint (or the scene root).
        aiMatrix4x4 prefix;
        const aiNode* ancestor = node->mParent;
        for (; ancestor != nullptr && isJoint.count(ancestor) == 0; ancestor = ancestor->mParent) {
            prefix = ancestor->mTransformation * prefix;
        }
        if (ancestor != nullptr) {
            joint.parent = jointIndex.at(ancestor);
        } else {
            prefix = axis * prefix;
        }
        joint.hasPrefix = !prefix.IsIdentity();
        joint.prefix = prefix;
        joint.base = DecomposeTransform(joint.hasPrefix ? prefix * node->mTransformation : node->mTransformation);
        jointIndex[node] = static_cast<int>(joints.size());
        joints.push_back(joint);
    });

    if (joints.size() > cMaxJoints) {
        throw DeadlyExportError("IQM: ", joints.size(), " joints; IQM's byte blend indices allow at most ", cMaxJoints, ".");
    }
    return joints;
}

/// Index of the joint for @p node or its nearest ancestor that is a joint, or -1.
int NearestJoint(const std::vector<Joint>& joints, const aiNode* node) {
    for (; node != nullptr; node = node->mParent) {
        for (size_t j = 0; j < joints.size(); ++j) {
            if (joints[j].node == node) return static_cast<int>(j);
        }
    }
    return -1;
}

/// Global rest transforms of the joints, in the exported (axis-converted) space.
std::vector<aiMatrix4x4> JointGlobals(const std::vector<Joint>& joints, const std::vector<JointTransform>& locals) {
    std::vector<aiMatrix4x4> globals(joints.size());
    for (size_t j = 0; j < joints.size(); ++j) {
        const aiMatrix4x4 local = ComposeTransform(locals[j]);
        globals[j] = joints[j].parent >= 0 ? globals[joints[j].parent] * local : local;
    }
    return globals;
}

/// Rounds normalized weights to bytes summing to 255; adapted from iqm.cpp blendcombo::serialize.
std::array<uint8_t, cMaxInfluences> QuantizeWeights(const std::array<float, cMaxInfluences>& weights, unsigned int count) {
    std::array<uint8_t, cMaxInfluences> bytes{};
    int total = 0;
    for (unsigned int k = 0; k < cMaxInfluences; ++k) {
        bytes[k] = static_cast<uint8_t>(0.5f + weights[k] * 255.0f);
        total += bytes[k];
    }
    if (count == 0) return bytes;
    while (total > 255) {
        for (unsigned int k = 0; k < cMaxInfluences && total > 255; ++k) {
            if (bytes[k] > 0) { --bytes[k]; --total; }
        }
    }
    while (total < 255) {
        for (unsigned int k = 0; k < cMaxInfluences && total < 255; ++k) {
            if (bytes[k] < 255) { ++bytes[k]; ++total; }
        }
    }
    return bytes;
}

/// Weighted bone list of one vertex, before limiting.
using InfluenceList = std::vector<std::pair<float, int>>;

/// Builds each vertex's influences (weight, joint index) from the mesh's bones.
std::vector<InfluenceList> GatherInfluences(const aiMesh* mesh, const std::map<std::string, int>& jointByName) {
    std::vector<InfluenceList> influences(mesh->mNumVertices);
    for (unsigned int b = 0; b < mesh->mNumBones; ++b) {
        const aiBone* bone = mesh->mBones[b];
        const int joint = jointByName.at(bone->mName.C_Str());
        for (unsigned int w = 0; w < bone->mNumWeights; ++w) {
            const aiVertexWeight& weight = bone->mWeights[w];
            if (weight.mVertexId < mesh->mNumVertices && weight.mWeight > 0.0f) {
                influences[weight.mVertexId].emplace_back(weight.mWeight, joint);
            }
        }
    }
    return influences;
}

/// Keeps the four largest influences, normalized; returns how many there are.
unsigned int LimitInfluences(InfluenceList influences, std::array<int, cMaxInfluences>& joints,
        std::array<float, cMaxInfluences>& weights) {
    std::stable_sort(influences.begin(), influences.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });
    const auto count = static_cast<unsigned int>(std::min<size_t>(influences.size(), cMaxInfluences));
    float total = 0.0f;
    for (unsigned int k = 0; k < count; ++k) total += influences[k].first;
    joints.fill(0);
    weights.fill(0.0f);
    for (unsigned int k = 0; k < count; ++k) {
        joints[k] = influences[k].second;
        weights[k] = influences[k].first / total;
    }
    return count;
}

aiVector3D TransformDirection(const aiMatrix3x3& matrix, const aiVector3D& direction) {
    aiVector3D result = matrix * direction;
    const float length = result.Length();
    return length > 0.0f ? result / length : result;
}

/// Everything the exporter needs from the scene, gathered once.
class IQMWriter {
public:
    IQMWriter(const aiScene* scene, const ExportProperties* properties) :
            mScene(scene),
            mYUp(properties != nullptr && properties->GetPropertyBool(AI_CONFIG_EXPORT_IQM_Y_UP, false)),
            mConfiguredFrameRate(properties != nullptr ? properties->GetPropertyFloat(AI_CONFIG_EXPORT_IQM_FRAMERATE, 0.0f) : 0.0f),
            mAxis(AxisConversion(mYUp)) {}

    std::vector<uint8_t> Write() {
        mInstances = CollectMeshInstances(mScene);
        mJoints = CollectJoints(mScene, mInstances, mAxis);
        for (size_t j = 0; j < mJoints.size(); ++j) {
            mJointByName[mJoints[j].node->mName.C_Str()] = static_cast<int>(j);
            // Every string must be in the table before Serialize writes the text section.
            mJoints[j].name = mStrings.Add(mJoints[j].node->mName.C_Str());
        }
        BuildMeshes();
        for (unsigned int set = 1; set < AI_MAX_NUMBER_OF_TEXTURECOORDS; ++set) {
            if (mHasExtraTexcoords[set - 1]) mExtraTexcoordNames[set - 1] = mStrings.Add("texcoord" + std::to_string(set));
        }
        BuildAdjacency();
        BuildAnimations();
        return Serialize();
    }

private:
    /// Decides which vertex arrays the file has: an array is written if any mesh has it.
    void ChooseArrays() {
        for (const MeshInstance& instance : mInstances) {
            const aiMesh* mesh = instance.mesh;
            mHasTexcoords |= mesh->HasTextureCoords(0);
            mHasNormals |= mesh->HasNormals();
            mHasTangents |= mesh->HasNormals() && mesh->HasTangentsAndBitangents();
            mHasColors |= mesh->HasVertexColors(0);
            for (unsigned int set = 1; set < AI_MAX_NUMBER_OF_TEXTURECOORDS; ++set) {
                mHasExtraTexcoords[set - 1] = mHasExtraTexcoords[set - 1] || mesh->HasTextureCoords(set);
            }
            if (mesh->HasVertexColors(1)) mDroppedColorSets = true;
            if (mesh->mNumAnimMeshes > 0) mDroppedMorphTargets = true;
        }
    }

    /// The material string: the diffuse (or base color) texture path when there is a file to
    /// point at, as the IQM tools do, else the material's name.
    std::string MaterialString(const aiMesh* mesh) {
        if (mesh->mMaterialIndex >= mScene->mNumMaterials) return std::string();
        const aiMaterial* material = mScene->mMaterials[mesh->mMaterialIndex];
        aiString name;
        material->Get(AI_MATKEY_NAME, name);
        for (aiTextureType type : { aiTextureType_DIFFUSE, aiTextureType_BASE_COLOR }) {
            aiString path;
            if (material->GetTextureCount(type) == 0 || material->GetTexture(type, 0, &path) != AI_SUCCESS) continue;
            if (path.length > 0 && path.data[0] == '*') {
                ASSIMP_LOG_WARN("IQM: material '", name.C_Str(), "' uses an embedded texture, which IQM cannot reference; writing the material name instead.");
                break;
            }
            return path.C_Str();
        }
        return name.C_Str();
    }

    void BuildMeshes() {
        ChooseArrays();
        for (const MeshInstance& instance : mInstances) {
            AppendMesh(instance);
        }
        if (mDroppedColorSets) ASSIMP_LOG_WARN("IQM: only the first vertex color set is exported.");
        if (mDroppedMorphTargets) ASSIMP_LOG_WARN("IQM: morph targets are not representable in IQM and were not exported.");
    }

    void AppendMesh(const MeshInstance& instance) {
        const aiMesh* mesh = instance.mesh;
        MeshRecord record;
        record.name = mStrings.Add(mesh->mName.length > 0 ? mesh->mName.C_Str() : instance.node->mName.C_Str());
        record.material = mStrings.Add(MaterialString(mesh));
        record.firstVertex = static_cast<uint32_t>(mVertices.positions.size());
        record.vertexCount = mesh->mNumVertices;
        record.firstTriangle = static_cast<uint32_t>(mTriangles.size());

        if (mesh->mNumVertices > cMaxLoaderVertices) {
            ASSIMP_LOG_WARN("IQM: mesh '", mesh->mName.C_Str(), "' has ", mesh->mNumVertices,
                    " vertices; loaders with 16-bit indices (raylib) cannot draw more than ", cMaxLoaderVertices, ".");
        }

        AppendVertices(instance);

        unsigned int skippedFaces = 0;
        for (unsigned int f = 0; f < mesh->mNumFaces; ++f) {
            const aiFace& face = mesh->mFaces[f];
            if (face.mNumIndices != 3) {
                ++skippedFaces;
                continue;
            }
            // Reverse the winding: IQM triangles are clockwise.
            mTriangles.push_back({ record.firstVertex + face.mIndices[0],
                    record.firstVertex + face.mIndices[2],
                    record.firstVertex + face.mIndices[1] });
        }
        if (skippedFaces > 0) {
            ASSIMP_LOG_WARN("IQM: mesh '", mesh->mName.C_Str(), "': skipped ", skippedFaces, " point/line primitive(s).");
        }
        record.triangleCount = static_cast<uint32_t>(mTriangles.size()) - record.firstTriangle;
        mMeshes.push_back(record);
    }

    /// Converts one mesh's vertices to bind-pose model space and appends every attribute.
    void AppendVertices(const MeshInstance& instance) {
        const aiMesh* mesh = instance.mesh;
        const bool skinned = mesh->mNumBones > 0;
        const aiMatrix4x4 nodeMatrix = mAxis * GlobalTransform(instance.node);
        // Vertices without weights (or meshes without bones) follow their node's nearest joint.
        const int rigidJoint = mJoints.empty() ? -1 : NearestJoint(mJoints, instance.node);

        // Per-bone skinning matrices for the pre-transform: axis * rest global * offset.
        std::map<int, aiMatrix4x4> boneMatrices;
        for (unsigned int b = 0; b < mesh->mNumBones; ++b) {
            const aiBone* bone = mesh->mBones[b];
            const int joint = mJointByName.at(bone->mName.C_Str());
            boneMatrices[joint] = mAxis * GlobalTransform(mJoints[joint].node) * bone->mOffsetMatrix;
        }
        const std::vector<InfluenceList> influences =
                skinned ? GatherInfluences(mesh, mJointByName) : std::vector<InfluenceList>(mesh->mNumVertices);

        unsigned int unweighted = 0;
        for (unsigned int v = 0; v < mesh->mNumVertices; ++v) {
            std::array<int, cMaxInfluences> joints{};
            std::array<float, cMaxInfluences> weights{};
            unsigned int count = LimitInfluences(influences[v], joints, weights);

            aiMatrix4x4 vertexMatrix = nodeMatrix;
            if (count > 0) {
                // Blend the bind matrices; equal to nodeMatrix when the offsets match the rest pose.
                vertexMatrix = aiMatrix4x4(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
                for (unsigned int k = 0; k < count; ++k) {
                    const aiMatrix4x4& bone = boneMatrices.at(joints[k]);
                    for (unsigned int r = 0; r < 4; ++r) {
                        for (unsigned int c = 0; c < 4; ++c) vertexMatrix[r][c] += bone[r][c] * weights[k];
                    }
                }
            } else if (rigidJoint >= 0) {
                if (skinned) ++unweighted;
                joints[0] = rigidJoint;
                weights[0] = 1.0f;
                count = 1;
            } else if (!mJoints.empty()) {
                ++unweighted;
            }

            const aiMatrix3x3 linear(vertexMatrix);
            const aiMatrix3x3 normalMatrix = aiMatrix3x3(linear).Inverse().Transpose();
            mVertices.positions.push_back(vertexMatrix * mesh->mVertices[v]);
            if (mHasNormals) {
                mVertices.normals.push_back(mesh->HasNormals() ? TransformDirection(normalMatrix, mesh->mNormals[v]) : aiVector3D());
            }
            if (mHasTexcoords) {
                mVertices.texcoords.push_back(mesh->HasTextureCoords(0)
                        ? aiVector2D(mesh->mTextureCoords[0][v].x, 1.0f - mesh->mTextureCoords[0][v].y)
                        : aiVector2D());
            }
            if (mHasTangents) {
                aiColor4D tangent(0, 0, 0, 1);
                if (mesh->HasNormals() && mesh->HasTangentsAndBitangents()) {
                    const aiVector3D t = TransformDirection(linear, mesh->mTangents[v]);
                    const aiVector3D b = TransformDirection(linear, mesh->mBitangents[v]);
                    const aiVector3D n = TransformDirection(normalMatrix, mesh->mNormals[v]);
                    // Same rule as assimp's glTF2 exporter (glTF and IQM share the top-down V):
                    // the glTF2 importer stores bitangents as glTF defines them, (n x t) * w,
                    // even though it flips V, so the sign carries over without a flip.
                    const float handedness = ((n ^ t) * b) < 0.0f ? -1.0f : 1.0f;
                    tangent = aiColor4D(t.x, t.y, t.z, handedness);
                }
                mVertices.tangents.push_back(tangent);
            }
            if (!mJoints.empty()) {
                VertexBlend blend;
                blend.weights = QuantizeWeights(weights, count);
                for (unsigned int k = 0; k < cMaxInfluences; ++k) blend.joints[k] = static_cast<uint8_t>(joints[k]);
                mVertices.blends.push_back(blend);
            }
            if (mHasColors) {
                std::array<uint8_t, 4> color = { 255, 255, 255, 255 };
                if (mesh->HasVertexColors(0)) {
                    const aiColor4D& source = mesh->mColors[0][v];
                    const float channels[4] = { source.r, source.g, source.b, source.a };
                    for (int c = 0; c < 4; ++c) {
                        color[c] = static_cast<uint8_t>(0.5f + std::clamp(channels[c], 0.0f, 1.0f) * 255.0f);
                    }
                }
                mVertices.colors.push_back(color);
            }
            for (unsigned int set = 1; set < AI_MAX_NUMBER_OF_TEXTURECOORDS; ++set) {
                if (!mHasExtraTexcoords[set - 1]) continue;
                mVertices.extraTexcoords[set - 1].push_back(mesh->HasTextureCoords(set)
                        ? aiVector2D(mesh->mTextureCoords[set][v].x, 1.0f - mesh->mTextureCoords[set][v].y)
                        : aiVector2D());
            }
        }
        if (unweighted > 0) {
            ASSIMP_LOG_WARN("IQM: mesh '", mesh->mName.C_Str(), "': ", unweighted, " vertices have no bone weights",
                    rigidJoint >= 0 ? "; bound to the mesh node's nearest joint." : " and no joint above the mesh node; they will not follow the skeleton.");
        }
    }

    /// Triangle adjacency; the rules follow iqm.cpp makeneighbors (see the file comment).
    void BuildAdjacency() {
        // Vertices that share a position (and blend) count as the same corner.
        using CornerKey = std::tuple<float, float, float, uint32_t, uint32_t>;
        std::map<CornerKey, uint32_t> cornerIds;
        std::vector<uint32_t> cornerOf(mVertices.positions.size());
        for (size_t v = 0; v < mVertices.positions.size(); ++v) {
            const aiVector3D& p = mVertices.positions[v];
            uint32_t blendJoints = 0, blendWeights = 0;
            if (!mVertices.blends.empty()) {
                std::memcpy(&blendJoints, mVertices.blends[v].joints.data(), 4);
                std::memcpy(&blendWeights, mVertices.blends[v].weights.data(), 4);
            }
            const CornerKey key(p.x, p.y, p.z, blendJoints, blendWeights);
            cornerOf[v] = cornerIds.emplace(key, static_cast<uint32_t>(cornerIds.size())).first->second;
        }

        // Each undirected edge maps to the first two triangles using it; a third marks it shared.
        struct EdgeUse {
            uint32_t first = cNoNeighbor;
            uint32_t second = cNoNeighbor;
            bool overused = false;
        };
        auto edgeKey = [&](uint32_t a, uint32_t b) {
            const uint32_t ca = cornerOf[a], cb = cornerOf[b];
            return std::make_pair(std::min(ca, cb), std::max(ca, cb));
        };
        std::map<std::pair<uint32_t, uint32_t>, EdgeUse> edges;
        for (uint32_t t = 0; t < mTriangles.size(); ++t) {
            for (int j = 0, p = 2; j < 3; p = j, ++j) {
                EdgeUse& use = edges[edgeKey(mTriangles[t][p], mTriangles[t][j])];
                if (use.first == cNoNeighbor) use.first = t;
                else if (use.second == cNoNeighbor) use.second = t;
                else use.overused = true;
            }
        }
        mAdjacency.resize(mTriangles.size());
        for (uint32_t t = 0; t < mTriangles.size(); ++t) {
            for (int j = 0, p = 2; j < 3; p = j, ++j) {
                const EdgeUse& use = edges.at(edgeKey(mTriangles[t][p], mTriangles[t][j]));
                mAdjacency[t][p] = use.overused ? cNoNeighbor : (use.first == t ? use.second : use.first);
            }
        }
    }

    /// Chooses the sampling rate of @p animation, in frames per second.
    double ChooseFrameRate(const aiAnimation* animation, double ticksPerSecond) const {
        if (mConfiguredFrameRate > 0.0f) return mConfiguredFrameRate;
        // True if every key time is a whole number of frames at @p ticksPerFrame.
        auto keysOnGrid = [animation](double ticksPerFrame) {
            auto onGrid = [ticksPerFrame](double time) {
                const double frames = time / ticksPerFrame;
                return std::fabs(frames - std::round(frames)) < cFrameGridTolerance;
            };
            for (unsigned int c = 0; c < animation->mNumChannels; ++c) {
                const aiNodeAnim* channel = animation->mChannels[c];
                for (unsigned int k = 0; k < channel->mNumPositionKeys; ++k) if (!onGrid(channel->mPositionKeys[k].mTime)) return false;
                for (unsigned int k = 0; k < channel->mNumRotationKeys; ++k) if (!onGrid(channel->mRotationKeys[k].mTime)) return false;
                for (unsigned int k = 0; k < channel->mNumScalingKeys; ++k) if (!onGrid(channel->mScalingKeys[k].mTime)) return false;
            }
            return true;
        };
        // Formats that store frame numbers as ticks: one frame per tick loses nothing.
        if (ticksPerSecond <= cMaxNativeFrameRate && keysOnGrid(1.0)) return ticksPerSecond;
        for (double rate : cStandardFrameRates) {
            if (keysOnGrid(ticksPerSecond / rate)) return rate;
        }
        ASSIMP_LOG_WARN("IQM: animation '", animation->mName.C_Str(), "' has keys off every standard frame rate; resampling at ",
                cFallbackFrameRate, " fps.");
        return cFallbackFrameRate;
    }

    /// Samples one joint's local transform at @p time (ticks) from @p channel, or its rest pose.
    JointTransform SampleChannel(const aiNodeAnim* channel, const aiNode* node, double time) const {
        JointTransform rest = DecomposeTransform(node->mTransformation);
        if (channel == nullptr) return rest;
        JointTransform sample;
        sample.translation = SampleKeys(channel->mPositionKeys, channel->mNumPositionKeys, time, rest.translation,
                [](const aiVector3D& a, const aiVector3D& b, float f) { return a + (b - a) * f; });
        sample.rotation = SampleKeys(channel->mRotationKeys, channel->mNumRotationKeys, time, rest.rotation,
                [](const aiQuaternion& a, const aiQuaternion& b, float f) {
                    aiQuaternion out;
                    aiQuaternion::Interpolate(out, a, b, f);
                    return out.Normalize();
                });
        sample.scale = SampleKeys(channel->mScalingKeys, channel->mNumScalingKeys, time, rest.scale,
                [](const aiVector3D& a, const aiVector3D& b, float f) { return a + (b - a) * f; });
        return sample;
    }

    /// Value of a key track at @p time; holds the first/last key outside the keyed range.
    template <class Key, class Value, class Lerp>
    static Value SampleKeys(const Key* keys, unsigned int count, double time, const Value& fallback, Lerp lerp) {
        if (count == 0) return fallback;
        if (count == 1 || time <= keys[0].mTime) return keys[0].mValue;
        if (time >= keys[count - 1].mTime) return keys[count - 1].mValue;
        const Key* next = std::upper_bound(keys, keys + count, time, [](double t, const Key& key) { return t < key.mTime; });
        const Key* previous = next - 1;
        if (previous->mInterpolation == aiAnimInterpolation_Step) return previous->mValue;
        const double span = next->mTime - previous->mTime;
        const float factor = span > 0.0 ? static_cast<float>((time - previous->mTime) / span) : 0.0f;
        return lerp(previous->mValue, next->mValue, factor);
    }

    void BuildAnimations() {
        if (mJoints.empty() || mScene->mNumAnimations == 0) {
            if (mScene->mNumAnimations > 0) ASSIMP_LOG_WARN("IQM: the scene has animations but no skinned joints; animations not exported.");
            return;
        }
        std::vector<JointTransform> baseLocals;
        for (const Joint& joint : mJoints) baseLocals.push_back(joint.base);
        const std::vector<aiMatrix4x4> baseGlobals = JointGlobals(mJoints, baseLocals);
        std::vector<aiMatrix4x4> inverseBase;
        for (const aiMatrix4x4& global : baseGlobals) inverseBase.push_back(aiMatrix4x4(global).Inverse());

        for (unsigned int a = 0; a < mScene->mNumAnimations; ++a) {
            const aiAnimation* animation = mScene->mAnimations[a];
            AppendAnimation(animation, inverseBase);
        }
        QuantizeChannels();
    }

    void AppendAnimation(const aiAnimation* animation, const std::vector<aiMatrix4x4>& inverseBase) {
        std::vector<const aiNodeAnim*> channels(mJoints.size(), nullptr);
        // Channels on non-joint nodes are dropped. Only ones that move their node lose motion
        // (nodes outside the skeleton, e.g. a camera or a rigid prop); constant ones are noise.
        unsigned int lostChannels = 0;
        for (unsigned int c = 0; c < animation->mNumChannels; ++c) {
            const aiNodeAnim* channel = animation->mChannels[c];
            const auto found = mJointByName.find(channel->mNodeName.C_Str());
            if (found != mJointByName.end()) {
                channels[found->second] = channel;
                continue;
            }
            const aiNode* node = mScene->mRootNode->FindNode(channel->mNodeName);
            if (node != nullptr && ChannelMovesNode(channel, node)) ++lostChannels;
        }
        if (lostChannels > 0) {
            ASSIMP_LOG_WARN("IQM: animation '", animation->mName.C_Str(), "': ", lostChannels,
                    " channel(s) move nodes outside the skeleton and were not exported.");
        }
        if (animation->mNumMorphMeshChannels > 0 || animation->mNumMeshChannels > 0) {
            ASSIMP_LOG_WARN("IQM: animation '", animation->mName.C_Str(), "': mesh/morph channels are not representable and were not exported.");
        }

        const double ticksPerSecond = animation->mTicksPerSecond > 0.0 ? animation->mTicksPerSecond : cDefaultTicksPerSecond;
        const double frameRate = ChooseFrameRate(animation, ticksPerSecond);
        const double durationFrames = animation->mDuration / ticksPerSecond * frameRate;
        const auto frameCount = static_cast<uint32_t>(std::llround(std::max(durationFrames, 0.0))) + 1;

        iqmanim record{};
        record.name = mStrings.Add(animation->mName.C_Str());
        record.first_frame = mFrameCount;
        record.num_frames = frameCount;
        record.framerate = static_cast<float>(frameRate);
        record.flags = 0;
        mAnimations.push_back(record);

        std::vector<JointTransform> previous(mJoints.size());
        for (size_t j = 0; j < mJoints.size(); ++j) previous[j] = mJoints[j].base;
        std::vector<JointTransform> frame(mJoints.size());
        for (uint32_t f = 0; f < frameCount; ++f) {
            const double time = f * ticksPerSecond / frameRate;
            for (size_t j = 0; j < mJoints.size(); ++j) {
                const Joint& joint = mJoints[j];
                JointTransform local = SampleChannel(channels[j], joint.node, time);
                if (joint.hasPrefix) local = DecomposeTransform(joint.prefix * ComposeTransform(local));
                // Keep each joint's quaternions in one hemisphere, so interpolating readers
                // never take the long way round and the channel ranges stay small.
                const aiQuaternion& last = previous[j].rotation;
                const float dot = last.x * local.rotation.x + last.y * local.rotation.y + last.z * local.rotation.z + last.w * local.rotation.w;
                if (dot < 0.0f) local.rotation = aiQuaternion(-local.rotation.w, -local.rotation.x, -local.rotation.y, -local.rotation.z);
                frame[j] = local;
            }
            previous = frame;
            mFrames.insert(mFrames.end(), frame.begin(), frame.end());
            mBounds.push_back(FrameBounds(frame, inverseBase));
        }
        mFrameCount += frameCount;
    }

    /// Bounds of the skinned vertices in one frame (iqm.cpp makebounds).
    iqmbounds FrameBounds(const std::vector<JointTransform>& frame, const std::vector<aiMatrix4x4>& inverseBase) const {
        const std::vector<aiMatrix4x4> globals = JointGlobals(mJoints, frame);
        std::vector<aiMatrix4x4> skin(mJoints.size());
        for (size_t j = 0; j < mJoints.size(); ++j) skin[j] = globals[j] * inverseBase[j];

        iqmbounds bounds{};
        double xyRadius = 0.0, radius = 0.0;
        for (size_t v = 0; v < mVertices.positions.size(); ++v) {
            const VertexBlend& blend = mVertices.blends[v];
            aiVector3D position;
            for (unsigned int k = 0; k < cMaxInfluences; ++k) {
                if (blend.weights[k] == 0) continue;
                position += (skin[blend.joints[k]] * mVertices.positions[v]) * (blend.weights[k] / 255.0f);
            }
            const float p[3] = { position.x, position.y, position.z };
            for (int axis = 0; axis < 3; ++axis) {
                bounds.bbmin[axis] = v == 0 ? p[axis] : std::min(bounds.bbmin[axis], p[axis]);
                bounds.bbmax[axis] = v == 0 ? p[axis] : std::max(bounds.bbmax[axis], p[axis]);
            }
            const double xy = double(p[0]) * p[0] + double(p[1]) * p[1];
            xyRadius = std::max(xyRadius, xy);
            radius = std::max(radius, xy + double(p[2]) * p[2]);
        }
        bounds.xyradius = static_cast<float>(std::sqrt(xyRadius));
        bounds.radius = static_cast<float>(std::sqrt(radius));
        return bounds;
    }

    /// One offset/scale per joint channel over every frame (iqm.cpp calcanimdata).
    void QuantizeChannels() {
        const size_t jointCount = mJoints.size();
        mPoses.assign(jointCount, iqmpose{});
        std::vector<std::array<float, 10>> low(jointCount), high(jointCount);
        for (size_t j = 0; j < jointCount; ++j) {
            low[j].fill(1e16f);
            high[j].fill(-1e16f);
        }
        for (size_t i = 0; i < mFrames.size(); ++i) {
            const size_t j = i % jointCount;
            const std::array<float, 10> values = ChannelValues(mFrames[i]);
            for (int k = 0; k < 10; ++k) {
                low[j][k] = std::min(low[j][k], values[k]);
                high[j][k] = std::max(high[j][k], values[k]);
            }
        }
        mFrameChannels = 0;
        for (size_t j = 0; j < jointCount; ++j) {
            iqmpose& pose = mPoses[j];
            pose.parent = mJoints[j].parent;
            pose.mask = 0;
            for (int k = 0; k < 10; ++k) {
                const float range = high[j][k] - low[j][k];
                pose.channeloffset[k] = low[j][k];
                if (range >= cChannelRangeEpsilon) {
                    pose.channelscale[k] = static_cast<float>(range / cQuantizeSteps);
                    pose.mask |= 1u << k;
                    ++mFrameChannels;
                } else {
                    pose.channelscale[k] = 0.0f;
                }
            }
        }
        for (size_t i = 0; i < mFrames.size(); ++i) {
            const iqmpose& pose = mPoses[i % jointCount];
            const std::array<float, 10> values = ChannelValues(mFrames[i]);
            for (int k = 0; k < 10; ++k) {
                if ((pose.mask & (1u << k)) == 0) continue;
                const double steps = (values[k] - pose.channeloffset[k]) / pose.channelscale[k];
                mFrameData.push_back(static_cast<uint16_t>(std::clamp(0.5 + steps, 0.0, cQuantizeSteps)));
            }
        }
    }

    /// The ten IQM channels: translation xyz, rotation xyzw, scale xyz.
    static std::array<float, 10> ChannelValues(const JointTransform& t) {
        return { t.translation.x, t.translation.y, t.translation.z,
            t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w,
            t.scale.x, t.scale.y, t.scale.z };
    }

    /// Lays the sections out in the reference writer's order and returns the file bytes.
    std::vector<uint8_t> Serialize() {
        // Vertex arrays: in the order iqm.txt requires (position, texcoord, normal, tangent,
        // blend indices, blend weights, color, custom). Offsets are relative to the data block.
        ByteWriter vertexData;
        std::vector<iqmvertexarray> arrays;
        auto beginArray = [&](uint32_t type, uint32_t format, uint32_t size) {
            vertexData.Align(4);
            arrays.push_back({ type, 0, format, size, vertexData.Size() });
        };
        beginArray(IQM_POSITION, IQM_FLOAT, 3);
        for (const aiVector3D& p : mVertices.positions) { vertexData.F32(p.x); vertexData.F32(p.y); vertexData.F32(p.z); }
        if (mHasTexcoords) {
            beginArray(IQM_TEXCOORD, IQM_FLOAT, 2);
            for (const aiVector2D& t : mVertices.texcoords) { vertexData.F32(t.x); vertexData.F32(t.y); }
        }
        if (mHasNormals) {
            beginArray(IQM_NORMAL, IQM_FLOAT, 3);
            for (const aiVector3D& n : mVertices.normals) { vertexData.F32(n.x); vertexData.F32(n.y); vertexData.F32(n.z); }
        }
        if (mHasTangents) {
            beginArray(IQM_TANGENT, IQM_FLOAT, 4);
            for (const aiColor4D& t : mVertices.tangents) { vertexData.F32(t.r); vertexData.F32(t.g); vertexData.F32(t.b); vertexData.F32(t.a); }
        }
        if (!mVertices.blends.empty()) {
            beginArray(IQM_BLENDINDEXES, IQM_UBYTE, 4);
            for (const VertexBlend& b : mVertices.blends) vertexData.Bytes(b.joints.data(), 4);
            beginArray(IQM_BLENDWEIGHTS, IQM_UBYTE, 4);
            for (const VertexBlend& b : mVertices.blends) vertexData.Bytes(b.weights.data(), 4);
        }
        if (mHasColors) {
            beginArray(IQM_COLOR, IQM_UBYTE, 4);
            for (const std::array<uint8_t, 4>& c : mVertices.colors) vertexData.Bytes(c.data(), 4);
        }
        for (unsigned int set = 1; set < AI_MAX_NUMBER_OF_TEXTURECOORDS; ++set) {
            if (!mHasExtraTexcoords[set - 1]) continue;
            // Custom arrays are typed by IQM_CUSTOM + the offset of their name in the text.
            beginArray(IQM_CUSTOM + mExtraTexcoordNames[set - 1], IQM_FLOAT, 2);
            for (const aiVector2D& t : mVertices.extraTexcoords[set - 1]) { vertexData.F32(t.x); vertexData.F32(t.y); }
        }

        ByteWriter out;
        iqmheader header{};
        std::memcpy(header.magic, IQM_MAGIC, sizeof(IQM_MAGIC));
        header.version = IQM_VERSION;
        out.Bytes(&header, sizeof(header)); // placeholder, rewritten at the end

        const std::vector<char>& text = mStrings.Data();
        header.num_text = static_cast<uint32_t>(text.size());
        header.ofs_text = out.Size();
        out.Bytes(text.data(), text.size());
        out.Align(4);
        header.num_text = out.Size() - header.ofs_text; // iqm.cpp counts the padding as text

        header.num_meshes = static_cast<uint32_t>(mMeshes.size());
        header.ofs_meshes = mMeshes.empty() ? 0 : out.Size();
        for (const MeshRecord& m : mMeshes) {
            out.U32(m.name); out.U32(m.material);
            out.U32(m.firstVertex); out.U32(m.vertexCount);
            out.U32(m.firstTriangle); out.U32(m.triangleCount);
        }

        header.num_vertexarrays = static_cast<uint32_t>(arrays.size());
        header.num_vertexes = static_cast<uint32_t>(mVertices.positions.size());
        header.ofs_vertexarrays = out.Size();
        const uint32_t arraysSize = static_cast<uint32_t>(arrays.size() * sizeof(iqmvertexarray));
        // The data block starts 8-byte aligned after the array records, as in iqm.cpp.
        const uint32_t dataStart = (header.ofs_vertexarrays + arraysSize + 7u) & ~7u;
        for (const iqmvertexarray& array : arrays) {
            out.U32(array.type); out.U32(array.flags); out.U32(array.format); out.U32(array.size);
            out.U32(dataStart + array.offset);
        }
        out.Align(8);
        out.Bytes(vertexData.Data().data(), vertexData.Size());
        out.Align(4);

        header.num_triangles = static_cast<uint32_t>(mTriangles.size());
        header.ofs_triangles = mTriangles.empty() ? 0 : out.Size();
        for (const auto& t : mTriangles) { out.U32(t[0]); out.U32(t[1]); out.U32(t[2]); }
        header.ofs_adjacency = mAdjacency.empty() ? 0 : out.Size();
        for (const auto& a : mAdjacency) { out.U32(a[0]); out.U32(a[1]); out.U32(a[2]); }

        header.num_joints = static_cast<uint32_t>(mJoints.size());
        header.ofs_joints = mJoints.empty() ? 0 : out.Size();
        for (const Joint& joint : mJoints) {
            out.U32(joint.name);
            out.I32(joint.parent);
            const std::array<float, 10> values = ChannelValues(joint.base);
            for (float value : values) out.F32(value);
        }

        header.num_poses = static_cast<uint32_t>(mPoses.size());
        header.ofs_poses = mPoses.empty() ? 0 : out.Size();
        for (const iqmpose& pose : mPoses) {
            out.I32(pose.parent);
            out.U32(pose.mask);
            for (float value : pose.channeloffset) out.F32(value);
            for (float value : pose.channelscale) out.F32(value);
        }

        header.num_anims = static_cast<uint32_t>(mAnimations.size());
        header.ofs_anims = mAnimations.empty() ? 0 : out.Size();
        for (const iqmanim& anim : mAnimations) {
            out.U32(anim.name); out.U32(anim.first_frame); out.U32(anim.num_frames);
            out.F32(anim.framerate); out.U32(anim.flags);
        }

        header.num_frames = mFrameCount;
        header.num_framechannels = mFrameChannels;
        header.ofs_frames = mFrameData.empty() ? 0 : out.Size();
        for (uint16_t value : mFrameData) out.U16(value);
        out.Align(4);
        header.ofs_bounds = mBounds.empty() ? 0 : out.Size();
        for (const iqmbounds& b : mBounds) {
            for (float value : b.bbmin) out.F32(value);
            for (float value : b.bbmax) out.F32(value);
            out.F32(b.xyradius);
            out.F32(b.radius);
        }

        header.filesize = out.Size();
        WriteHeader(out, header);
        return out.Data();
    }

    /// Rewrites the header fields (everything after the magic) in place, little-endian.
    static void WriteHeader(ByteWriter& out, const iqmheader& header) {
        const unsigned int fields[] = { header.version, header.filesize, header.flags,
            header.num_text, header.ofs_text, header.num_meshes, header.ofs_meshes,
            header.num_vertexarrays, header.num_vertexes, header.ofs_vertexarrays,
            header.num_triangles, header.ofs_triangles, header.ofs_adjacency,
            header.num_joints, header.ofs_joints, header.num_poses, header.ofs_poses,
            header.num_anims, header.ofs_anims,
            header.num_frames, header.num_framechannels, header.ofs_frames, header.ofs_bounds,
            header.num_comment, header.ofs_comment, header.num_extensions, header.ofs_extensions };
        static_assert(sizeof(fields) == sizeof(iqmheader) - sizeof(iqmheader::magic), "every header field is written");
        size_t offset = sizeof(iqmheader::magic);
        for (unsigned int field : fields) {
            out.PatchU32(offset, field);
            offset += 4;
        }
    }

    const aiScene* mScene;
    const bool mYUp;
    const float mConfiguredFrameRate;
    const aiMatrix4x4 mAxis;

    std::vector<MeshInstance> mInstances;
    std::vector<Joint> mJoints;
    std::map<std::string, int> mJointByName;

    bool mHasTexcoords = false, mHasNormals = false, mHasTangents = false, mHasColors = false;
    std::array<bool, cMaxExtraUvSets> mHasExtraTexcoords{};
    std::array<uint32_t, cMaxExtraUvSets> mExtraTexcoordNames{};
    bool mDroppedColorSets = false, mDroppedMorphTargets = false;

    StringTable mStrings;
    std::vector<MeshRecord> mMeshes;
    VertexData mVertices;
    std::vector<std::array<uint32_t, 3>> mTriangles;
    std::vector<std::array<uint32_t, 3>> mAdjacency;

    std::vector<iqmanim> mAnimations;
    std::vector<JointTransform> mFrames;   ///< mFrameCount * joint count transforms.
    uint32_t mFrameCount = 0;
    std::vector<iqmpose> mPoses;
    uint32_t mFrameChannels = 0;
    std::vector<uint16_t> mFrameData;
    std::vector<iqmbounds> mBounds;
};

} // namespace

void ExportSceneIQM(const char* pFile, IOSystem* pIOSystem, const aiScene* pScene, const ExportProperties* pProperties) {
    IQMWriter writer(pScene, pProperties);
    const std::vector<uint8_t> bytes = writer.Write();
    std::unique_ptr<IOStream> file(pIOSystem->Open(pFile, "wb"));
    if (!file) {
        throw DeadlyExportError("Could not open output .iqm file: ", pFile);
    }
    if (file->Write(bytes.data(), 1, bytes.size()) != bytes.size()) {
        throw DeadlyExportError("Failed to write .iqm file: ", pFile);
    }
}

} // namespace Assimp

#endif // !ASSIMP_BUILD_NO_EXPORT && !ASSIMP_BUILD_NO_IQM_EXPORTER
