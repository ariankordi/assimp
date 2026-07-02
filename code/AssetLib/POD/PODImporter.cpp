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
#include "PVRAssets/fileio/PODReader.h"
#include "PVRCore/stream/BufferStream.h"

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

// PODReader.cpp negates the rotation quaternion's w component while
// parsing both static node rotations and animation keyframes (it does
// this unconditionally, regardless of the authoring SDK version). That
// negation reverses rotation direction and, empirically, produces a
// skeleton reflected top-to-bottom for a file authored with
// SDK 3.5: composing the raw node hierarchy with PVRAssets' own
// getWorldMatrix()/constructSRT() puts e.g. Spine_2 *below* Waist
// instead of above it. Negating w back here undoes PODReader's
// negation and reproduces the expected bind pose.
aiQuaternion ToAiQuaternion(const glm::quat &q) {
    return {-q.w, q.x, q.y, q.z};
}

aiVector3D ToAiVector3D(const glm::vec3 &v) {
    return {v.x, v.y, v.z};
}

// Composes a node's bind-pose world transform by walking its aiNode
// parent chain (all of which were just built from POD's local SRT data,
// via the corrected ToAiQuaternion above). Used instead of trusting
// PVRAssets' own precomputed Skeleton::invBindMatrices, because those
// were baked in at POD-load time using PVRAssets' own (uncorrected)
// quaternion convention -- mixing them with our corrected node
// hierarchy would make bone world transform and inverse bind matrix
// mutually inconsistent, distorting the mesh even in bind pose.
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

void PODImporter::BuildNodes(Model &model, aiScene *pScene) {
    const uint32_t numNodes = model.getNumNodes();
    mNodesByIndex.assign(numNodes, nullptr);

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

        aiNode *ainode = new aiNode(name);
        ainode->mTransformation = aiMatrix4x4(
                ToAiVector3D(node.getInternalData().getScale()),
                ToAiQuaternion(node.getInternalData().getRotate()),
                ToAiVector3D(node.getInternalData().getTranslation()));
        mNodesByIndex[i] = ainode;
    }

    std::vector<aiNode *> rootLevelNodes;
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
        }
    }

    if (rootLevelNodes.size() == 1) {
        pScene->mRootNode = rootLevelNodes[0];
    } else {
        // POD's own top level can have several sibling nodes (e.g. mesh nodes
        // sitting alongside a joint hierarchy root), but assimp needs exactly
        // one aiScene::mRootNode. Rather than inventing a new node, reuse
        // whichever existing top-level node has the most descendants (in
        // practice, the joint hierarchy root) and reparent the rest under it;
        // its transform is identity in every POD file seen so far, so this is
        // a purely structural change. Only fabricate a node if every top-level
        // node is a leaf (no reasonable candidate to reuse).
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
        if (bestDescendants > 0) {
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
                bone->mOffsetMatrix = ComputeWorldTransform(boneNode);
                bone->mOffsetMatrix.Inverse();
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

void PODImporter::BuildAnimations(Model &model, aiScene *pScene) {
    Model::InternalData &modelData = model.getInternalData();
    if (modelData.animationsData.empty() || modelData.animationInstances.empty()) {
        return;
    }

    AnimationData &animData = modelData.animationsData[0];
    AnimationData::InternalData &animInternal = animData.getInternalData();
    AnimationInstance &instance = model.getAnimationInstance(0);

    float fps = model.getFPS();
    if (fps <= 0.0f) fps = 30.0f;

    // Map each Node's address (as used by AnimationInstance::KeyframeChannel::nodes)
    // back to its POD node index, so channels can be attached to the right aiNode.
    std::unordered_map<const void *, uint32_t> nodePtrToIndex;
    const uint32_t numNodes = model.getNumNodes();
    for (uint32_t i = 0; i < numNodes; ++i) {
        nodePtrToIndex[static_cast<const void *>(&model.getNode(i))] = i;
    }

    struct Channels {
        std::vector<aiVectorKey> position;
        std::vector<aiQuatKey> rotation;
        std::vector<aiVectorKey> scaling;
    };
    std::map<uint32_t, Channels> perNode;
    double maxTicks = 0.0;

    for (const AnimationInstance::KeyframeChannel &channel : instance.keyframeChannels) {
        if (channel.keyFrame >= animInternal.keyFrames.size()) continue;
        const KeyFrameData &kf = animInternal.keyFrames[channel.keyFrame];

        for (void *rawNode : channel.nodes) {
            auto it = nodePtrToIndex.find(rawNode);
            if (it == nodePtrToIndex.end()) continue;
            Channels &out = perNode[it->second];

            if (!kf.translation.empty()) {
                for (size_t k = 0; k < kf.translation.size(); ++k) {
                    const double t = static_cast<double>(kf.timeInSeconds[k]) * fps;
                    out.position.emplace_back(t, ToAiVector3D(kf.translation[k]));
                    maxTicks = std::max(maxTicks, t);
                }
            }
            if (!kf.rotate.empty()) {
                for (size_t k = 0; k < kf.rotate.size(); ++k) {
                    const double t = static_cast<double>(kf.timeInSeconds[k]) * fps;
                    out.rotation.emplace_back(t, ToAiQuaternion(kf.rotate[k]));
                    maxTicks = std::max(maxTicks, t);
                }
            }
            if (!kf.scale.empty()) {
                for (size_t k = 0; k < kf.scale.size(); ++k) {
                    const double t = static_cast<double>(kf.timeInSeconds[k]) * fps;
                    out.scaling.emplace_back(t, ToAiVector3D(kf.scale[k]));
                    maxTicks = std::max(maxTicks, t);
                }
            }
            // Matrix-only channels (kf.mat4) aren't produced by SRT-animated POD
            // exports like from Miitomo; not handled here.
        }
    }

    if (perNode.empty()) {
        return;
    }

    aiAnimation *anim = new aiAnimation();
    anim->mName = aiString(animInternal.animationName.empty() ? std::string("Take") : animInternal.animationName);
    anim->mTicksPerSecond = fps;
    anim->mDuration = maxTicks;

    anim->mNumChannels = static_cast<unsigned int>(perNode.size());
    anim->mChannels = new aiNodeAnim *[anim->mNumChannels];
    unsigned int channelIdx = 0;
    for (auto &entry : perNode) {
        aiNodeAnim *nodeAnim = new aiNodeAnim();
        nodeAnim->mNodeName = mNodesByIndex[entry.first]->mName;

        Channels &c = entry.second;
        nodeAnim->mNumPositionKeys = static_cast<unsigned int>(c.position.size());
        if (nodeAnim->mNumPositionKeys) {
            nodeAnim->mPositionKeys = new aiVectorKey[c.position.size()];
            std::copy(c.position.begin(), c.position.end(), nodeAnim->mPositionKeys);
        }
        nodeAnim->mNumRotationKeys = static_cast<unsigned int>(c.rotation.size());
        if (nodeAnim->mNumRotationKeys) {
            nodeAnim->mRotationKeys = new aiQuatKey[c.rotation.size()];
            std::copy(c.rotation.begin(), c.rotation.end(), nodeAnim->mRotationKeys);
        }
        nodeAnim->mNumScalingKeys = static_cast<unsigned int>(c.scaling.size());
        if (nodeAnim->mNumScalingKeys) {
            nodeAnim->mScalingKeys = new aiVectorKey[c.scaling.size()];
            std::copy(c.scaling.begin(), c.scaling.end(), nodeAnim->mScalingKeys);
        }

        anim->mChannels[channelIdx++] = nodeAnim;
    }

    pScene->mNumAnimations = 1;
    pScene->mAnimations = new aiAnimation *[1];
    pScene->mAnimations[0] = anim;
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

    pvr::BufferStream stream(pFile, buffer.data(), buffer.size());

    Model model;
    try {
        pvr::assets::readPOD(stream, model);
    } catch (const std::exception &e) {
        throw DeadlyImportError("POD: failed to parse file ", pFile, ": ", e.what());
    }

    BuildNodes(model, pScene);
    BuildMeshesAndMaterials(model, pScene);
    BuildAnimations(model, pScene);
}

} // namespace Assimp

#endif // ASSIMP_BUILD_NO_POD_IMPORTER
