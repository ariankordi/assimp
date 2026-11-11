/*
Open Asset Import Library (assimp)
----------------------------------------------------------------------

Copyright (c) 2006-2021, assimp team

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

#ifndef ASSIMP_BUILD_NO_IQM_IMPORTER

#include <assimp/DefaultIOSystem.h>
#include <assimp/IOStreamBuffer.h>
#include <assimp/ai_assert.h>
#include <assimp/importerdesc.h>
#include <assimp/scene.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/Importer.hpp>
#include <assimp/ByteSwapper.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "IQMImporter.h"
#include "iqm.h"

// RESOURCES:
// http://sauerbraten.org/iqm/
// https://github.com/lsalzman/iqm

inline void swap_block( uint32_t *block, size_t size ){
    (void)block; // suppress 'unreferenced formal parameter' MSVC warning
    size >>= 2;
    for ( size_t i = 0; i < size; ++i )
        AI_SWAP4( block[ i ] );
}

static constexpr aiImporterDesc desc = {
    "Inter-Quake Model Importer",
    "",
    "",
    "",
    aiImporterFlags_SupportBinaryFlavour,
    0,
    0,
    0,
    0,
    "iqm"
};

namespace Assimp {

// ------------------------------------------------------------------------------------------------
//  Default constructor
IQMImporter::IQMImporter() :
        mScene(nullptr) {
    // empty
}

// ------------------------------------------------------------------------------------------------
//  Returns true, if file is a binary Inter-Quake Model file.
bool IQMImporter::CanRead(const std::string &pFile, IOSystem *pIOHandler, bool checkSig) const {
    const std::string extension = GetExtension(pFile);

    if (extension == "iqm")
        return true;
    else if (!extension.length() || checkSig) {
        if (!pIOHandler) {
            return true;
        }
        std::unique_ptr<IOStream> pStream(pIOHandler->Open(pFile, "rb"));
        unsigned char data[15];
        if (!pStream || 15 != pStream->Read(data, 1, 15)) {
            return false;
        }
        return !memcmp(data, "INTERQUAKEMODEL", 15);
    }
    return false;
}

// ------------------------------------------------------------------------------------------------
const aiImporterDesc *IQMImporter::GetInfo() const {
    return &desc;
}

// ------------------------------------------------------------------------------------------------
//  Helpers for the skeleton, skin and animation import (added with the IQM exporter; the mesh
//  import below is the original code with bounds checks added).
namespace {

/// Throws unless @p count elements of @p elementSize bytes at @p offset lie inside the file.
void CheckRange(uint64_t fileSize, uint64_t offset, uint64_t count, uint64_t elementSize, const char *what) {
    if (count == 0) {
        return;
    }
    if (offset > fileSize || elementSize == 0 || count > (fileSize - offset) / elementSize) {
        throw DeadlyImportError("IQM: ", what, " lie outside the file.");
    }
}

// Products of file values are computed in uint64_t throughout: size_t is 32 bits on wasm32.
constexpr uint64_t cMaxAnimationKeys = uint64_t(1) << 26; ///< Poses x frames over all animations.
constexpr uint64_t cMaxPaddedBones = uint64_t(1) << 20;   ///< Meshes x joints before weightless bones are left out.
constexpr uint64_t cMaxSkinWeights = uint64_t(1) << 26;   ///< Vertex weights over all meshes (meshes may share vertices).

/// Rejects files whose sections overlap the header or each other (sections are byte-swapped in
/// place on big-endian hosts, so an overlap could change values after they were validated), or
/// whose sections are not 4-byte aligned as iqm.txt requires (they are read in place as structs).
void CheckSectionLayout(const iqmheader &hdr) {
    struct Section { uint64_t begin, end; };
    std::vector<Section> sections;
    auto add = [&](uint64_t offset, uint64_t count, uint64_t size) {
        if (count == 0) return;
        if (size > 1 && offset % 4 != 0) {
            throw DeadlyImportError("IQM: a section is not 4-byte aligned.");
        }
        sections.push_back({ offset, offset + count * size });
    };
    add(hdr.ofs_text, hdr.num_text, 1);
    add(hdr.ofs_meshes, hdr.num_meshes, sizeof(iqmmesh));
    add(hdr.ofs_vertexarrays, hdr.num_vertexarrays, sizeof(iqmvertexarray));
    add(hdr.ofs_triangles, hdr.num_triangles, sizeof(iqmtriangle));
    add(hdr.ofs_joints, hdr.num_joints, sizeof(iqmjoint));
    add(hdr.ofs_poses, hdr.num_poses, sizeof(iqmpose));
    add(hdr.ofs_anims, hdr.num_anims, sizeof(iqmanim));
    add(hdr.ofs_frames, static_cast<uint64_t>(hdr.num_frames) * hdr.num_framechannels, sizeof(uint16_t));
    std::sort(sections.begin(), sections.end(), [](const Section &a, const Section &b) { return a.begin < b.begin; });
    uint64_t previousEnd = sizeof(iqmheader);
    for (const Section &section : sections) {
        if (section.begin < previousEnd) {
            throw DeadlyImportError("IQM: file sections overlap each other or the header.");
        }
        previousEnd = section.end;
    }
}

/// A string from the text section, or "" when the offset or its terminator is outside it.
std::string TextAt(const unsigned char *data, const iqmheader &hdr, unsigned int offset) {
    if (offset >= hdr.num_text) {
        return std::string();
    }
    const char *begin = reinterpret_cast<const char *>(data + hdr.ofs_text + offset);
    const char *end = static_cast<const char *>(memchr(begin, '\0', hdr.num_text - offset));
    return end != nullptr ? std::string(begin, end) : std::string();
}

/// IQM transforms are translate * rotate * scale, rotation stored x, y, z, w.
aiMatrix4x4 ComposeJoint(const float *translate, const float *rotate, const float *scale) {
    aiQuaternion rotation(rotate[3], rotate[0], rotate[1], rotate[2]);
    rotation.Normalize();
    return aiMatrix4x4(aiVector3D(scale[0], scale[1], scale[2]), rotation,
            aiVector3D(translate[0], translate[1], translate[2]));
}

/// Gives @p node the children in @p children (ownership moves to the node).
void AttachChildren(aiNode *node, const std::vector<aiNode *> &children) {
    if (children.empty()) {
        return;
    }
    node->mNumChildren = static_cast<unsigned int>(children.size());
    node->mChildren = new aiNode *[children.size()];
    std::copy(children.begin(), children.end(), node->mChildren);
    for (aiNode *child : children) {
        child->mParent = node;
    }
}

} // namespace

// ------------------------------------------------------------------------------------------------
//  Model 3D import implementation
void IQMImporter::InternReadFile(const std::string &file, aiScene *pScene, IOSystem *pIOHandler) {
    mPosesSwapped = false; // The importer may be reused for another file.
    // Read file into memory
    std::unique_ptr<IOStream> pStream(pIOHandler->Open(file, "rb"));
    if (!pStream) {
        throw DeadlyImportError("Failed to open file ", file, ".");
    }

    // Get the file-size and validate it, throwing an exception when fails
    const size_t fileSize = pStream->FileSize();
    if (fileSize < sizeof( iqmheader )) {
        throw DeadlyImportError("IQM-file ", file, " is too small.");
    }
    std::vector<unsigned char> buffer(fileSize);
    unsigned char *data = buffer.data();
    if (fileSize != pStream->Read(data, 1, fileSize)) {
        throw DeadlyImportError("Failed to read the file ", file, ".");
    }

    // get header
    // A copy, not a reference into the buffer: in-place swaps of other sections must not reach it.
    iqmheader hdr;
    memcpy(&hdr, data, sizeof(hdr));
    swap_block( &hdr.version, sizeof( iqmheader ) - sizeof( iqmheader::magic ) );

    // extra check for header
    if (memcmp(data, IQM_MAGIC, sizeof( IQM_MAGIC ) )
     || hdr.version != IQM_VERSION
     || hdr.filesize != fileSize) {
        throw DeadlyImportError("Bad binary header in file ", file, ".");
    }

    // Every section is read in place, so check that each lies inside the file first.
    CheckRange(fileSize, hdr.ofs_text, hdr.num_text, 1, "text");
    CheckRange(fileSize, hdr.ofs_meshes, hdr.num_meshes, sizeof(iqmmesh), "meshes");
    CheckRange(fileSize, hdr.ofs_vertexarrays, hdr.num_vertexarrays, sizeof(iqmvertexarray), "vertex arrays");
    CheckRange(fileSize, hdr.ofs_triangles, hdr.num_triangles, sizeof(iqmtriangle), "triangles");
    CheckRange(fileSize, hdr.ofs_joints, hdr.num_joints, sizeof(iqmjoint), "joints");
    CheckRange(fileSize, hdr.ofs_poses, hdr.num_poses, sizeof(iqmpose), "poses");
    CheckRange(fileSize, hdr.ofs_anims, hdr.num_anims, sizeof(iqmanim), "animations");
    if (hdr.num_framechannels != 0) {
        CheckRange(fileSize, hdr.ofs_frames, static_cast<uint64_t>(hdr.num_frames) * hdr.num_framechannels, sizeof(uint16_t), "frames");
    }
    CheckSectionLayout(hdr);

    ASSIMP_LOG_DEBUG("IQM: loading ", file);

    // create the root node
    pScene->mRootNode = new aiNode( "<IQMRoot>" );
    // Now rotate the whole scene 90 degrees around the x axis to convert to internal coordinate system
    pScene->mRootNode->mTransformation = aiMatrix4x4(
            1.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f,
            0.f, -1.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 1.f);
    pScene->mRootNode->mNumMeshes = hdr.num_meshes;
    pScene->mRootNode->mMeshes = new unsigned int[hdr.num_meshes];
    std::iota( pScene->mRootNode->mMeshes, pScene->mRootNode->mMeshes + pScene->mRootNode->mNumMeshes, 0 );

    mScene = pScene;

    // Joints become nodes under the root, next to the meshes, so a joint's global transform
    // relative to the root is its bind pose in the same space as the vertices. An animation-only
    // file has poses but no joints; it gets nodes "pose<n>" with identity transforms instead, so
    // its channels have nodes to target.
    if (hdr.num_joints == 0 && hdr.num_poses > 0) {
        ReadPoseNodes(data, hdr);
    }
    std::vector<std::string> jointNames(hdr.num_joints);
    std::vector<aiMatrix4x4> jointGlobals(hdr.num_joints);
    {
        auto joints = reinterpret_cast<iqmjoint *>(data + hdr.ofs_joints);
        std::vector<aiNode *> nodes(hdr.num_joints);
        std::vector<std::vector<aiNode *>> children(hdr.num_joints);
        std::vector<aiNode *> rootChildren;
        for (unsigned int i = 0; i < hdr.num_joints; ++i) {
            iqmjoint &joint = joints[i];
            swap_block(reinterpret_cast<uint32_t *>(&joint), sizeof(iqmjoint));
            // IQM lists parents before their children; anything else is not a tree.
            if (joint.parent >= static_cast<int>(i)) {
                // None of the nodes made so far is attached yet, so each is deleted on its own.
                for (unsigned int made = 0; made < i; ++made) delete nodes[made];
                throw DeadlyImportError("IQM: joint ", i, " has parent ", joint.parent, ", which does not precede it.");
            }
            jointNames[i] = TextAt(data, hdr, joint.name);
            if (jointNames[i].empty()) {
                jointNames[i] = "joint" + std::to_string(i);
            }
            const aiMatrix4x4 local = ComposeJoint(joint.translate, joint.rotate, joint.scale);
            jointGlobals[i] = joint.parent >= 0 ? jointGlobals[joint.parent] * local : local;
            nodes[i] = new aiNode(jointNames[i]);
            nodes[i]->mTransformation = local;
            (joint.parent >= 0 ? children[joint.parent] : rootChildren).push_back(nodes[i]);
        }
        for (unsigned int i = 0; i < hdr.num_joints; ++i) {
            AttachChildren(nodes[i], children[i]);
        }
        AttachChildren(pScene->mRootNode, rootChildren);
    }

    // Allocate output storage
    pScene->mNumMeshes = 0;
    pScene->mMeshes = new aiMesh *[hdr.num_meshes](); // Set arrays to zero to ensue proper destruction if an exception is raised

    pScene->mNumMaterials = 0;
    pScene->mMaterials = new aiMaterial *[hdr.num_meshes]();

    // swap vertex arrays beforehand...
    for( auto array = reinterpret_cast<iqmvertexarray*>( data + hdr.ofs_vertexarrays ), end = array + hdr.num_vertexarrays; array != end; ++array )
    {
        swap_block( &array->type, sizeof( iqmvertexarray ) );
        // Sizes of the formats the reader below understands; others are skipped, not read.
        const size_t componentSize = array->format == IQM_FLOAT ? 4 : array->format == IQM_UBYTE ? 1 : 0;
        if (componentSize == 4 && array->offset % 4 != 0) {
            throw DeadlyImportError("IQM: a float vertex array is not 4-byte aligned.");
        }
        if (componentSize != 0) {
            CheckRange(fileSize, array->offset, static_cast<uint64_t>(hdr.num_vertexes) * array->size, componentSize, "vertex data");
        }
    }
    // Triangles are swapped per mesh below, so swap them all once here instead (meshes may share ranges).
    swap_block(reinterpret_cast<uint32_t *>(data + hdr.ofs_triangles), static_cast<size_t>(hdr.num_triangles) * sizeof(iqmtriangle));

    // Read all surfaces from the file
    uint64_t totalSkinWeights = 0; // Over all meshes, which may share vertex ranges.
    for( auto imesh = reinterpret_cast<iqmmesh*>( data + hdr.ofs_meshes ), end_ = imesh + hdr.num_meshes; imesh != end_; ++imesh )
    {
        swap_block( &imesh->name, sizeof( iqmmesh ) );
        if (imesh->first_vertex > hdr.num_vertexes || imesh->num_vertexes > hdr.num_vertexes - imesh->first_vertex
                || imesh->first_triangle > hdr.num_triangles || imesh->num_triangles > hdr.num_triangles - imesh->first_triangle) {
            throw DeadlyImportError("IQM: a mesh's vertex or triangle range lies outside the file's arrays.");
        }
        // Allocate output mesh & material
        auto mesh = pScene->mMeshes[pScene->mNumMeshes++] = new aiMesh();
        mesh->mName = TextAt(data, hdr, imesh->name);
        mesh->mMaterialIndex = pScene->mNumMaterials;
        auto mat = pScene->mMaterials[pScene->mNumMaterials++] = new aiMaterial();

        {
            aiString name( TextAt(data, hdr, imesh->material) );
            mat->AddProperty( &name, AI_MATKEY_NAME );
            mat->AddProperty( &name, AI_MATKEY_TEXTURE_DIFFUSE(0) );
        }

        // Fill mesh information
        mesh->mPrimitiveTypes = aiPrimitiveType_TRIANGLE;
        mesh->mNumFaces = 0;
        mesh->mFaces = new aiFace[imesh->num_triangles];

        // Fill in all triangles
        for( auto tri = reinterpret_cast<iqmtriangle*>( data + hdr.ofs_triangles ) + imesh->first_triangle, end = tri + imesh->num_triangles; tri != end; ++tri )
        {
            for (unsigned int corner : tri->vertex) {
                if (corner < imesh->first_vertex || corner - imesh->first_vertex >= imesh->num_vertexes) {
                    throw DeadlyImportError("IQM: a triangle uses a vertex outside its mesh.");
                }
            }
            auto& face = mesh->mFaces[mesh->mNumFaces++];
            face.mNumIndices = 3;
            face.mIndices = new unsigned int[3]{ tri->vertex[0] - imesh->first_vertex,
                                                 tri->vertex[2] - imesh->first_vertex,
                                                 tri->vertex[1] - imesh->first_vertex };
        }

        // Blend data, turned into bones after all arrays are read.
        const unsigned char *blendIndices = nullptr;
        const unsigned char *blendWeightBytes = nullptr;
        const float *blendWeightFloats = nullptr;
        unsigned int blendIndexStep = 0, blendWeightStep = 0;
        const float *tangentData = nullptr;
        unsigned int tangentStep = 0;

        // Fill in all vertices
        for( auto array = reinterpret_cast<const iqmvertexarray*>( data + hdr.ofs_vertexarrays ), end__ = array + hdr.num_vertexarrays; array != end__; ++array )
        {
            const unsigned int nVerts = imesh->num_vertexes;
            const unsigned int step = array->size;

            switch ( array->type )
            {
            case IQM_POSITION:
                if( array->format == IQM_FLOAT && step >= 3 ){
                    mesh->mNumVertices = nVerts;
                    auto v = mesh->mVertices = new aiVector3D[nVerts];
                    for( auto f = reinterpret_cast<const float*>( data + array->offset ) + imesh->first_vertex * step,
                        end = f + nVerts * step; f != end; f += step, ++v )
                    {
                        *v = { AI_BE( f[0] ),
                               AI_BE( f[1] ),
                               AI_BE( f[2] ) };
                    }
                }
                break;
               case IQM_TEXCOORD:
                if( array->format == IQM_FLOAT && step >= 2)
                {
                    auto v = mesh->mTextureCoords[0] = new aiVector3D[nVerts];
                    mesh->mNumUVComponents[0] = 2;
                    for( auto f = reinterpret_cast<const float*>( data + array->offset ) + imesh->first_vertex * step,
                        end = f + nVerts * step; f != end; f += step, ++v )
                    {
                        *v = { AI_BE( f[0] ),
                           1 - AI_BE( f[1] ), 0 };
                    }
                }
                break;
            case IQM_NORMAL:
                if (array->format == IQM_FLOAT && step >= 3)
                {
                    auto v = mesh->mNormals = new aiVector3D[nVerts];
                    for( auto f = reinterpret_cast<const float*>( data + array->offset ) + imesh->first_vertex * step,
                        end = f + nVerts * step; f != end; f += step, ++v )
                    {
                        *v = { AI_BE( f[0] ),
                               AI_BE( f[1] ),
                               AI_BE( f[2] ) };
                    }
                }
                break;
            case IQM_COLOR:
                if (array->format == IQM_UBYTE && step >= 3)
                {
                    auto v = mesh->mColors[0] = new aiColor4D[nVerts];
                    for( auto f = ( data + array->offset ) + imesh->first_vertex * step,
                        end = f + nVerts * step; f != end; f += step, ++v )
                    {
                        *v = { ( f[0] ) / 255.f,
                               ( f[1] ) / 255.f,
                               ( f[2] ) / 255.f,
                               step == 3? 1 : ( f[3] ) / 255.f };
                    }
                }
                else if (array->format == IQM_FLOAT && step >= 3)
                {
                    auto v = mesh->mColors[0] = new aiColor4D[nVerts];
                    for( auto f = reinterpret_cast<const float*>( data + array->offset ) + imesh->first_vertex * step,
                        end = f + nVerts * step; f != end; f += step, ++v )
                    {
                        *v = { AI_BE( f[0] ),
                               AI_BE( f[1] ),
                               AI_BE( f[2] ),
                               step == 3? 1 : AI_BE( f[3] ) };
                    }
                }
                break;
            case IQM_TANGENT:
                // Read after the loop: the bitangent needs the normals (the sign is in w).
                if (array->format == IQM_FLOAT && step >= 4) {
                    tangentData = reinterpret_cast<const float *>(data + array->offset) + imesh->first_vertex * step;
                    tangentStep = step;
                }
                break;
            case IQM_BLENDINDEXES:
                if (array->format == IQM_UBYTE && step >= 1) {
                    blendIndices = data + array->offset + imesh->first_vertex * step;
                    blendIndexStep = step;
                }
                break;
            case IQM_BLENDWEIGHTS:
                if (array->format == IQM_UBYTE && step >= 1) {
                    blendWeightBytes = data + array->offset + imesh->first_vertex * step;
                    blendWeightStep = step;
                } else if (array->format == IQM_FLOAT && step >= 1) {
                    blendWeightFloats = reinterpret_cast<const float *>(data + array->offset) + imesh->first_vertex * step;
                    blendWeightStep = step;
                }
                break;
            case IQM_CUSTOM:
                break; // these attributes are not relevant.

            default:
                break;
            }
        }

        // Tangents: bitangent = (normal x tangent) * w, the same rule as the glTF2 importer
        // (IQM and glTF both have V running top to bottom).
        if (tangentData != nullptr && mesh->mNormals != nullptr && mesh->mNumVertices == imesh->num_vertexes) {
            mesh->mTangents = new aiVector3D[mesh->mNumVertices];
            mesh->mBitangents = new aiVector3D[mesh->mNumVertices];
            for (unsigned int v = 0; v < mesh->mNumVertices; ++v) {
                const float *t = tangentData + v * tangentStep;
                mesh->mTangents[v] = aiVector3D(AI_BE(t[0]), AI_BE(t[1]), AI_BE(t[2]));
                mesh->mBitangents[v] = (mesh->mNormals[v] ^ mesh->mTangents[v]) * AI_BE(t[3]);
            }
        }

        // Skin: one aiBone per joint, offset = inverse bind pose. Joints without weights get a
        // bone too, as in the glTF2 importer, so the whole skeleton survives a re-export.
        const bool hasWeights = blendWeightBytes != nullptr || blendWeightFloats != nullptr;
        // Weightless bones keep the whole skeleton through a re-export, but cost a bone per joint
        // per mesh; leave them out when a file asks for an absurd number of those.
        const bool padBones = static_cast<uint64_t>(hdr.num_meshes) * hdr.num_joints <= cMaxPaddedBones;
        if (blendIndices != nullptr && hasWeights && hdr.num_joints > 0 && mesh->mNumVertices > 0
                && mesh->mNumVertices == imesh->num_vertexes) {
            std::vector<std::vector<aiVertexWeight>> weightsPerJoint(hdr.num_joints);
            const unsigned int influences = std::min(blendIndexStep, blendWeightStep);
            totalSkinWeights += static_cast<uint64_t>(mesh->mNumVertices) * influences;
            if (totalSkinWeights > cMaxSkinWeights) {
                throw DeadlyImportError("IQM: the meshes ask for more than ", cMaxSkinWeights, " skin weights.");
            }
            for (unsigned int v = 0; v < mesh->mNumVertices; ++v) {
                for (unsigned int k = 0; k < influences; ++k) {
                    const unsigned int joint = blendIndices[v * blendIndexStep + k];
                    const float weight = blendWeightBytes != nullptr
                            ? blendWeightBytes[v * blendWeightStep + k] / 255.0f
                            : AI_BE(blendWeightFloats[v * blendWeightStep + k]);
                    if (weight <= 0.0f) {
                        continue;
                    }
                    if (joint >= hdr.num_joints) {
                        throw DeadlyImportError("IQM: a vertex is weighted to joint ", joint, " of ", hdr.num_joints, ".");
                    }
                    weightsPerJoint[joint].emplace_back(v, weight);
                }
            }
            std::vector<aiBone *> bones;
            for (unsigned int j = 0; j < hdr.num_joints; ++j) {
                if (!padBones && weightsPerJoint[j].empty()) {
                    continue;
                }
                auto bone = new aiBone();
                bone->mName = jointNames[j];
                bone->mOffsetMatrix = aiMatrix4x4(jointGlobals[j]).Inverse();
                bone->mNumWeights = static_cast<unsigned int>(weightsPerJoint[j].size());
                if (bone->mNumWeights > 0) {
                    bone->mWeights = new aiVertexWeight[bone->mNumWeights];
                    std::copy(weightsPerJoint[j].begin(), weightsPerJoint[j].end(), bone->mWeights);
                }
                bones.push_back(bone);
            }
            if (!bones.empty()) {
                mesh->mNumBones = static_cast<unsigned int>(bones.size());
                mesh->mBones = new aiBone *[bones.size()];
                std::copy(bones.begin(), bones.end(), mesh->mBones);
            }
        }
    }

    ReadAnimations(data, hdr, jointNames);
}

// ------------------------------------------------------------------------------------------------
//  Nodes for an animation-only file, built from the pose parents. Swaps the poses in place, so
//  ReadAnimations must not swap them again (it checks mPosesSwapped).
void IQMImporter::ReadPoseNodes(unsigned char *data, const iqmheader &hdr) {
    auto poses = reinterpret_cast<iqmpose *>(data + hdr.ofs_poses);
    std::vector<aiNode *> nodes(hdr.num_poses);
    std::vector<std::vector<aiNode *>> children(hdr.num_poses);
    std::vector<aiNode *> rootChildren;
    for (unsigned int i = 0; i < hdr.num_poses; ++i) {
        swap_block(reinterpret_cast<uint32_t *>(&poses[i]), sizeof(iqmpose));
        if (poses[i].parent >= static_cast<int>(i)) {
            for (unsigned int made = 0; made < i; ++made) delete nodes[made];
            throw DeadlyImportError("IQM: pose ", i, " has parent ", poses[i].parent, ", which does not precede it.");
        }
        nodes[i] = new aiNode("pose" + std::to_string(i));
        (poses[i].parent >= 0 ? children[poses[i].parent] : rootChildren).push_back(nodes[i]);
    }
    for (unsigned int i = 0; i < hdr.num_poses; ++i) {
        AttachChildren(nodes[i], children[i]);
    }
    AttachChildren(mScene->mRootNode, rootChildren);
    mPosesSwapped = true;
}

// ------------------------------------------------------------------------------------------------
//  Animations: one aiAnimation per IQM anim, one channel per pose with a key at every frame.
//  Ticks are frames: mTicksPerSecond is the anim's frame rate, mDuration its last frame.
void IQMImporter::ReadAnimations(unsigned char *data, const iqmheader &hdr, const std::vector<std::string> &jointNames) {
    if (hdr.num_anims == 0 || hdr.num_poses == 0) {
        return;
    }
    auto poses = reinterpret_cast<iqmpose *>(data + hdr.ofs_poses);
    unsigned int channelsPerFrame = 0;
    for (unsigned int p = 0; p < hdr.num_poses; ++p) {
        if (!mPosesSwapped) {
            swap_block(reinterpret_cast<uint32_t *>(&poses[p]), sizeof(iqmpose));
        }
        for (unsigned int k = 0; k < 10; ++k) {
            channelsPerFrame += (poses[p].mask >> k) & 1u;
        }
    }
    if (channelsPerFrame != hdr.num_framechannels) {
        throw DeadlyImportError("IQM: the pose masks give ", channelsPerFrame, " channels per frame, the header ", hdr.num_framechannels, ".");
    }
    auto frames = reinterpret_cast<uint16_t *>(data + hdr.ofs_frames);
    for (uint64_t i = 0, count = static_cast<uint64_t>(hdr.num_frames) * hdr.num_framechannels; i < count; ++i) {
        AI_SWAP2(frames[i]);
    }

    auto anims = reinterpret_cast<iqmanim *>(data + hdr.ofs_anims);
    // Frames with no channels take no file space, so their count is otherwise unbounded:
    // cap the keys all animations would allocate before allocating any.
    uint64_t totalKeys = 0;
    for (unsigned int a = 0; a < hdr.num_anims; ++a) {
        uint32_t frameCount = anims[a].num_frames;
        AI_SWAP4(frameCount);
        // Both factors are 32-bit and the running total stays below the cap, so this cannot wrap.
        totalKeys += static_cast<uint64_t>(frameCount) * hdr.num_poses;
        if (totalKeys > cMaxAnimationKeys) {
            throw DeadlyImportError("IQM: the animations ask for more than ", cMaxAnimationKeys, " keys.");
        }
    }
    mScene->mAnimations = new aiAnimation *[hdr.num_anims]();
    for (unsigned int a = 0; a < hdr.num_anims; ++a) {
        iqmanim &record = anims[a];
        swap_block(reinterpret_cast<uint32_t *>(&record), sizeof(iqmanim));
        if (record.first_frame > hdr.num_frames || record.num_frames > hdr.num_frames - record.first_frame) {
            throw DeadlyImportError("IQM: animation ", a, " uses frames outside the file's frames.");
        }
        auto animation = mScene->mAnimations[mScene->mNumAnimations++] = new aiAnimation();
        animation->mName = TextAt(data, hdr, record.name);
        animation->mTicksPerSecond = record.framerate > 0.0f ? record.framerate : 0.0;
        animation->mDuration = record.num_frames > 1 ? record.num_frames - 1 : 0;
        if (record.num_frames == 0) {
            continue;
        }
        animation->mNumChannels = hdr.num_poses;
        animation->mChannels = new aiNodeAnim *[hdr.num_poses]();
        for (unsigned int p = 0; p < hdr.num_poses; ++p) {
            auto channel = animation->mChannels[p] = new aiNodeAnim();
            channel->mNodeName = p < jointNames.size() ? jointNames[p] : "pose" + std::to_string(p);
            channel->mNumPositionKeys = channel->mNumRotationKeys = channel->mNumScalingKeys = record.num_frames;
            channel->mPositionKeys = new aiVectorKey[record.num_frames];
            channel->mRotationKeys = new aiQuatKey[record.num_frames];
            channel->mScalingKeys = new aiVectorKey[record.num_frames];
        }
        for (unsigned int f = 0; f < record.num_frames; ++f) {
            const uint16_t *values = frames + static_cast<size_t>(record.first_frame + f) * hdr.num_framechannels;
            for (unsigned int p = 0; p < hdr.num_poses; ++p) {
                const iqmpose &pose = poses[p];
                float channels[10];
                for (unsigned int k = 0; k < 10; ++k) {
                    channels[k] = pose.channeloffset[k];
                    if ((pose.mask >> k) & 1u) {
                        channels[k] += *values++ * pose.channelscale[k];
                    }
                }
                aiNodeAnim *channel = animation->mChannels[p];
                const double time = f;
                aiQuaternion rotation(channels[6], channels[3], channels[4], channels[5]);
                rotation.Normalize();
                channel->mPositionKeys[f] = aiVectorKey(time, aiVector3D(channels[0], channels[1], channels[2]));
                channel->mRotationKeys[f] = aiQuatKey(time, rotation);
                channel->mScalingKeys[f] = aiVectorKey(time, aiVector3D(channels[7], channels[8], channels[9]));
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------

} // Namespace Assimp

#endif // !! ASSIMP_BUILD_NO_IQM_IMPORTER
