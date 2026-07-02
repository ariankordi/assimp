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

// assimp_merge_anim: takes the aiAnimation(s) from one file and grafts them
// onto the aiScene of another file, then exports a single self-contained
// result. Not tied to any particular importer -- it only touches assimp's
// generic scene graph (aiNode names, aiAnimation channels), so it works for
// any pair of formats assimp can read as long as the animation file's
// aiNodeAnim::mNodeName values match node/bone names in the model file
// (true whenever both were exported from the same rig, which is the usual
// reason a format splits mesh and animation across separate files).
//
// This exists because round-tripping mesh+skeleton and animation-only files
// through intermediate glTF files and general-purpose "glTF merge" tools is
// unreliable in practice (skins and animation channels frequently get
// dropped or mismatched). Since assimp already has both files fully parsed
// in memory as aiScene graphs, merging in-process and exporting once avoids
// that round trip entirely.

#include <assimp/Exporter.hpp>
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {

void PrintUsage(const char *argv0) {
    std::fprintf(stderr,
            "usage: %s <model-file> <anim-file> <output-file> [--format <id>] [--anim-index <n>]\n\n"
            "Imports <model-file> (mesh + skeleton) and <anim-file> (animation, sharing\n"
            "the same node/bone names), grafts the animation(s) from <anim-file> onto\n"
            "<model-file>'s scene, and exports the combined result to <output-file>.\n\n"
            "  --format <id>     Explicit assimp export format id (see `assimp listext`).\n"
            "                    If omitted, guessed from <output-file>'s extension.\n"
            "  --anim-index <n>  Merge only animation index n from <anim-file> instead of\n"
            "                    all of its animations.\n",
            argv0);
}

// Finds the export format id whose recommended extension matches
// <output-file>'s extension (assimp's C API has no built-in reverse lookup
// for this, unlike aiImportFile's importer-side extension sniffing).
std::string GuessExportFormatId(const std::string &outputFile) {
    const size_t dot = outputFile.find_last_of('.');
    if (dot == std::string::npos) {
        return {};
    }
    std::string ext = outputFile.substr(dot + 1);
    for (char &c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    Assimp::Exporter exporter;
    const size_t count = exporter.GetExportFormatCount();
    for (size_t i = 0; i < count; ++i) {
        const aiExportFormatDesc *desc = exporter.GetExportFormatDescription(i);
        if (ext == desc->fileExtension) {
            return desc->id;
        }
    }
    return {};
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 4) {
        PrintUsage(argv[0]);
        return 1;
    }

    const std::string modelPath = argv[1];
    const std::string animPath = argv[2];
    const std::string outputPath = argv[3];
    std::string formatId;
    int animIndex = -1;

    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            formatId = argv[++i];
        } else if (std::strcmp(argv[i], "--anim-index") == 0 && i + 1 < argc) {
            animIndex = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "unrecognized argument: %s\n", argv[i]);
            PrintUsage(argv[0]);
            return 1;
        }
    }

    if (formatId.empty()) {
        formatId = GuessExportFormatId(outputPath);
        if (formatId.empty()) {
            std::fprintf(stderr, "error: could not guess export format from '%s'; pass --format explicitly\n", outputPath.c_str());
            return 1;
        }
    }

    Assimp::Importer modelImporter;
    const aiScene *constModelScene = modelImporter.ReadFile(modelPath, 0);
    if (!constModelScene) {
        std::fprintf(stderr, "error: failed to import '%s': %s\n", modelPath.c_str(), modelImporter.GetErrorString());
        return 1;
    }
    aiScene *modelScene = const_cast<aiScene *>(constModelScene);

    Assimp::Importer animImporter;
    if (!animImporter.ReadFile(animPath, 0)) {
        std::fprintf(stderr, "error: failed to import '%s': %s\n", animPath.c_str(), animImporter.GetErrorString());
        return 1;
    }
    // Detach the anim scene from its Importer so we can transfer ownership of
    // individual aiAnimation objects into modelScene without a double free
    // when animImporter goes out of scope.
    aiScene *animScene = animImporter.GetOrphanedScene();

    if (animScene->mNumAnimations == 0) {
        std::fprintf(stderr, "error: '%s' has no animations to merge\n", animPath.c_str());
        delete animScene;
        return 1;
    }
    if (animIndex >= 0 && static_cast<unsigned int>(animIndex) >= animScene->mNumAnimations) {
        std::fprintf(stderr, "error: --anim-index %d out of range (file has %u animation(s))\n", animIndex, animScene->mNumAnimations);
        delete animScene;
        return 1;
    }

    // Collect the set of node names present in the model scene so we can warn
    // about (but still carry over) animation channels that won't attach to
    // anything -- a mismatched name is a silent no-op in most exporters.
    std::set<std::string> modelNodeNames;
    {
        std::vector<aiNode *> stack(1, modelScene->mRootNode);
        while (!stack.empty()) {
            aiNode *n = stack.back();
            stack.pop_back();
            modelNodeNames.insert(n->mName.C_Str());
            for (unsigned int i = 0; i < n->mNumChildren; ++i) stack.push_back(n->mChildren[i]);
        }
    }

    const unsigned int oldCount = modelScene->mNumAnimations;
    const unsigned int mergeCount = (animIndex >= 0) ? 1 : animScene->mNumAnimations;
    aiAnimation **merged = new aiAnimation *[oldCount + mergeCount];
    for (unsigned int i = 0; i < oldCount; ++i) merged[i] = modelScene->mAnimations[i];

    unsigned int written = 0;
    for (unsigned int i = 0; i < animScene->mNumAnimations; ++i) {
        if (animIndex >= 0 && static_cast<unsigned int>(animIndex) != i) continue;

        aiAnimation *anim = animScene->mAnimations[i];

        // Exporters (e.g. glTF2's AssetWriter) resolve each channel's
        // mNodeName against the target scene's node list and crash on an
        // unresolved name rather than skipping it gracefully, so channels
        // that don't correspond to a node in the model scene must be
        // dropped here before the merged aiAnimation is ever handed to an
        // exporter.
        std::vector<aiNodeAnim *> keptChannels;
        keptChannels.reserve(anim->mNumChannels);
        unsigned int dropped = 0;
        for (unsigned int c = 0; c < anim->mNumChannels; ++c) {
            if (modelNodeNames.find(anim->mChannels[c]->mNodeName.C_Str()) == modelNodeNames.end()) {
                delete anim->mChannels[c];
                ++dropped;
                continue;
            }
            keptChannels.push_back(anim->mChannels[c]);
        }
        if (dropped > 0) {
            std::fprintf(stderr, "warning: animation '%s': dropped %u/%u channel(s) whose node name isn't present in '%s' (name mismatch between rigs?)\n",
                    anim->mName.C_Str(), dropped, anim->mNumChannels, modelPath.c_str());
        }
        delete[] anim->mChannels;
        anim->mNumChannels = static_cast<unsigned int>(keptChannels.size());
        anim->mChannels = anim->mNumChannels ? new aiNodeAnim *[anim->mNumChannels] : nullptr;
        std::copy(keptChannels.begin(), keptChannels.end(), anim->mChannels);

        merged[oldCount + written++] = anim;
        animScene->mAnimations[i] = nullptr; // ownership transferred; keep animScene's destructor from freeing it
    }

    delete[] modelScene->mAnimations;
    modelScene->mAnimations = merged;
    modelScene->mNumAnimations = oldCount + written;

    delete animScene; // safe: transferred aiAnimation slots were nulled out above

    Assimp::Exporter exporter;
    const aiReturn result = exporter.Export(modelScene, formatId.c_str(), outputPath.c_str(), 0);
    if (result != aiReturn_SUCCESS) {
        std::fprintf(stderr, "error: export failed: %s\n", exporter.GetErrorString());
        return 1;
    }

    std::fprintf(stderr, "wrote '%s' (%u animation(s), format '%s')\n", outputPath.c_str(), modelScene->mNumAnimations, formatId.c_str());
    return 0;
}
