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
#include <assimp/config.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

// Assimp's glTF2 exporter (glTF2Exporter.cpp, ExportSkin) builds each skin's
// joint list by scanning aiMesh::mBones in mesh order, then bone order within
// each mesh, and assigns joint indices by first-seen order. Downstream pose
// composers -- raylib's BuildPoseFromParentJoints, used for both its glTF and
// IQM loaders -- hard-require bones[i].parent < i and silently drop any bone
// that violates it (leaving it un-parented), which shows up as a distorted
// skeleton. DCC tools commonly emit mBones in vertex-group creation order,
// which has no relation to skeleton depth, so the invariant routinely doesn't
// hold on merged/re-exported rigs. This reorders each mesh's aiBone list into
// scene-graph (parent-before-child) order, padding in zero-weight placeholder
// bones for any ancestor missing from a given mesh so that the exporter's
// global first-seen order comes out correct too. It doesn't change skinning:
// aiBone array order carries no meaning of its own beyond which joint index a
// bone name resolves to.
void SortBonesTopologically(aiScene *scene) {
    // Pre-order rank over the node tree: a parent's rank is always lower than
    // any of its descendants'.
    std::map<std::string, int> rank;
    std::map<std::string, aiNode *> nodeByName;
    {
        std::vector<aiNode *> stack(1, scene->mRootNode);
        int counter = 0;
        while (!stack.empty()) {
            aiNode *n = stack.back();
            stack.pop_back();
            rank[n->mName.C_Str()] = counter++;
            nodeByName[n->mName.C_Str()] = n;
            for (unsigned int i = n->mNumChildren; i-- > 0;) stack.push_back(n->mChildren[i]);
        }
    }

    // One representative aiBone per name, to source the offset/bind matrix
    // for any zero-weight placeholder bones added below.
    std::map<std::string, const aiBone *> representative;
    for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
        const aiMesh *mesh = scene->mMeshes[m];
        for (unsigned int b = 0; b < mesh->mNumBones; ++b) {
            representative.emplace(mesh->mBones[b]->mName.C_Str(), mesh->mBones[b]);
        }
    }

    unsigned int totalPadded = 0;
    for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
        aiMesh *mesh = scene->mMeshes[m];
        if (mesh->mNumBones == 0) continue;

        std::set<std::string> present;
        for (unsigned int b = 0; b < mesh->mNumBones; ++b) present.insert(mesh->mBones[b]->mName.C_Str());

        // Any ancestor of a bone already in this mesh that is itself a bone
        // somewhere in the scene, but isn't in this mesh yet, needs a
        // zero-weight placeholder so it gets visited (and assigned a joint
        // index) before its descendant, regardless of which mesh the
        // exporter happens to process first.
        std::vector<aiBone *> additions;
        for (unsigned int b = 0; b < mesh->mNumBones; ++b) {
            auto nodeIt = nodeByName.find(mesh->mBones[b]->mName.C_Str());
            aiNode *ancestor = (nodeIt != nodeByName.end()) ? nodeIt->second->mParent : nullptr;
            while (ancestor != nullptr) {
                const std::string ancestorName = ancestor->mName.C_Str();
                auto repIt = representative.find(ancestorName);
                if (repIt != representative.end() && present.find(ancestorName) == present.end()) {
                    aiBone *phantom = new aiBone(*repIt->second);
                    delete[] phantom->mWeights;
                    phantom->mWeights = nullptr;
                    phantom->mNumWeights = 0;
                    additions.push_back(phantom);
                    present.insert(ancestorName);
                }
                ancestor = ancestor->mParent;
            }
        }

        if (!additions.empty()) {
            aiBone **combined = new aiBone *[mesh->mNumBones + additions.size()];
            std::copy(mesh->mBones, mesh->mBones + mesh->mNumBones, combined);
            std::copy(additions.begin(), additions.end(), combined + mesh->mNumBones);
            delete[] mesh->mBones;
            mesh->mBones = combined;
            mesh->mNumBones += static_cast<unsigned int>(additions.size());
            totalPadded += static_cast<unsigned int>(additions.size());
        }

        std::stable_sort(mesh->mBones, mesh->mBones + mesh->mNumBones,
                [&rank](const aiBone *a, const aiBone *b) {
                    const auto ra = rank.find(a->mName.C_Str());
                    const auto rb = rank.find(b->mName.C_Str());
                    const int ia = (ra != rank.end()) ? ra->second : std::numeric_limits<int>::max();
                    const int ib = (rb != rank.end()) ? rb->second : std::numeric_limits<int>::max();
                    return ia < ib;
                });
    }

    if (totalPadded > 0) {
        std::fprintf(stderr, "note: added %u zero-weight placeholder bone(s) across meshes to keep the exported skin's joint order parent-before-child\n", totalPadded);
    }
}

// Some DCC exports give each mesh a skin containing only the bones that
// mesh's vertices are actually weighted to, rather than the full skeleton.
// Assimp's glTF2 exporter assigns joint indices by first-seen order while
// scanning meshes (see SortBonesTopologically above), so a bone that's
// simply absent from one mesh's aiBone list still gets a consistent index
// as long as it's present in the union of all meshes -- but on models where
// a bone in the "total" skeleton isn't referenced by *any* mesh, or where
// per-mesh bone subsets disagree in relative order, joint indices can drift
// between skins. This pads every mesh's aiBone list out to the full set of
// bone names referenced anywhere in the scene (zero-weight placeholders for
// bones the mesh doesn't influence) and sorts each list into the same
// parent-before-child scene-graph order, so every mesh's skin agrees on
// exactly the same bone name -> joint index mapping.
void UnifySkinsWithSkeleton(aiScene *scene) {
    std::map<std::string, int> rank;
    {
        std::vector<aiNode *> stack(1, scene->mRootNode);
        int counter = 0;
        while (!stack.empty()) {
            aiNode *n = stack.back();
            stack.pop_back();
            rank[n->mName.C_Str()] = counter++;
            for (unsigned int i = n->mNumChildren; i-- > 0;) stack.push_back(n->mChildren[i]);
        }
    }

    // Full skeleton: every bone name referenced by any mesh's skin, with one
    // representative aiBone per name to source the offset matrix for
    // placeholders added to meshes that don't already have that bone.
    std::map<std::string, const aiBone *> representative;
    for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
        const aiMesh *mesh = scene->mMeshes[m];
        for (unsigned int b = 0; b < mesh->mNumBones; ++b) {
            representative.emplace(mesh->mBones[b]->mName.C_Str(), mesh->mBones[b]);
        }
    }

    unsigned int totalPadded = 0;
    for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
        aiMesh *mesh = scene->mMeshes[m];
        if (mesh->mNumBones == 0) continue; // unskinned mesh: leave alone

        std::set<std::string> present;
        for (unsigned int b = 0; b < mesh->mNumBones; ++b) present.insert(mesh->mBones[b]->mName.C_Str());

        std::vector<aiBone *> additions;
        for (const auto &entry : representative) {
            if (present.find(entry.first) != present.end()) continue;
            aiBone *phantom = new aiBone(*entry.second);
            delete[] phantom->mWeights;
            phantom->mWeights = nullptr;
            phantom->mNumWeights = 0;
            additions.push_back(phantom);
        }

        if (!additions.empty()) {
            aiBone **combined = new aiBone *[mesh->mNumBones + additions.size()];
            std::copy(mesh->mBones, mesh->mBones + mesh->mNumBones, combined);
            std::copy(additions.begin(), additions.end(), combined + mesh->mNumBones);
            delete[] mesh->mBones;
            mesh->mBones = combined;
            mesh->mNumBones += static_cast<unsigned int>(additions.size());
            totalPadded += static_cast<unsigned int>(additions.size());
        }

        std::stable_sort(mesh->mBones, mesh->mBones + mesh->mNumBones,
                [&rank](const aiBone *a, const aiBone *b) {
                    const auto ra = rank.find(a->mName.C_Str());
                    const auto rb = rank.find(b->mName.C_Str());
                    const int ia = (ra != rank.end()) ? ra->second : std::numeric_limits<int>::max();
                    const int ib = (rb != rank.end()) ? rb->second : std::numeric_limits<int>::max();
                    return ia < ib;
                });
    }

    if (totalPadded > 0) {
        std::fprintf(stderr, "note: added %u zero-weight placeholder bone(s) across meshes so every mesh's skin covers the full skeleton\n", totalPadded);
    }
}

void PrintUsage(const char *argv0) {
    std::fprintf(stderr,
            "usage: %s <output-file> <model-file> <anim-file>... [options]\n\n"
            "Imports <model-file> (mesh + skeleton) and one or more <anim-file>s\n"
            "(animation, sharing the same node/bone names as <model-file>), grafts all\n"
            "of their animations onto <model-file>'s scene, and exports the combined\n"
            "result to <output-file>.\n\n"
            "  --format <id>     Explicit assimp export format id (see `assimp listext`).\n"
            "                    If omitted, guessed from <output-file>'s extension.\n"
            "  --anim-index <n>  Merge only animation index n from the anim-file most\n"
            "                    recently named, instead of all of its animations.\n"
            "                    Repeatable (once per anim-file that needs it).\n"
            "  --fps <value>     Override mTicksPerSecond on the animation(s) merged from\n"
            "                    the anim-file most recently named, without touching\n"
            "                    keyframe times -- use when the source format's reported\n"
            "                    rate doesn't match how the target engine will play it\n"
            "                    back (e.g. BFRES exports that come out wrong in glTF).\n"
            "  --fps-scale <f>   Multiply mTicksPerSecond by <f> instead of setting it\n"
            "                    outright, for the anim-file most recently named. Mutually\n"
            "                    exclusive with --fps for the same anim-file.\n"
            "  --no-sort-bones-topologically\n"
            "                    Skip the fixup that reorders each mesh's bones into\n"
            "                    parent-before-child order (on by default; see\n"
            "                    SortBonesTopologically in the source for why).\n"
            "  --unify-skins     Pad every mesh's skin out to the full skeleton (union of\n"
            "                    all bones referenced anywhere in the scene) instead of\n"
            "                    just the bones that particular mesh is weighted to, so\n"
            "                    bone/joint indices agree across skins. Off by default.\n"
            "  --name-anims-from-file\n"
            "                    Rename each merged animation after its anim-file's\n"
            "                    basename (e.g. animActionBlowOff.Anim.pod -> \n"
            "                    \"animActionBlowOff\"), instead of keeping whatever name\n"
            "                    the importer gave it. Use this for formats where every\n"
            "                    animation comes back with the same generic name (e.g.\n"
            "                    PODImporter always names them \"Default Animation\").\n",
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

// Strips directory components and everything from the first '.' onward, so
// e.g. "dump/animActionBlowOff.Anim.pod" becomes "animActionBlowOff". Some
// importers (PODImporter among them) name every aiAnimation the same generic
// string (e.g. "Default Animation") regardless of source file, which makes
// merged animations indistinguishable by name; the anim-file's own basename
// is normally the actual intended animation name in that case.
std::string BaseNameWithoutExtension(const std::string &path) {
    const size_t slash = path.find_last_of("/\\");
    const std::string filename = (slash == std::string::npos) ? path : path.substr(slash + 1);
    const size_t dot = filename.find('.');
    return (dot == std::string::npos) ? filename : filename.substr(0, dot);
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 4) {
        PrintUsage(argv[0]);
        return 1;
    }

    const std::string outputPath = argv[1];
    const std::string modelPath = argv[2];
    std::string formatId;
    bool sortBonesTopologically = true;
    bool unifySkins = false;
    bool nameAnimsFromFile = false;

    // One entry per positional anim-file argument. --anim-index/--fps/
    // --fps-scale each apply to whichever anim-file was most recently named,
    // mirroring how flags read left-to-right, and are repeatable (once per
    // anim-file that needs one).
    struct AnimSource {
        std::string path;
        int animIndex = -1;
        double fpsOverride = -1.0;
        double fpsScale = -1.0;
    };
    std::vector<AnimSource> animSources;

    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            formatId = argv[++i];
        } else if (std::strcmp(argv[i], "--anim-index") == 0 && i + 1 < argc) {
            if (animSources.empty()) {
                std::fprintf(stderr, "error: --anim-index must follow an anim-file argument\n");
                return 1;
            }
            animSources.back().animIndex = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--fps") == 0 && i + 1 < argc) {
            if (animSources.empty()) {
                std::fprintf(stderr, "error: --fps must follow an anim-file argument\n");
                return 1;
            }
            animSources.back().fpsOverride = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--fps-scale") == 0 && i + 1 < argc) {
            if (animSources.empty()) {
                std::fprintf(stderr, "error: --fps-scale must follow an anim-file argument\n");
                return 1;
            }
            animSources.back().fpsScale = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--no-sort-bones-topologically") == 0) {
            sortBonesTopologically = false;
        } else if (std::strcmp(argv[i], "--unify-skins") == 0) {
            unifySkins = true;
        } else if (std::strcmp(argv[i], "--name-anims-from-file") == 0) {
            nameAnimsFromFile = true;
        } else if (std::strncmp(argv[i], "--", 2) == 0) {
            std::fprintf(stderr, "unrecognized argument: %s\n", argv[i]);
            PrintUsage(argv[0]);
            return 1;
        } else {
            animSources.push_back({argv[i], -1, -1.0, -1.0});
        }
    }

    if (animSources.empty()) {
        std::fprintf(stderr, "error: at least one <anim-file> is required\n");
        PrintUsage(argv[0]);
        return 1;
    }

    if (formatId.empty()) {
        formatId = GuessExportFormatId(outputPath);
        if (formatId.empty()) {
            std::fprintf(stderr, "error: could not guess export format from '%s'; pass --format explicitly\n", outputPath.c_str());
            return 1;
        }
    }

    Assimp::Importer modelImporter;
    // Collada nodes default to their (often prefixed, e.g. "Armature_chest") id; use the
    // human-readable name so bones match the other formats' channel names.
    modelImporter.SetPropertyInteger(AI_CONFIG_IMPORT_COLLADA_USE_COLLADA_NAMES, 1);
    const aiScene *constModelScene = modelImporter.ReadFile(modelPath, 0);
    if (!constModelScene) {
        std::fprintf(stderr, "error: failed to import '%s': %s\n", modelPath.c_str(), modelImporter.GetErrorString());
        return 1;
    }
    aiScene *modelScene = const_cast<aiScene *>(constModelScene);

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

    std::vector<aiAnimation *> merged(modelScene->mAnimations, modelScene->mAnimations + modelScene->mNumAnimations);

    for (const AnimSource &source : animSources) {
        Assimp::Importer animImporter;
        animImporter.SetPropertyInteger(AI_CONFIG_IMPORT_COLLADA_USE_COLLADA_NAMES, 1);
        if (!animImporter.ReadFile(source.path, 0)) {
            std::fprintf(stderr, "warning: skipping '%s': failed to import: %s\n", source.path.c_str(), animImporter.GetErrorString());
            continue;
        }
        // Detach the anim scene from its Importer so we can transfer ownership
        // of individual aiAnimation objects into modelScene without a double
        // free when animImporter goes out of scope.
        aiScene *animScene = animImporter.GetOrphanedScene();

        // A single anim-file with no animations (or an out-of-range
        // --anim-index) is common when batch-merging many files -- e.g. one
        // turns out to be a camera-only or otherwise empty export -- and
        // shouldn't abort the whole run over the rest of the sources.
        if (animScene->mNumAnimations == 0) {
            std::fprintf(stderr, "warning: skipping '%s': has no animations to merge\n", source.path.c_str());
            delete animScene;
            continue;
        }
        if (source.animIndex >= 0 && static_cast<unsigned int>(source.animIndex) >= animScene->mNumAnimations) {
            std::fprintf(stderr, "warning: skipping '%s': --anim-index %d out of range (file has %u animation(s))\n",
                    source.path.c_str(), source.animIndex, animScene->mNumAnimations);
            delete animScene;
            continue;
        }

        for (unsigned int i = 0; i < animScene->mNumAnimations; ++i) {
            if (source.animIndex >= 0 && static_cast<unsigned int>(source.animIndex) != i) continue;

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

            if (nameAnimsFromFile) {
                std::string name = BaseNameWithoutExtension(source.path);
                // Only this source's own anims are kept when --anim-index is
                // set, so the basename alone is unambiguous; otherwise merging
                // several animations out of one file needs a suffix so they
                // don't all collide on the same name.
                if (source.animIndex < 0 && animScene->mNumAnimations > 1) {
                    name += "_" + std::to_string(i);
                }
                anim->mName = name.c_str();
            }

            // Keyframe mTime values are in ticks, scaled to seconds by
            // mTicksPerSecond; overriding/scaling it here changes playback
            // speed without touching the keyframes themselves, which is the
            // right knob when the source format's own reported rate doesn't
            // match how the target engine ends up interpreting it.
            if (source.fpsOverride >= 0.0) {
                anim->mTicksPerSecond = source.fpsOverride;
            } else if (source.fpsScale >= 0.0) {
                anim->mTicksPerSecond *= source.fpsScale;
            }

            merged.push_back(anim);
            animScene->mAnimations[i] = nullptr; // ownership transferred; keep animScene's destructor from freeing it
        }

        delete animScene; // safe: transferred aiAnimation slots were nulled out above
    }

    delete[] modelScene->mAnimations;
    modelScene->mAnimations = new aiAnimation *[merged.size()];
    std::copy(merged.begin(), merged.end(), modelScene->mAnimations);
    modelScene->mNumAnimations = static_cast<unsigned int>(merged.size());

    if (unifySkins) UnifySkinsWithSkeleton(modelScene);
    if (sortBonesTopologically) SortBonesTopologically(modelScene);

    Assimp::Exporter exporter;
    const aiReturn result = exporter.Export(modelScene, formatId.c_str(), outputPath.c_str(), 0);
    if (result != aiReturn_SUCCESS) {
        std::fprintf(stderr, "error: export failed: %s\n", exporter.GetErrorString());
        return 1;
    }

    std::fprintf(stderr, "wrote '%s' (%u animation(s), format '%s')\n", outputPath.c_str(), modelScene->mNumAnimations, formatId.c_str());
    return 0;
}
