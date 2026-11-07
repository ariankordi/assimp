/*
---------------------------------------------------------------------------
Open Asset Import Library (assimp)
---------------------------------------------------------------------------

Copyright (c) 2006-2026, assimp team

All rights reserved.

Redistribution and use of this software in source and binary forms,
with or without modification, are permitted provided that the following
conditions are met:

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
---------------------------------------------------------------------------
*/
#include "UnitTestPCH.h"

#include <assimp/config.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/Importer.hpp>

#include <cmath>
#include <string>

using namespace Assimp;

namespace {

/// Header shared by the test files: 60 fps (ntscf), degrees, frames 0..10.
const std::string cHeader =
        "animVersion 1.1;\n"
        "mayaVersion 2015;\n"
        "timeUnit ntscf;\n"
        "linearUnit cm;\n"
        "angularUnit deg;\n"
        "startTime 0;\n"
        "endTime 10;\n";

/// Builds one "anim ... animData { ... }" block for attribute <leaf> of
/// <node>, with the given key rows (each without its trailing ';').
std::string Curve(const std::string &node, const std::string &group, const std::string &leaf, const std::string &rows) {
    return "anim " + group + "." + leaf + " " + leaf + " " + node + " 0 0 0;\n"
                                                                    "animData {\n"
                                                                    "  input time;\n"
                                                                    "  output " +
           std::string(group == "rotate" ? "angular" : "linear") + ";\n"
                                                                   "  weighted 0;\n"
                                                                   "  preInfinity constant;\n"
                                                                   "  postInfinity constant;\n"
                                                                   "  keys {\n" +
           rows + "  }\n"
                  "}\n";
}

/// Imports .anim text from memory, optionally with a reference scene.
const aiScene *ImportAnim(Importer &importer, const std::string &text, const aiScene *reference = nullptr) {
    if (reference) {
        importer.SetPropertyPointer(AI_CONFIG_IMPORT_MAYA_ANIM_REFERENCE_SCENE, const_cast<aiScene *>(reference));
    }
    return importer.ReadFileFromMemory(text.data(), text.size(), aiProcess_ValidateDataStructure, "anim");
}

/// Finds the channel animating the named node, or nullptr.
const aiNodeAnim *FindChannel(const aiScene *scene, const char *name) {
    const aiAnimation *anim = scene->mAnimations[0];
    for (unsigned int i = 0; i < anim->mNumChannels; ++i) {
        if (anim->mChannels[i]->mNodeName == aiString(name)) return anim->mChannels[i];
    }
    return nullptr;
}

/// Rotation matrix for Maya's "xyz" order (X first): Rz * Ry * Rx, angles in degrees.
aiMatrix3x3 RotationXyz(float xDeg, float yDeg, float zDeg) {
    constexpr float cDegToRad = 3.14159265358979f / 180.0f;
    aiMatrix4x4 rx, ry, rz;
    aiMatrix4x4::RotationX(xDeg * cDegToRad, rx);
    aiMatrix4x4::RotationY(yDeg * cDegToRad, ry);
    aiMatrix4x4::RotationZ(zDeg * cDegToRad, rz);
    return aiMatrix3x3(rz * ry * rx);
}

/// True when two quaternions describe the same rotation (q and -q included).
bool SameRotation(const aiQuaternion &a, const aiQuaternion &b) {
    const float dot = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
    return std::fabs(std::fabs(dot) - 1.0f) < 1e-4f;
}

} // namespace

TEST(utMayaAnimImporter, readsHeaderAndNamesNodes) {
    const std::string text = cHeader + Curve("joint1", "translate", "translateX", "    0 5 linear linear 1 1 0;\n");
    Importer importer;
    const aiScene *scene = ImportAnim(importer, text);
    ASSERT_NE(nullptr, scene);
    ASSERT_EQ(1u, scene->mNumAnimations);

    const aiAnimation *anim = scene->mAnimations[0];
    EXPECT_DOUBLE_EQ(60.0, anim->mTicksPerSecond); // ntscf is 60 fps, not 30.
    EXPECT_DOUBLE_EQ(10.0, anim->mDuration);
    ASSERT_EQ(1u, anim->mNumChannels);
    EXPECT_STREQ("joint1", anim->mChannels[0]->mNodeName.C_Str());
    ASSERT_NE(nullptr, scene->mRootNode->FindNode("joint1"));
}

TEST(utMayaAnimImporter, acceptsCrlfAndRejectsMissingVersion) {
    std::string text = cHeader + Curve("joint1", "translate", "translateX", "    0 5 linear linear 1 1 0;\n");
    std::string crlf;
    for (char c : text) {
        if (c == '\n') crlf += '\r';
        crlf += c;
    }
    Importer crlfImporter;
    EXPECT_NE(nullptr, ImportAnim(crlfImporter, crlf));

    Importer noVersionImporter;
    EXPECT_EQ(nullptr, ImportAnim(noVersionImporter, text.substr(text.find('\n') + 1)));
}

TEST(utMayaAnimImporter, constantCurveBakesToRangeEnds) {
    const std::string text = cHeader + Curve("joint1", "translate", "translateX", "    0 5 fixed fixed 1 1 0 0 1 0 1;\n");
    Importer importer;
    const aiScene *scene = ImportAnim(importer, text);
    ASSERT_NE(nullptr, scene);
    const aiNodeAnim *channel = FindChannel(scene, "joint1");
    ASSERT_NE(nullptr, channel);
    ASSERT_EQ(2u, channel->mNumPositionKeys);
    EXPECT_DOUBLE_EQ(0.0, channel->mPositionKeys[0].mTime);
    EXPECT_DOUBLE_EQ(10.0, channel->mPositionKeys[1].mTime);
    EXPECT_FLOAT_EQ(5.0f, channel->mPositionKeys[1].mValue.x);
}

TEST(utMayaAnimImporter, linearTangentsInterpolateLinearly) {
    const std::string rows =
            "    0 0 linear linear 1 1 0;\n"
            "    10 20 linear linear 1 1 0;\n";
    Importer importer;
    const aiScene *scene = ImportAnim(importer, cHeader + Curve("joint1", "translate", "translateY", rows));
    ASSERT_NE(nullptr, scene);
    const aiNodeAnim *channel = FindChannel(scene, "joint1");
    ASSERT_NE(nullptr, channel);
    ASSERT_EQ(11u, channel->mNumPositionKeys); // One key per frame, 0..10.
    EXPECT_FLOAT_EQ(10.0f, channel->mPositionKeys[5].mValue.y);
    EXPECT_FLOAT_EQ(0.0f, channel->mPositionKeys[5].mValue.x); // Unkeyed axis defaults to 0.
}

TEST(utMayaAnimImporter, stepTangentHoldsValue) {
    const std::string rows =
            "    0 1 step step 1 1 0;\n"
            "    10 3 step step 1 1 0;\n";
    Importer importer;
    const aiScene *scene = ImportAnim(importer, cHeader + Curve("joint1", "scale", "scaleZ", rows));
    ASSERT_NE(nullptr, scene);
    const aiNodeAnim *channel = FindChannel(scene, "joint1");
    ASSERT_NE(nullptr, channel);
    ASSERT_EQ(11u, channel->mNumScalingKeys);
    EXPECT_FLOAT_EQ(1.0f, channel->mScalingKeys[9].mValue.z);
    EXPECT_FLOAT_EQ(3.0f, channel->mScalingKeys[10].mValue.z);
    EXPECT_FLOAT_EQ(1.0f, channel->mScalingKeys[9].mValue.x); // Unkeyed scale defaults to 1.
}

TEST(utMayaAnimImporter, fixedTangentsUsePerFrameSlope) {
    // 45 degree tangents (slope 1 per frame) on both ends of a flat-valued
    // segment: the Hermite curve rises then falls, peaking above 0 mid-way.
    const std::string rows =
            "    0 0 fixed fixed 1 1 0 45 1 45 1;\n"
            "    10 0 fixed fixed 1 1 0 -45 1 -45 1;\n";
    Importer importer;
    const aiScene *scene = ImportAnim(importer, cHeader + Curve("joint1", "translate", "translateX", rows));
    ASSERT_NE(nullptr, scene);
    const aiNodeAnim *channel = FindChannel(scene, "joint1");
    ASSERT_NE(nullptr, channel);
    ASSERT_EQ(11u, channel->mNumPositionKeys);
    // h10(0.5) * 10 * 1 + h11(0.5) * 10 * -1 = 1.25 + 1.25.
    EXPECT_NEAR(2.5f, channel->mPositionKeys[5].mValue.x, 1e-4f);
}

TEST(utMayaAnimImporter, rotationUsesXyzOrder) {
    const std::string text = cHeader +
                             Curve("joint1", "rotate", "rotateX", "    0 30 linear linear 1 1 0;\n") +
                             Curve("joint1", "rotate", "rotateY", "    0 40 linear linear 1 1 0;\n") +
                             Curve("joint1", "rotate", "rotateZ", "    0 50 linear linear 1 1 0;\n");
    Importer importer;
    const aiScene *scene = ImportAnim(importer, text);
    ASSERT_NE(nullptr, scene);
    const aiNodeAnim *channel = FindChannel(scene, "joint1");
    ASSERT_NE(nullptr, channel);
    ASSERT_GE(channel->mNumRotationKeys, 1u);
    EXPECT_TRUE(SameRotation(aiQuaternion(RotationXyz(30, 40, 50)), channel->mRotationKeys[0].mValue));
}

TEST(utMayaAnimImporter, referenceSceneFillsUnkeyedAxes) {
    // Bind pose: translated, rotated 30 degrees about X.
    aiScene reference;
    reference.mRootNode = new aiNode("root");
    aiNode *joint = new aiNode("joint1");
    joint->mParent = reference.mRootNode;
    reference.mRootNode->addChildren(1, &joint);
    aiMatrix4x4 rotation(RotationXyz(30, 0, 0));
    aiMatrix4x4 translation;
    aiMatrix4x4::Translation(aiVector3D(1, 2, 3), translation);
    joint->mTransformation = translation * rotation;

    // Keys only translateX and rotateZ; the other axes must come from the bind pose.
    const std::string text = cHeader +
                             Curve("joint1", "translate", "translateX", "    0 7 linear linear 1 1 0;\n") +
                             Curve("joint1", "rotate", "rotateZ", "    0 90 linear linear 1 1 0;\n");
    Importer importer;
    const aiScene *scene = ImportAnim(importer, text, &reference);
    ASSERT_NE(nullptr, scene);
    const aiNodeAnim *channel = FindChannel(scene, "joint1");
    ASSERT_NE(nullptr, channel);
    ASSERT_GE(channel->mNumPositionKeys, 1u);
    EXPECT_NEAR(7.0f, channel->mPositionKeys[0].mValue.x, 1e-4f);
    EXPECT_NEAR(2.0f, channel->mPositionKeys[0].mValue.y, 1e-4f);
    EXPECT_NEAR(3.0f, channel->mPositionKeys[0].mValue.z, 1e-4f);
    ASSERT_GE(channel->mNumRotationKeys, 1u);
    EXPECT_TRUE(SameRotation(aiQuaternion(RotationXyz(30, 0, 90)), channel->mRotationKeys[0].mValue));
}

TEST(utMayaAnimImporter, referenceSceneMatchesKeyedEulerBranch) {
    // Rz(-90) is also (180, 180, 90) in XYZ order. A file that keys only
    // rotateY = -180 was authored on the second branch; taking X/Z from the
    // first one, (0, 0, -90), would flip the joint 180 degrees.
    aiScene reference;
    reference.mRootNode = new aiNode("root");
    aiNode *joint = new aiNode("ankle_r");
    joint->mParent = reference.mRootNode;
    reference.mRootNode->addChildren(1, &joint);
    joint->mTransformation = aiMatrix4x4(RotationXyz(0, 0, -90));

    const std::string text = cHeader + Curve("ankle_r", "rotate", "rotateY", "    0 -180 fixed fixed 1 1 0 0 1 0 1;\n");
    Importer importer;
    const aiScene *scene = ImportAnim(importer, text, &reference);
    ASSERT_NE(nullptr, scene);
    const aiNodeAnim *channel = FindChannel(scene, "ankle_r");
    ASSERT_NE(nullptr, channel);
    ASSERT_GE(channel->mNumRotationKeys, 1u);
    EXPECT_TRUE(SameRotation(aiQuaternion(RotationXyz(0, 0, -90)), channel->mRotationKeys[0].mValue));
}

TEST(utMayaAnimImporter, strayBracesTerminate) {
    // A '{' inside keys and a '}' at top level used to leave the parser
    // looping without consuming a token.
    const std::string rows =
            "    0 1 linear linear 1 1 0;\n"
            "    { junk }\n"
            "    10 2 linear linear 1 1 0;\n";
    const std::string text = cHeader + "}\n" + Curve("joint1", "translate", "translateX", rows);
    Importer importer;
    const aiScene *scene = ImportAnim(importer, text);
    ASSERT_NE(nullptr, scene);
    const aiNodeAnim *channel = FindChannel(scene, "joint1");
    ASSERT_NE(nullptr, channel);
    EXPECT_FLOAT_EQ(2.0f, channel->mPositionKeys[channel->mNumPositionKeys - 1].mValue.x);
}

TEST(utMayaAnimImporter, rejectsHugeOrNonFiniteTimeRange) {
    const std::string rows =
            "    0 0 linear linear 1 1 0;\n"
            "    10 1 linear linear 1 1 0;\n";
    const std::string curve = Curve("joint1", "translate", "translateX", rows);
    const std::string prefix = "animVersion 1.1;\ntimeUnit ntscf;\nstartTime 0;\n";

    Importer hugeImporter;
    EXPECT_EQ(nullptr, ImportAnim(hugeImporter, prefix + "endTime 1e18;\n" + curve));

    Importer infImporter;
    EXPECT_EQ(nullptr, ImportAnim(infImporter, prefix + "endTime inf;\n" + curve));

    Importer nanImporter;
    EXPECT_EQ(nullptr, ImportAnim(nanImporter, prefix + "endTime nan;\n" + curve));
}
