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

// Importer for Maya ASCII animation curve files (.anim).
//
// Written from scratch against Autodesk's format description:
//   "Anim File Format", Maya 2009 help,
//   https://download.autodesk.com/us/maya/2009help/files/Animation_curves_Anim_File_Format.htm
//
// The following were consulted for how real-world writers fill the format in
// (no code was copied from either; both are (L)GPL-licensed):
//   - BrawlCrate, BrawlLib/Wii/Animations/AnimFormat.cs (LGPL-3.0), the writer
//     of the Wii-era files this importer was built for:
//     https://github.com/soopercool101/BrawlCrate
//     Taken from it: fixed tangent angles are stored as atan(slope) with the
//     slope in value units per frame, and Euler order is XYZ.
//   - Blender_io-scene-ANIM, parse_anim.py / import_anim.py / anim_utils.py
//     (GPL-3.0): https://github.com/luvvyamy/Blender_io-scene-ANIM
//     Taken from it: the key-row field layout (fixed tangents append an
//     angle/weight pair each), and the timeUnit -> fps table.

#ifndef ASSIMP_BUILD_NO_MAYAANIM_IMPORTER

#include "MayaAnimImporter.h"

#include <assimp/commonMetaData.h>
#include <assimp/config.h>
#include <assimp/fast_atof.h>
#include <assimp/importerdesc.h>
#include <assimp/scene.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/IOSystem.hpp>
#include <assimp/Importer.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Assimp {

namespace {

const aiImporterDesc cDesc = {
    "Maya ASCII Animation Curve Importer",
    "",
    "",
    "Animation curves only; bakes to per-frame keys",
    aiImporterFlags_SupportTextFlavour,
    0,
    0,
    0,
    0,
    "anim"
};

constexpr double cPi = 3.14159265358979323846;

/// Upper bound on baked keys across all channels (about 240 MB of aiQuatKey),
/// so a hostile startTime/endTime can't make the bake exhaust memory. Real
/// clips are a few thousand frames times a few hundred channels.
constexpr double cMaxBakedSamples = 1e7;
constexpr double cDegToRad = cPi / 180.0;

/// Number of transform attributes the importer understands (3 groups x 3 axes).
constexpr size_t cNumAttributes = 9;

/// Attribute groups, in the order their curves are stored per node.
enum AttributeGroup : size_t {
    Group_Translate = 0, ///< translateX/Y/Z, indices 0..2.
    Group_Rotate = 1, ///< rotateX/Y/Z, indices 3..5.
    Group_Scale = 2 ///< scaleX/Y/Z, indices 6..8.
};

/// Maya keyframe tangent types. Only Fixed carries explicit angle/weight data.
enum class TangentType {
    Spline,
    Linear,
    Fast,
    Slow,
    Flat,
    Step,
    StepNext,
    Fixed,
    Clamped,
    Plateau,
    Auto
};

/// How a curve continues before its first / after its last key.
enum class Infinity {
    Constant,
    Linear,
    Cycle,
    CycleRelative,
    Oscillate
};

/// One parsed keyframe. Slopes are filled in by FinalizeCurve() after parsing.
struct Key {
    double time = 0.0; ///< In frames of the file's timeUnit.
    double value = 0.0; ///< In the file's units (degrees or radians for rotation).
    TangentType inType = TangentType::Spline;
    TangentType outType = TangentType::Spline;
    std::optional<double> inAngle; ///< Radians; only present for fixed tangents.
    std::optional<double> outAngle; ///< Radians; only present for fixed tangents.
    double inSlope = 0.0; ///< Value units per frame, arriving at this key.
    double outSlope = 0.0; ///< Value units per frame, leaving this key.
};

/// One animated attribute (e.g. rotateX of node "chest").
struct Curve {
    std::vector<Key> keys; ///< Sorted by time once finalized.
    Infinity preInfinity = Infinity::Constant;
    Infinity postInfinity = Infinity::Constant;
};

/// All curves that target one node, indexed by attribute (see AttributeGroup).
struct NodeCurves {
    std::string name;
    std::array<std::optional<Curve>, cNumAttributes> curves;
};

/// Header values plus every node's curves, in order of first appearance.
struct AnimFile {
    std::string animVersion;
    std::string mayaVersion;
    double framesPerSecond = 24.0; ///< Maya's default timeUnit is "film".
    double angularToRadians = cDegToRad; ///< Maya's default angularUnit is "deg".
    std::optional<double> startTime;
    std::optional<double> endTime;
    std::vector<NodeCurves> nodes;
};

// ------------------------------------------------------------------------------------------------
// Tokenizing.

/// Splits the file into whitespace-separated words, with ';', '{' and '}'
/// always standing as tokens of their own so statements and blocks can be
/// found regardless of line breaks (BrawlCrate writes CRLF, Maya LF).
std::vector<std::string_view> Tokenize(std::string_view text) {
    std::vector<std::string_view> tokens;
    size_t i = 0;
    while (i < text.size()) {
        const char c = text[i];
        if (std::isspace(static_cast<unsigned char>(c)) != 0) {
            ++i;
        } else if (c == ';' || c == '{' || c == '}') {
            tokens.push_back(text.substr(i, 1));
            ++i;
        } else {
            const size_t start = i;
            while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])) == 0 &&
                    text[i] != ';' && text[i] != '{' && text[i] != '}') {
                ++i;
            }
            tokens.push_back(text.substr(start, i - start));
        }
    }
    return tokens;
}

/// Sequential reader over the token list.
class TokenCursor {
public:
    explicit TokenCursor(std::vector<std::string_view> tokens) :
            mTokens(std::move(tokens)) {}

    bool AtEnd() const { return mPos >= mTokens.size(); }
    std::string_view Peek() const { return AtEnd() ? std::string_view() : mTokens[mPos]; }
    std::string_view Next() { return AtEnd() ? std::string_view() : mTokens[mPos++]; }

    /// Reads tokens up to the next ';' (consumed), or '{' / '}' (left in
    /// place), whichever comes first, so the caller can tell a plain
    /// statement from a block opener, and a missing ';' can't swallow the
    /// enclosing block's closer. Callers must handle a '{' or '}' left in
    /// front, since this then returns without advancing.
    std::vector<std::string_view> ReadStatement() {
        std::vector<std::string_view> statement;
        while (!AtEnd()) {
            const std::string_view token = Peek();
            if (token == "{" || token == "}") break;
            ++mPos;
            if (token == ";") break;
            statement.push_back(token);
        }
        return statement;
    }

    /// Skips a balanced { ... } block; the cursor must be on its '{'.
    void SkipBlock() {
        int depth = 0;
        while (!AtEnd()) {
            const std::string_view token = Next();
            if (token == "{") {
                ++depth;
            } else if (token == "}" && --depth == 0) {
                return;
            }
        }
    }

private:
    std::vector<std::string_view> mTokens;
    size_t mPos = 0;
};

// ------------------------------------------------------------------------------------------------
// Value parsing.

/// Locale-independent number parse; assimp's fast_atof avoids strtod's
/// dependence on the C locale's decimal separator. Rejects inf/nan, which no
/// valid file contains and which would break key sorting and frame counts.
double ParseNumber(std::string_view token) {
    const std::string copy(token); // fast_atoreal_move needs a terminated string.
    double value = 0.0;
    fast_atoreal_move(copy.c_str(), value);
    if (!std::isfinite(value)) {
        throw DeadlyImportError("Maya ANIM: non-finite number '", copy, "'");
    }
    return value;
}

/// Returns the multiplier from the named angular unit to radians.
double AngularUnitToRadians(std::string_view unit) {
    if (unit == "rad" || unit == "radian" || unit == "radians") return 1.0;
    if (unit != "deg" && unit != "degree" && unit != "degrees") {
        ASSIMP_LOG_WARN("Maya ANIM: unknown angular unit '", std::string(unit), "', assuming degrees");
    }
    return cDegToRad;
}

/// Converts a Maya timeUnit name to frames per second. Maya names video
/// rates after their broadcast standard, and the "f" suffix means "fields",
/// i.e. double rate: ntsc = 30, ntscf = 60.
double TimeUnitToFps(std::string_view unit) {
    static const std::map<std::string_view, double> cNamedRates = {
        { "game", 15.0 }, { "film", 24.0 }, { "pal", 25.0 }, { "ntsc", 30.0 },
        { "show", 48.0 }, { "palf", 50.0 }, { "ntscf", 60.0 },
        { "hour", 1.0 / 3600.0 }, { "min", 1.0 / 60.0 }, { "sec", 1.0 }, { "millisec", 1000.0 }
    };
    if (const auto it = cNamedRates.find(unit); it != cNamedRates.end()) {
        return it->second;
    }

    // Newer Maya versions also write arbitrary rates such as "120fps" or "23.976fps".
    constexpr std::string_view cFpsSuffix = "fps";
    if (unit.size() > cFpsSuffix.size() && unit.substr(unit.size() - cFpsSuffix.size()) == cFpsSuffix) {
        const double fps = ParseNumber(unit.substr(0, unit.size() - cFpsSuffix.size()));
        if (fps > 0.0) return fps;
    }

    ASSIMP_LOG_WARN("Maya ANIM: unknown time unit '", std::string(unit), "', assuming 24 fps (film)");
    return 24.0;
}

TangentType ParseTangentType(std::string_view name) {
    static const std::map<std::string_view, TangentType> cTypes = {
        { "spline", TangentType::Spline }, { "linear", TangentType::Linear },
        { "fast", TangentType::Fast }, { "slow", TangentType::Slow },
        { "flat", TangentType::Flat }, { "step", TangentType::Step },
        { "stepnext", TangentType::StepNext }, { "fixed", TangentType::Fixed },
        { "clamped", TangentType::Clamped }, { "plateau", TangentType::Plateau },
        { "auto", TangentType::Auto }
    };
    const auto it = cTypes.find(name);
    return it != cTypes.end() ? it->second : TangentType::Spline;
}

Infinity ParseInfinity(std::string_view name) {
    if (name == "linear") return Infinity::Linear;
    if (name == "cycle") return Infinity::Cycle;
    if (name == "cycleRelative") return Infinity::CycleRelative;
    if (name == "oscillate") return Infinity::Oscillate;
    return Infinity::Constant;
}

/// Maps a leaf attribute name to its index in NodeCurves::curves, or
/// nothing for attributes this importer doesn't represent (visibility,
/// custom attributes, ...).
std::optional<size_t> AttributeIndex(std::string_view leafName) {
    static const std::map<std::string_view, size_t> cIndices = {
        { "translateX", 0 }, { "translateY", 1 }, { "translateZ", 2 },
        { "rotateX", 3 }, { "rotateY", 4 }, { "rotateZ", 5 },
        { "scaleX", 6 }, { "scaleY", 7 }, { "scaleZ", 8 }
    };
    const auto it = cIndices.find(leafName);
    return it != cIndices.end() ? std::optional<size_t>(it->second) : std::nullopt;
}

// ------------------------------------------------------------------------------------------------
// Parsing.

/// Parses one key row: "time value [inType outType locks... [inAngle inWeight] [outAngle outWeight]]".
/// The number of lock/breakdown integers differs between animVersion 1.0 and
/// 1.1, so the angle positions are derived from the token count instead.
Key ParseKeyRow(const std::vector<std::string_view> &row, double tangentAngleToRadians) {
    Key key;
    key.time = ParseNumber(row[0]);
    key.value = ParseNumber(row[1]);
    if (row.size() < 4) return key;

    key.inType = ParseTangentType(row[2]);
    key.outType = ParseTangentType(row[3]);

    const size_t inFixedFields = key.inType == TangentType::Fixed ? 2 : 0;
    const size_t outFixedFields = key.outType == TangentType::Fixed ? 2 : 0;
    const size_t trailing = row.size() - 4;
    if (trailing < inFixedFields + outFixedFields) {
        // A fixed tangent without its angle is malformed; fall back to a smooth tangent.
        ASSIMP_LOG_WARN("Maya ANIM: key at time ", key.time, " has a fixed tangent without angle data");
        if (inFixedFields) key.inType = TangentType::Spline;
        if (outFixedFields) key.outType = TangentType::Spline;
        return key;
    }

    size_t angleIndex = 4 + trailing - inFixedFields - outFixedFields;
    if (inFixedFields) {
        key.inAngle = ParseNumber(row[angleIndex]) * tangentAngleToRadians;
        angleIndex += 2; // Skip the weight; weighted curves are approximated (see ParseAnimData).
    }
    if (outFixedFields) {
        key.outAngle = ParseNumber(row[angleIndex]) * tangentAngleToRadians;
    }
    return key;
}

/// Parses an "animData { ... }" block; the cursor must be on its '{'. Returns
/// nothing for curves that aren't driven by time (set-driven keys), since
/// those can't be expressed as an aiAnimation.
std::optional<Curve> ParseAnimData(TokenCursor &cursor, double headerAngularToRadians) {
    Curve curve;
    bool timeDriven = true;
    double tangentAngleToRadians = headerAngularToRadians;

    cursor.Next(); // '{'
    while (!cursor.AtEnd() && cursor.Peek() != "}") {
        const std::vector<std::string_view> statement = cursor.ReadStatement();
        if (cursor.Peek() == "{") {
            if (!statement.empty() && statement[0] == "keys") {
                cursor.Next(); // '{'
                while (!cursor.AtEnd() && cursor.Peek() != "}") {
                    const std::vector<std::string_view> row = cursor.ReadStatement();
                    // ReadStatement stops in front of '{' without consuming
                    // it; skip the stray block so the loop always advances.
                    if (cursor.Peek() == "{") {
                        cursor.SkipBlock();
                        continue;
                    }
                    if (row.size() >= 2) {
                        curve.keys.push_back(ParseKeyRow(row, tangentAngleToRadians));
                    }
                }
                cursor.Next(); // '}'
            } else {
                cursor.SkipBlock();
            }
            continue;
        }
        if (statement.size() < 2) continue;

        const std::string_view name = statement[0];
        const std::string_view value = statement[1];
        if (name == "input") {
            timeDriven = (value == "time");
        } else if (name == "tangentAngleUnit") {
            tangentAngleToRadians = AngularUnitToRadians(value);
        } else if (name == "preInfinity") {
            curve.preInfinity = ParseInfinity(value);
        } else if (name == "postInfinity") {
            curve.postInfinity = ParseInfinity(value);
        } else if (name == "weighted" && value != "0") {
            // Weighted tangents make each segment a 2D Bezier whose handle
            // lengths also bend time; evaluating as Hermite on the angles
            // alone keeps the shape's slopes but not its exact timing.
            ASSIMP_LOG_WARN("Maya ANIM: weighted tangents are approximated as unweighted");
        }
    }
    cursor.Next(); // '}'

    if (!timeDriven) return std::nullopt;
    return curve;
}

/// Parses the whole token stream into header values and per-node curves.
AnimFile ParseAnimFile(TokenCursor &cursor) {
    AnimFile file;
    std::map<std::string, size_t> nodeIndexByName;

    // Target of the most recent "anim" statement; an animData block belongs to it.
    std::string pendingNode;
    std::optional<size_t> pendingAttribute;

    while (!cursor.AtEnd()) {
        const std::vector<std::string_view> statement = cursor.ReadStatement();
        if (cursor.Peek() == "}") {
            cursor.Next(); // Stray closer with no open block; skip it so the loop advances.
            continue;
        }
        if (cursor.Peek() == "{") {
            if (!statement.empty() && statement[0] == "animData" && pendingAttribute) {
                std::optional<Curve> curve = ParseAnimData(cursor, file.angularToRadians);
                if (curve && !curve->keys.empty()) {
                    auto [it, inserted] = nodeIndexByName.emplace(pendingNode, file.nodes.size());
                    if (inserted) file.nodes.push_back(NodeCurves{ pendingNode, {} });
                    std::optional<Curve> &slot = file.nodes[it->second].curves[*pendingAttribute];
                    if (slot) {
                        ASSIMP_LOG_WARN("Maya ANIM: node '", pendingNode, "' has duplicate curves for one attribute; keeping the last");
                    }
                    slot = std::move(curve);
                }
            } else {
                cursor.SkipBlock();
            }
            pendingAttribute.reset();
            continue;
        }
        if (statement.empty()) continue;

        const std::string_view name = statement[0];
        if (name == "anim") {
            // "anim <attr.path> <leafAttr> <node> <row> <children> <attrIndex>" for an
            // animated attribute, or "anim <node> <row> <children> <attrIndex>" for a
            // node written only to preserve hierarchy order, which carries no curve.
            pendingAttribute.reset();
            if (statement.size() >= 6) {
                pendingNode = std::string(statement[3]);
                pendingAttribute = AttributeIndex(statement[2]);
                if (!pendingAttribute) {
                    ASSIMP_LOG_VERBOSE_DEBUG("Maya ANIM: ignoring unsupported attribute '", std::string(statement[2]), "' on '", pendingNode, "'");
                }
            }
        } else if (statement.size() >= 2) {
            const std::string_view value = statement[1];
            if (name == "animVersion") {
                file.animVersion = std::string(value);
            } else if (name == "mayaVersion") {
                file.mayaVersion = std::string(value);
            } else if (name == "timeUnit") {
                file.framesPerSecond = TimeUnitToFps(value);
            } else if (name == "angularUnit") {
                file.angularToRadians = AngularUnitToRadians(value);
            } else if (name == "startTime") {
                file.startTime = ParseNumber(value);
            } else if (name == "endTime") {
                file.endTime = ParseNumber(value);
            }
        }
    }
    return file;
}

// ------------------------------------------------------------------------------------------------
// Curve evaluation.

/// Slope of the straight line between two keys.
double SecantSlope(const Key &from, const Key &to) {
    const double dt = to.time - from.time;
    return dt != 0.0 ? (to.value - from.value) / dt : 0.0;
}

/// Slope for one side of key i of the given tangent type. Smooth types
/// approximate Maya's own solver with a Catmull-Rom slope; clamped, plateau
/// and auto additionally flatten at local extrema, which is the overshoot
/// those types exist to prevent.
double TangentSlope(const std::vector<Key> &keys, size_t i, TangentType type, const std::optional<double> &angle, bool inSide) {
    const bool hasPrev = i > 0;
    const bool hasNext = i + 1 < keys.size();
    switch (type) {
    case TangentType::Fixed:
        // Stored as atan(slope) in value units per frame (BrawlCrate's convention).
        if (angle) return std::tan(*angle);
        break;
    case TangentType::Linear:
        if (inSide && hasPrev) return SecantSlope(keys[i - 1], keys[i]);
        if (!inSide && hasNext) return SecantSlope(keys[i], keys[i + 1]);
        break;
    case TangentType::Flat:
    case TangentType::Step:
    case TangentType::StepNext:
        return 0.0;
    default:
        break;
    }

    if (hasPrev && hasNext) {
        const bool clampsExtrema = type == TangentType::Clamped || type == TangentType::Plateau || type == TangentType::Auto;
        const double rise = keys[i].value - keys[i - 1].value;
        const double fall = keys[i + 1].value - keys[i].value;
        if (clampsExtrema && rise * fall <= 0.0) return 0.0;
        return SecantSlope(keys[i - 1], keys[i + 1]);
    }
    if (hasPrev) return SecantSlope(keys[i - 1], keys[i]);
    if (hasNext) return SecantSlope(keys[i], keys[i + 1]);
    return 0.0;
}

/// Sorts the curve's keys and resolves every key's in/out slope.
void FinalizeCurve(Curve &curve) {
    std::stable_sort(curve.keys.begin(), curve.keys.end(),
            [](const Key &a, const Key &b) { return a.time < b.time; });
    for (size_t i = 0; i < curve.keys.size(); ++i) {
        Key &key = curve.keys[i];
        key.inSlope = TangentSlope(curve.keys, i, key.inType, key.inAngle, true);
        key.outSlope = TangentSlope(curve.keys, i, key.outType, key.outAngle, false);
    }
}

/// Evaluates the segment between two adjacent keys at time t (t0 <= t <= t1).
double EvaluateSegment(const Key &k0, const Key &k1, double t) {
    if (k0.outType == TangentType::Step) return t < k1.time ? k0.value : k1.value;
    if (k0.outType == TangentType::StepNext) return t > k0.time ? k1.value : k0.value;

    const double dt = k1.time - k0.time;
    if (dt <= 0.0) return k1.value;

    // Cubic Hermite with the slopes scaled to the segment's length.
    const double s = (t - k0.time) / dt;
    const double s2 = s * s;
    const double s3 = s2 * s;
    const double h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
    const double h10 = s3 - 2.0 * s2 + s;
    const double h01 = -2.0 * s3 + 3.0 * s2;
    const double h11 = s3 - s2;
    return h00 * k0.value + h10 * dt * k0.outSlope + h01 * k1.value + h11 * dt * k1.inSlope;
}

/// Evaluates the curve within its keyed range [first.time, last.time].
double EvaluateInside(const Curve &curve, double t) {
    const std::vector<Key> &keys = curve.keys;
    const auto next = std::upper_bound(keys.begin(), keys.end(), t,
            [](double time, const Key &key) { return time < key.time; });
    if (next == keys.begin()) return keys.front().value;
    if (next == keys.end()) return keys.back().value;
    return EvaluateSegment(*(next - 1), *next, t);
}

/// Evaluates the curve at any time, applying its pre/post infinity outside the keyed range.
double EvaluateCurve(const Curve &curve, double t) {
    const Key &first = curve.keys.front();
    const Key &last = curve.keys.back();
    const double span = last.time - first.time;
    if (t >= first.time && t <= last.time) return EvaluateInside(curve, t);

    const bool before = t < first.time;
    const Infinity mode = before ? curve.preInfinity : curve.postInfinity;
    if (span <= 0.0 || mode == Infinity::Constant) {
        return before ? first.value : last.value;
    }
    if (mode == Infinity::Linear) {
        return before ? first.value + first.inSlope * (t - first.time) : last.value + last.outSlope * (t - last.time);
    }

    // Repeating modes: fold t back into the keyed range, counting whole cycles.
    const double cycles = std::floor((t - first.time) / span);
    double local = t - cycles * span;
    if (mode == Infinity::Oscillate && std::fmod(std::fabs(cycles), 2.0) == 1.0) {
        local = first.time + last.time - local; // Odd cycles play backwards.
    }
    double value = EvaluateInside(curve, local);
    if (mode == Infinity::CycleRelative) {
        value += cycles * (last.value - first.value);
    }
    return value;
}

/// True when the curve yields the same value everywhere, which lets a group
/// be written as two keys instead of one per frame.
bool IsConstant(const Curve &curve) {
    for (const Key &key : curve.keys) {
        if (key.value != curve.keys.front().value) return false;
    }
    // With a single key, linear infinity extrapolates along its slope.
    const Key &only = curve.keys.front();
    const bool slopedPre = curve.preInfinity == Infinity::Linear && only.inSlope != 0.0;
    const bool slopedPost = curve.postInfinity == Infinity::Linear && only.outSlope != 0.0;
    if (slopedPre || slopedPost) return false;
    // Equal values can still bulge between keys if the tangents aren't flat.
    for (const Key &key : curve.keys) {
        if (curve.keys.size() > 1 && (key.inSlope != 0.0 || key.outSlope != 0.0)) return false;
    }
    return true;
}

// ------------------------------------------------------------------------------------------------
// Transform math.

/// Builds the quaternion for Maya's default "xyz" rotate order: X is applied
/// first, then Y, then Z, i.e. R = Rz * Ry * Rx for column vectors.
aiQuaternion QuaternionFromEulerXyz(double x, double y, double z) {
    const double cx = std::cos(x * 0.5), sx = std::sin(x * 0.5);
    const double cy = std::cos(y * 0.5), sy = std::sin(y * 0.5);
    const double cz = std::cos(z * 0.5), sz = std::sin(z * 0.5);
    return aiQuaternion(
            static_cast<ai_real>(cz * cy * cx + sz * sy * sx),
            static_cast<ai_real>(cz * cy * sx - sz * sy * cx),
            static_cast<ai_real>(cz * sy * cx + sz * cy * sx),
            static_cast<ai_real>(sz * cy * cx - cz * sy * sx));
}

/// Inverse of QuaternionFromEulerXyz for a pure rotation matrix. Returns the
/// branch with Y in [-90, 90] degrees; see AlternateEulerBranch for the other.
aiVector3D EulerXyzFromMatrix(const aiMatrix3x3 &m) {
    const double sinY = std::clamp(-static_cast<double>(m.c1), -1.0, 1.0);
    const double y = std::asin(sinY);
    double x = 0.0;
    double z = 0.0;
    if (std::fabs(sinY) < 0.99999) {
        x = std::atan2(m.c2, m.c3);
        z = std::atan2(m.b1, m.a1);
    } else {
        // Gimbal lock: X and Z rotate about the same axis, so fold it all into Z.
        z = std::atan2(-m.a2, m.b2);
    }
    return aiVector3D(static_cast<ai_real>(x), static_cast<ai_real>(y), static_cast<ai_real>(z));
}

/// Per-attribute values used for axes a node's file leaves unkeyed, in the
/// same layout as NodeCurves::curves (rotation in radians).
using AttributeValues = std::array<double, cNumAttributes>;

/// Maya's own defaults for an attribute with no value from anywhere else.
AttributeValues IdentityAttributes() {
    return { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 };
}

/// Decomposes a node's local transform into the attribute layout.
AttributeValues AttributesFromTransform(const aiMatrix4x4 &transform) {
    aiVector3D scaling;
    aiQuaternion rotation;
    aiVector3D position;
    transform.Decompose(scaling, rotation, position);
    const aiVector3D euler = EulerXyzFromMatrix(rotation.GetMatrix());
    return { position.x, position.y, position.z, euler.x, euler.y, euler.z, scaling.x, scaling.y, scaling.z };
}

/// Every XYZ Euler rotation (x, y, z) has exactly one other representation
/// in the same order, (x + 180, 180 - y, z + 180) in degrees.
std::array<double, 3> AlternateEulerBranch(const std::array<double, 3> &euler) {
    return { euler[0] + cPi, cPi - euler[1], euler[2] + cPi };
}

/// Sum of angular distances (wrapped to [0, pi]) between the keyed rotation
/// axes' first values and the matching components of a candidate branch.
double KeyedAxisDistance(const NodeCurves &node, const std::array<double, 3> &branch, double angularToRadians) {
    double distance = 0.0;
    for (size_t axis = 0; axis < 3; ++axis) {
        const std::optional<Curve> &curve = node.curves[Group_Rotate * 3 + axis];
        if (!curve) continue;
        const double keyed = curve->keys.front().value * angularToRadians;
        distance += std::fabs(std::remainder(keyed - branch[axis], 2.0 * cPi));
    }
    return distance;
}

/// Picks the bind pose's Euler branch that agrees with the node's keyed
/// rotation axes. When a node keys only some axes, the rest come from the
/// bind pose, and the two must belong to the same branch: the bind matrix
/// alone can't say which branch the author used. For example, a joint
/// authored as (180, 180, 90) that keys rotateY = 180 decomposes to
/// (0, 0, -90), and mixing that branch's X/Z with the keyed Y flips the joint.
/// The keyed values reveal the branch, so this picks whichever is closer to
/// them. Returns a copy; the input is not modified.
AttributeValues MatchBindEulerBranch(const AttributeValues &bind, const NodeCurves &node, double angularToRadians) {
    const std::array<double, 3> primary = { bind[3], bind[4], bind[5] };
    const std::array<double, 3> alternate = AlternateEulerBranch(primary);
    if (KeyedAxisDistance(node, alternate, angularToRadians) >= KeyedAxisDistance(node, primary, angularToRadians)) {
        return bind;
    }
    AttributeValues matched = bind;
    std::copy(alternate.begin(), alternate.end(), matched.begin() + Group_Rotate * 3);
    return matched;
}

// ------------------------------------------------------------------------------------------------
// Scene building.

/// Time range the animation is baked over, in file frames.
struct BakeRange {
    double start = 0.0;
    double end = 0.0;
};

/// Uses the header's playback range when present; it's what the author saw
/// as the clip, including any hold past the last key. Falls back to the span
/// of all keys.
BakeRange ComputeBakeRange(const AnimFile &file) {
    double minTime = 0.0;
    double maxTime = 0.0;
    bool any = false;
    for (const NodeCurves &node : file.nodes) {
        for (const std::optional<Curve> &curve : node.curves) {
            if (!curve) continue;
            const double first = curve->keys.front().time;
            const double last = curve->keys.back().time;
            minTime = any ? std::min(minTime, first) : first;
            maxTime = any ? std::max(maxTime, last) : last;
            any = true;
        }
    }

    BakeRange range{ file.startTime.value_or(minTime), file.endTime.value_or(maxTime) };
    if (range.end < range.start) range.end = range.start;
    return range;
}

/// True when every keyed axis of the group is constant, so two keys suffice.
bool IsConstantGroup(const NodeCurves &node, size_t group) {
    for (size_t axis = 0; axis < 3; ++axis) {
        const std::optional<Curve> &curve = node.curves[group * 3 + axis];
        if (curve && !IsConstant(*curve)) return false;
    }
    return true;
}

/// Throws if baking every non-constant group on every frame would exceed
/// cMaxBakedSamples. Must run before SampleTimes, which casts the frame count
/// to size_t and reserves for it.
void CheckBakeSize(const AnimFile &file, const BakeRange &range) {
    const double framesPerGroup = std::floor(range.end - range.start) + 2.0;
    double total = 0.0;
    for (const NodeCurves &node : file.nodes) {
        for (size_t group = 0; group < 3; ++group) {
            const bool hasGroup = node.curves[group * 3] || node.curves[group * 3 + 1] || node.curves[group * 3 + 2];
            if (hasGroup && !IsConstantGroup(node, group)) total += framesPerGroup;
        }
    }
    if (total > cMaxBakedSamples) {
        throw DeadlyImportError("Maya ANIM: time range ", range.start, "..", range.end,
                " would bake ", total, " keys, more than the limit of ", cMaxBakedSamples);
    }
}

/// Frame times (in file frames) at which one attribute group is sampled.
/// Constant groups only need the range's ends; anything else is sampled on
/// every frame, since aiNodeAnim interpolates linearly and the Euler
/// components must be combined at common times anyway.
std::vector<double> SampleTimes(const NodeCurves &node, size_t group, const BakeRange &range) {
    std::vector<double> times;
    if (IsConstantGroup(node, group)) {
        times.push_back(range.start);
        if (range.end > range.start) times.push_back(range.end);
        return times;
    }

    const double span = range.end - range.start;
    const auto wholeFrames = static_cast<size_t>(std::floor(span));
    times.reserve(wholeFrames + 2);
    for (size_t frame = 0; frame <= wholeFrames; ++frame) {
        times.push_back(range.start + static_cast<double>(frame));
    }
    if (times.back() < range.end) times.push_back(range.end);
    return times;
}

/// Evaluates one group's three components at time t, using the fallback for unkeyed axes.
std::array<double, 3> EvaluateGroup(const NodeCurves &node, size_t group, double t, const AttributeValues &fallback, double valueScale) {
    std::array<double, 3> result{};
    for (size_t axis = 0; axis < 3; ++axis) {
        const size_t index = group * 3 + axis;
        const std::optional<Curve> &curve = node.curves[index];
        result[axis] = curve ? EvaluateCurve(*curve, t) * valueScale : fallback[index];
    }
    return result;
}

bool HasGroup(const NodeCurves &node, size_t group) {
    return node.curves[group * 3] || node.curves[group * 3 + 1] || node.curves[group * 3 + 2];
}

bool HasPartialGroup(const NodeCurves &node, size_t group) {
    return HasGroup(node, group) && !(node.curves[group * 3] && node.curves[group * 3 + 1] && node.curves[group * 3 + 2]);
}

/// Copies a filled vector into a new[]-allocated array as aiNodeAnim expects.
template <typename T>
T *ToOwnedArray(const std::vector<T> &values, unsigned int &count) {
    count = static_cast<unsigned int>(values.size());
    if (values.empty()) return nullptr;
    T *array = new T[values.size()];
    std::copy(values.begin(), values.end(), array);
    return array;
}

/// Bakes all curves of one node into a channel. Groups the file never keys
/// get no keys at all, which consumers treat as "keep the node's own value".
/// Key times are shifted so the bake range starts at tick 0.
aiNodeAnim *BuildChannel(const NodeCurves &node, const BakeRange &range, const AttributeValues &fallback, double angularToRadians) {
    std::unique_ptr<aiNodeAnim> channel(new aiNodeAnim());
    channel->mNodeName.Set(node.name);

    if (HasGroup(node, Group_Translate)) {
        std::vector<aiVectorKey> keys;
        for (double t : SampleTimes(node, Group_Translate, range)) {
            const std::array<double, 3> v = EvaluateGroup(node, Group_Translate, t, fallback, 1.0);
            keys.emplace_back(t - range.start, aiVector3D(static_cast<ai_real>(v[0]), static_cast<ai_real>(v[1]), static_cast<ai_real>(v[2])));
        }
        channel->mPositionKeys = ToOwnedArray(keys, channel->mNumPositionKeys);
    }

    if (HasGroup(node, Group_Rotate)) {
        std::vector<aiQuatKey> keys;
        for (double t : SampleTimes(node, Group_Rotate, range)) {
            const std::array<double, 3> euler = EvaluateGroup(node, Group_Rotate, t, fallback, angularToRadians);
            aiQuaternion q = QuaternionFromEulerXyz(euler[0], euler[1], euler[2]);
            // Keep consecutive keys in the same hemisphere so that consumers
            // which don't pick the shortest path still interpolate the short way.
            if (!keys.empty()) {
                const aiQuaternion &prev = keys.back().mValue;
                if (prev.w * q.w + prev.x * q.x + prev.y * q.y + prev.z * q.z < 0) {
                    q = aiQuaternion(-q.w, -q.x, -q.y, -q.z);
                }
            }
            keys.emplace_back(t - range.start, q);
        }
        channel->mRotationKeys = ToOwnedArray(keys, channel->mNumRotationKeys);
    }

    if (HasGroup(node, Group_Scale)) {
        std::vector<aiVectorKey> keys;
        for (double t : SampleTimes(node, Group_Scale, range)) {
            const std::array<double, 3> v = EvaluateGroup(node, Group_Scale, t, fallback, 1.0);
            keys.emplace_back(t - range.start, aiVector3D(static_cast<ai_real>(v[0]), static_cast<ai_real>(v[1]), static_cast<ai_real>(v[2])));
        }
        channel->mScalingKeys = ToOwnedArray(keys, channel->mNumScalingKeys);
    }

    return channel.release();
}

/// Strips directories and the extension, for naming the animation.
std::string BaseName(const std::string &path) {
    const size_t slash = path.find_last_of("/\\");
    std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    const size_t dot = name.find_last_of('.');
    if (dot != std::string::npos) name.resize(dot);
    return name;
}

} // namespace

// ------------------------------------------------------------------------------------------------
bool MayaAnimImporter::CanRead(const std::string &pFile, IOSystem *pIOHandler, bool /*checkSig*/) const {
    // The version line is mandatory and first; it also rules out Unity's
    // unrelated YAML-based .anim files.
    static const char *tokens[] = { "animVersion" };
    return SearchFileHeaderForToken(pIOHandler, pFile, tokens, AI_COUNT_OF(tokens));
}

// ------------------------------------------------------------------------------------------------
void MayaAnimImporter::SetupProperties(const Importer *pImp) {
    mReferenceScene = static_cast<const aiScene *>(pImp->GetPropertyPointer(AI_CONFIG_IMPORT_MAYA_ANIM_REFERENCE_SCENE, nullptr));
}

// ------------------------------------------------------------------------------------------------
const aiImporterDesc *MayaAnimImporter::GetInfo() const {
    return &cDesc;
}

// ------------------------------------------------------------------------------------------------
void MayaAnimImporter::InternReadFile(const std::string &pFile, aiScene *pScene, IOSystem *pIOHandler) {
    std::unique_ptr<IOStream> stream(pIOHandler->Open(pFile, "rb"));
    if (!stream) {
        throw DeadlyImportError("Failed to open Maya ANIM file ", pFile, ".");
    }
    std::vector<char> buffer;
    TextFileToBuffer(stream.get(), buffer);

    TokenCursor cursor(Tokenize(std::string_view(buffer.data())));
    AnimFile file = ParseAnimFile(cursor);
    if (file.animVersion.empty()) {
        throw DeadlyImportError("Maya ANIM: missing animVersion header in ", pFile);
    }
    if (file.nodes.empty()) {
        throw DeadlyImportError("Maya ANIM: no time-driven transform curves in ", pFile);
    }
    for (NodeCurves &node : file.nodes) {
        for (std::optional<Curve> &curve : node.curves) {
            if (curve) FinalizeCurve(*curve);
        }
    }

    const BakeRange range = ComputeBakeRange(file);
    CheckBakeSize(file, range);

    // A flat list of nodes under a dummy root: the format's row/child counts
    // could describe a hierarchy, but writers omit unanimated nodes, which
    // makes them unreliable.
    pScene->mRootNode = new aiNode("MayaAnimRoot");
    pScene->mRootNode->mNumChildren = static_cast<unsigned int>(file.nodes.size());
    pScene->mRootNode->mChildren = new aiNode *[file.nodes.size()];

    std::unique_ptr<aiAnimation> anim(new aiAnimation());
    anim->mName.Set(BaseName(pFile));
    anim->mTicksPerSecond = file.framesPerSecond;
    anim->mDuration = range.end - range.start;
    anim->mNumChannels = static_cast<unsigned int>(file.nodes.size());
    anim->mChannels = new aiNodeAnim *[file.nodes.size()];

    unsigned int missingReference = 0;
    unsigned int defaultedPartial = 0;
    for (size_t i = 0; i < file.nodes.size(); ++i) {
        const NodeCurves &node = file.nodes[i];
        aiNode *sceneNode = new aiNode(node.name);
        sceneNode->mParent = pScene->mRootNode;
        pScene->mRootNode->mChildren[i] = sceneNode;

        AttributeValues fallback = IdentityAttributes();
        const aiNode *bindNode = mReferenceScene && mReferenceScene->mRootNode ? mReferenceScene->mRootNode->FindNode(node.name.c_str()) : nullptr;
        if (bindNode) {
            fallback = MatchBindEulerBranch(AttributesFromTransform(bindNode->mTransformation), node, file.angularToRadians);
            // Also gives the dummy node the bind pose, so the output is
            // meaningful on its own (e.g. in a viewer).
            sceneNode->mTransformation = bindNode->mTransformation;
        } else {
            if (mReferenceScene) ++missingReference;
            const bool partial = HasPartialGroup(node, Group_Translate) || HasPartialGroup(node, Group_Rotate) || HasPartialGroup(node, Group_Scale);
            if (partial) ++defaultedPartial;
        }

        anim->mChannels[i] = BuildChannel(node, range, fallback, file.angularToRadians);
    }

    if (missingReference) {
        ASSIMP_LOG_WARN("Maya ANIM: ", missingReference, " animated node(s) not found in the reference scene");
    }
    if (defaultedPartial) {
        ASSIMP_LOG_WARN("Maya ANIM: ", defaultedPartial, " node(s) key only some axes of translate/rotate/scale; "
                                                         "unkeyed axes were set to 0 (scale: 1) instead of their bind value. Set "
                                                         "AI_CONFIG_IMPORT_MAYA_ANIM_REFERENCE_SCENE to fill them from the model.");
    }

    pScene->mNumAnimations = 1;
    pScene->mAnimations = new aiAnimation *[1];
    pScene->mAnimations[0] = anim.release();

    // Animation-only: there is no geometry to validate against.
    pScene->mFlags |= AI_SCENE_FLAGS_INCOMPLETE;

    pScene->mMetaData = new aiMetadata();
    pScene->mMetaData->Add(AI_METADATA_SOURCE_FORMAT, aiString("Maya ANIM"));
    pScene->mMetaData->Add(AI_METADATA_SOURCE_FORMAT_VERSION, aiString(file.animVersion));
    if (!file.mayaVersion.empty()) {
        pScene->mMetaData->Add(AI_METADATA_SOURCE_GENERATOR, aiString("Maya " + file.mayaVersion));
    }
}

} // namespace Assimp

#endif // ASSIMP_BUILD_NO_MAYAANIM_IMPORTER
