# Fetches only the parts of the PowerVR Native SDK that the POD importer needs.
#
# The SDK tag archive is ~700 MB, almost all of it examples and prebuilt
# libraries. FetchContent downloads it (without extracting) and the needed
# directories are then unpacked with a PATTERNS filter, leaving a few MB.
#
# By default FetchContent stores a downloaded archive twice. To keep a single
# copy: CMP0168 NEW avoids the sub-build copy, SOURCE_DIR and DOWNLOAD_DIR are
# the same directory so the archive is not copied again, and the remaining
# scratch copy in <FETCHCONTENT_BASE_DIR>/<name>-tmp is deleted afterwards.
#
# glm is a git submodule of the SDK, and GitHub's tag archives omit submodule
# contents, so glm is fetched separately, pinned to the commit the SDK tag
# references. Only its headers (glm/) are extracted.
#
# On return, PVR_SDK_DIR (a normal variable shadowing the cache entry) points at
# a directory laid out like an SDK checkout, containing:
#   framework/PVRAssets, framework/PVRCore, external/glm/source/glm
#
# Both upstream projects are MIT licensed.

include_guard(GLOBAL)

# Needs CMake >= 3.30. On older versions the archive is stored twice (in the
# FetchContent sub-build and in the source directory), which still works.
if (POLICY CMP0168)
    cmake_policy(SET CMP0168 NEW)
endif ()
include(FetchContent)

set(ASSIMP_PVR_SDK_TAG "R26.1-v5.16")
set(ASSIMP_PVR_SDK_URL "https://github.com/powervr-graphics/Native_SDK/archive/refs/tags/${ASSIMP_PVR_SDK_TAG}.zip")
# Commit of external/glm/source in the SDK tag above.
set(ASSIMP_PVR_GLM_COMMIT "658d8960d081e0c9c312d49758c7ef919371b428")
set(ASSIMP_PVR_GLM_URL "https://github.com/g-truc/glm/archive/${ASSIMP_PVR_GLM_COMMIT}.zip")

# Where the downloaded archives are kept. Point this at a shared location to
# avoid re-downloading the 700 MB SDK archive for every build directory.
set(ASSIMP_PVR_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/pvr_downloads" CACHE PATH
    "Directory in which the PowerVR SDK and glm archives are cached.")

# Makes sure the archive at url is present in ASSIMP_PVR_DOWNLOAD_DIR (under its
# URL basename) and returns its path in the variable named by out_archive.
function(_assimp_pvr_download name url out_archive)
    get_filename_component(archive_name "${url}" NAME)
    set(archive "${ASSIMP_PVR_DOWNLOAD_DIR}/${archive_name}")
    if (NOT EXISTS "${archive}")
        FetchContent_Declare(${name}
            URL "${url}"
            DOWNLOAD_NO_EXTRACT TRUE
            SOURCE_DIR "${ASSIMP_PVR_DOWNLOAD_DIR}"
            DOWNLOAD_DIR "${ASSIMP_PVR_DOWNLOAD_DIR}")
        FetchContent_MakeAvailable(${name})
        # Drop FetchContent's scratch copy of the archive.
        file(REMOVE_RECURSE "${FETCHCONTENT_BASE_DIR}/${name}-tmp")
    endif ()
    set(${out_archive} "${archive}" PARENT_SCOPE)
endfunction()

set(_pvr_root "${CMAKE_BINARY_DIR}/pvr_sdk_subset")
set(_pvr_sdk_dir "${_pvr_root}/sdk")
set(_pvr_stamp "${_pvr_root}/extracted-${ASSIMP_PVR_SDK_TAG}-${ASSIMP_PVR_GLM_COMMIT}.stamp")

if (NOT EXISTS "${_pvr_stamp}")
    message(STATUS "Fetching PowerVR Native SDK ${ASSIMP_PVR_SDK_TAG} (large download, cached in ${ASSIMP_PVR_DOWNLOAD_DIR})...")
    _assimp_pvr_download(assimp_pvr_sdk "${ASSIMP_PVR_SDK_URL}" _pvr_sdk_zip)
    _assimp_pvr_download(assimp_pvr_glm "${ASSIMP_PVR_GLM_URL}" _pvr_glm_zip)

    # Extract into scratch directories first, since GitHub archives have a
    # versioned top-level directory that we want to drop.
    set(_pvr_stage "${_pvr_root}/stage")
    file(REMOVE_RECURSE "${_pvr_stage}" "${_pvr_sdk_dir}")
    file(MAKE_DIRECTORY "${_pvr_stage}/sdk" "${_pvr_stage}/glm")

    file(ARCHIVE_EXTRACT INPUT "${_pvr_sdk_zip}" DESTINATION "${_pvr_stage}/sdk"
        PATTERNS "*/framework/PVRAssets/*" "*/framework/PVRCore/*")
    file(ARCHIVE_EXTRACT INPUT "${_pvr_glm_zip}" DESTINATION "${_pvr_stage}/glm"
        PATTERNS "*/glm/*")

    file(GLOB _pvr_sdk_top "${_pvr_stage}/sdk/Native_SDK-*")
    file(GLOB _pvr_glm_top "${_pvr_stage}/glm/glm-*")
    if (NOT IS_DIRECTORY "${_pvr_sdk_top}/framework/PVRAssets" OR NOT IS_DIRECTORY "${_pvr_glm_top}/glm")
        message(FATAL_ERROR "Unexpected PowerVR SDK / glm archive layout; extraction under ${_pvr_stage} failed.")
    endif ()

    file(MAKE_DIRECTORY "${_pvr_sdk_dir}/external/glm")
    file(RENAME "${_pvr_sdk_top}/framework" "${_pvr_sdk_dir}/framework")
    file(RENAME "${_pvr_glm_top}" "${_pvr_sdk_dir}/external/glm/source")
    file(REMOVE_RECURSE "${_pvr_stage}")
    file(WRITE "${_pvr_stamp}" "")
endif ()

set(PVR_SDK_DIR "${_pvr_sdk_dir}")
