if(NOT DEFINED PARAKEET_SOURCE_DIR)
    message(FATAL_ERROR "Parakeet install-option test requires the source directory")
endif()

string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef _build_suffix)
set(_build_dir
    "${CMAKE_CURRENT_BINARY_DIR}/parakeet-install-options-${_build_suffix}")

execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${PARAKEET_SOURCE_DIR}" -B "${_build_dir}"
            -DPARAKEET_BUILD_CLI=OFF
            -DPARAKEET_BUILD_TESTS=OFF
            -DPARAKEET_BUILD_EXAMPLES=OFF
            -DPARAKEET_INSTALL=ON
            -DAXIOM_INSTALL=OFF
    RESULT_VARIABLE _configure_result
    OUTPUT_VARIABLE _configure_stdout
    ERROR_VARIABLE _configure_stderr
)

if(_configure_result EQUAL 0)
    message(FATAL_ERROR
        "PARAKEET_INSTALL=ON must reject an explicit AXIOM_INSTALL=OFF")
endif()

set(_configure_output "${_configure_stdout}\n${_configure_stderr}")
string(FIND "${_configure_output}" "PARAKEET_INSTALL=ON requires AXIOM_INSTALL=ON"
       _required_message_index)
if(_required_message_index EQUAL -1)
    message(FATAL_ERROR
        "The install-option failure must explain the required Axiom export:\n"
        "${_configure_output}")
endif()

# Parakeet owns the initial AXIOM_INSTALL=OFF value for a non-install build.
# Turning PARAKEET_INSTALL on later in the same build tree must promote that
# managed value instead of mistaking it for an explicit user choice.
set(_managed_build_dir "${_build_dir}-managed")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${PARAKEET_SOURCE_DIR}" -B "${_managed_build_dir}"
            -DPARAKEET_BUILD_CLI=OFF
            -DPARAKEET_BUILD_TESTS=OFF
            -DPARAKEET_BUILD_EXAMPLES=OFF
            -DPARAKEET_INSTALL=OFF
    RESULT_VARIABLE _managed_initial_result
    OUTPUT_VARIABLE _managed_initial_stdout
    ERROR_VARIABLE _managed_initial_stderr
)
if(NOT _managed_initial_result EQUAL 0)
    message(FATAL_ERROR
        "Parakeet initial non-install configuration failed:\n"
        "${_managed_initial_stdout}\n${_managed_initial_stderr}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${PARAKEET_SOURCE_DIR}" -B "${_managed_build_dir}"
            -DPARAKEET_BUILD_CLI=OFF
            -DPARAKEET_BUILD_TESTS=OFF
            -DPARAKEET_BUILD_EXAMPLES=OFF
            -DPARAKEET_INSTALL=ON
    RESULT_VARIABLE _managed_promote_result
    OUTPUT_VARIABLE _managed_promote_stdout
    ERROR_VARIABLE _managed_promote_stderr
)
if(NOT _managed_promote_result EQUAL 0)
    message(FATAL_ERROR
        "PARAKEET_INSTALL=ON must promote Parakeet's prior managed Axiom "
        "install value:\n${_managed_promote_stdout}\n${_managed_promote_stderr}")
endif()

file(READ "${_managed_build_dir}/CMakeCache.txt" _managed_cache)
if(NOT _managed_cache MATCHES "AXIOM_INSTALL:BOOL=ON")
    message(FATAL_ERROR
        "Promoting PARAKEET_INSTALL must set the managed AXIOM_INSTALL cache value to ON")
endif()

# A parent project can set AXIOM_INSTALL as an ordinary (non-cache) variable.
# That is still an explicit decision and must fail with the same clear error.
set(_parent_off_build_dir "${_build_dir}-parent-off")
execute_process(
    COMMAND "${CMAKE_COMMAND}"
            -S "${PARAKEET_SOURCE_DIR}/tests/install_parent_off"
            -B "${_parent_off_build_dir}"
            "-DPARAKEET_SOURCE_DIR=${PARAKEET_SOURCE_DIR}"
    RESULT_VARIABLE _parent_off_result
    OUTPUT_VARIABLE _parent_off_stdout
    ERROR_VARIABLE _parent_off_stderr
)
if(_parent_off_result EQUAL 0)
    message(FATAL_ERROR
        "PARAKEET_INSTALL=ON must reject a parent-scope AXIOM_INSTALL=OFF")
endif()

set(_parent_off_output "${_parent_off_stdout}\n${_parent_off_stderr}")
string(FIND "${_parent_off_output}" "PARAKEET_INSTALL=ON requires AXIOM_INSTALL=ON"
       _parent_required_message_index)
if(_parent_required_message_index EQUAL -1)
    message(FATAL_ERROR
        "The parent-scope install-option failure must explain the required Axiom export:\n"
        "${_parent_off_output}")
endif()
