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
