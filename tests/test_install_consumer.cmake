if(NOT DEFINED PARAKEET_BINARY_DIR OR NOT DEFINED PARAKEET_CONSUMER_SOURCE_DIR)
    message(FATAL_ERROR
        "Parakeet install-consumer test requires build and consumer paths")
endif()

set(_root "${CMAKE_CURRENT_BINARY_DIR}/parakeet-install-consumer")
set(_prefix "${_root}/prefix")
set(_consumer_build "${_root}/consumer-build")
file(REMOVE_RECURSE "${_root}")

execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${PARAKEET_BINARY_DIR}" --prefix "${_prefix}"
    RESULT_VARIABLE _install_result
    OUTPUT_VARIABLE _install_stdout
    ERROR_VARIABLE _install_stderr
)
if(NOT _install_result EQUAL 0)
    message(FATAL_ERROR
        "Parakeet installation failed:\n${_install_stdout}\n${_install_stderr}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -S "${PARAKEET_CONSUMER_SOURCE_DIR}" -B "${_consumer_build}"
            "-DCMAKE_PREFIX_PATH=${_prefix}"
    RESULT_VARIABLE _consumer_configure_result
    OUTPUT_VARIABLE _consumer_configure_stdout
    ERROR_VARIABLE _consumer_configure_stderr
)
if(NOT _consumer_configure_result EQUAL 0)
    message(FATAL_ERROR
        "Installed Parakeet package could not configure an external consumer:\n"
        "${_consumer_configure_stdout}\n${_consumer_configure_stderr}")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" --build "${_consumer_build}"
    RESULT_VARIABLE _consumer_build_result
    OUTPUT_VARIABLE _consumer_build_stdout
    ERROR_VARIABLE _consumer_build_stderr
)
if(NOT _consumer_build_result EQUAL 0)
    message(FATAL_ERROR
        "Installed Parakeet package could not build an external consumer:\n"
        "${_consumer_build_stdout}\n${_consumer_build_stderr}")
endif()

execute_process(
    COMMAND "${_consumer_build}/parakeet_install_consumer"
    RESULT_VARIABLE _consumer_run_result
    OUTPUT_VARIABLE _consumer_run_stdout
    ERROR_VARIABLE _consumer_run_stderr
)
if(NOT _consumer_run_result EQUAL 0)
    message(FATAL_ERROR
        "Installed Parakeet consumer failed at runtime:\n"
        "${_consumer_run_stdout}\n${_consumer_run_stderr}")
endif()
