execute_process(COMMAND "/home/zhuyihang/simEnv/build/simenv_cbf_train/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/zhuyihang/simEnv/build/simenv_cbf_train/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
