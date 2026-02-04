nvidia-cuda-toolkit 11.3
========================

:note: comaptible with ubuntu 20.04 and cupy as well

.. _Official_nVIDIA_guide: https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local
.. _Guiedence_3rd_party: http://ubuntuhandbook.org/index.php/2017/07/install-python-3-6-1-in-ubuntu-16-04-lts/

- guide
    - Guiedence_3rd_party_
    - Official_nVIDIA_guide_

.bashrc::

    ### nvidia-cuda-toolkit 11.3 ###
    if [ -z "${PATH}" ] ; then
        PATH=/usr/local/cuda-11.3/bin
    else
        PATH=/usr/local/cuda-11.3/bin:${PATH}
    fi
    if [ -z "${LD_LIBRARY_PATH}" ] ; then
        LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:/usr/local/cuda-11.3/targets/x86_64-linux/lib
    else
        LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:/usr/local/cuda-11.3/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
    fi

Add following to ENV variables in PyCharm run config::

    LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:/usr/local/cuda-11.3/targets/x86_64-linux/lib


