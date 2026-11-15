# nvidia-cuda-toolkit 11.3

**Note:** Compatible with Ubuntu 20.04 and CuPy as well.

Guides:

- [Third-party guide](http://ubuntuhandbook.org/index.php/2017/07/install-python-3-6-1-in-ubuntu-16-04-lts/)
- [Official NVIDIA guide](https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local)

Add to `.bashrc`:

```bash
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
```

Add the following environment variable in the PyCharm run configuration:

```text
LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:/usr/local/cuda-11.3/targets/x86_64-linux/lib
```
