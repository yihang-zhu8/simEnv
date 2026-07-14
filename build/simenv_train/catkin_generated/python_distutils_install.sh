#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/zhuyihang/simEnv/src/simenv_train"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/zhuyihang/simEnv/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/zhuyihang/simEnv/install/lib/python3/dist-packages:/home/zhuyihang/simEnv/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/zhuyihang/simEnv/build" \
    "/usr/bin/python3" \
    "/home/zhuyihang/simEnv/src/simenv_train/setup.py" \
    egg_info --egg-base /home/zhuyihang/simEnv/build/simenv_train \
    build --build-base "/home/zhuyihang/simEnv/build/simenv_train" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/zhuyihang/simEnv/install" --install-scripts="/home/zhuyihang/simEnv/install/bin"
