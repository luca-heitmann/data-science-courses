#!/bin/bash

DIR_NAME=$(cd "$(dirname "$0")" && pwd)

(
    echo Executing A
    python "${DIR_NAME}/hw7a.py"
    echo ----------------------------------
    echo Executing B
    python "${DIR_NAME}/hw7b.py"
    echo ----------------------------------
    echo Executing C
    python "${DIR_NAME}/hw7c.py"
) | tee "${DIR_NAME}/hw7.log"
