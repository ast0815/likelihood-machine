#!/bin/bash

SCRIPT=`realpath $BASH_SOURCE`
SCRIPTPATH=`dirname $SCRIPT`

# Check whether the ENV exists
if [ -d "$SCRIPTPATH/ENV" ]
then
    # Activate the environment
    echo "Activating virtual environment."
    . "$SCRIPTPATH/ENV/bin/activate"
else
    # Create a new virtual environment
    virtualenv $@ "$SCRIPTPATH/ENV"

    # Activate the environment
    . "$SCRIPTPATH/ENV/bin/activate" && (

        # Upgrade pip to the latest version (optional)
        pip install --upgrade pip

        # Install all required packages
        pip install -r "$SCRIPTPATH/requirements.txt"
        pip install -r "$SCRIPTPATH/test-requirements.txt"
        pip install -r "$SCRIPTPATH/documentation-requirements.txt"

        # Install actual package
        pip install -e "$SCRIPTPATH"

    )
fi
