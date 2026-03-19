#!/bin/bash

SUBDIR=$1

# The date of the first commit (First Commit)
git log --reverse --format="%ad" --date=format:"%Y-W%V" $SUBDIR | head -n 1

# The date of the latest commit (Last Commit)
git log -1 --format="%ad" --date=format:"%Y-W%V" $SUBDIR
