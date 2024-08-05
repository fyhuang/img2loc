#!/bin/bash

set -exu

OUT_DIR=../../ghpages_img2loc

rm -rf dist
npm run build
rm -f $OUT_DIR/*
rsync -av dist/ $OUT_DIR/
