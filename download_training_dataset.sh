#!/bin/bash
DSDIR="{1:-$DSDIR}"
gdown --fuzzy https://drive.google.com/file/d/12hOYsMa8t1ErKj6PZA352icsx9mz1TwB/view

sudo mv DIV2K_datasets.tar.gz $DSDIR
sudo tar -xf $DSDIR/DIV2K_datasets.tar.gz --directory $DSDIR/DIV2K
