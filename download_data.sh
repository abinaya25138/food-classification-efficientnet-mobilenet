#!/bin/bash
# Downloads Food-101 (Bossard, Guillaumin & Van Gool, 2014) -
# 101 food categories, 101,000 images, ~5GB.
set -e
curl -L -o food-101.tar.gz "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
tar -xzf food-101.tar.gz
rm food-101.tar.gz
echo "Done. Images are in food-101/images/<class>/"
