#!/bin/bash
set -e

mkdir -p models/rdm/imagenet models/rdm/imagenet_in-db models/rdm/ffhq
mkdir -p models/rarm/imagenet/dogs models/rarm/imagenet/mammals models/rarm/imagenet/animals

wget -P models/rdm/imagenet/ https://ommer-lab.com/files/rdm/models/rdm/imagenet/model.ckpt
wget -P models/rdm/imagenet_in-db/ https://ommer-lab.com/files/rdm/models/rdm/imagenet_in-db/model.ckpt
wget -P models/rdm/ffhq/ https://ommer-lab.com/files/rdm/models/rdm/ffhq/model.ckpt

wget -P models/rarm/imagenet/dogs/ https://ommer-lab.com/files/rdm/models/rarm/imagenet/dogs/model.ckpt
wget -P models/rarm/imagenet/mammals/ https://ommer-lab.com/files/rdm/models/rarm/imagenet/mammals/model.ckpt
wget -P models/rarm/imagenet/animals/ https://ommer-lab.com/files/rdm/models/rarm/imagenet/animals/model.ckpt
