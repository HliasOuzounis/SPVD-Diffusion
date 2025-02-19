export HF_TOKEN="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"

export DOWNLOAD_DIR="./data/ShapeNetCore/obj"
mkdir -p $DOWNLOAD_DIR

huggingface-cli download --repo-type dataset --token $HF_TOKEN --local-dir $DOWNLOAD_DIR ShapeNet/ShapeNetCore
# export URL="https://huggingface.co/datasets/ShapeNet/ShapeNetCore/resolve/main"


# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/02691156.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/02747177.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/02773838.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/02801938.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/02808440.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/02818832.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/02828884.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/02843684.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/02871439.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/02876657.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/02880940.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/02924116.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/02933112.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/02942699.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/02946921.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/02954340.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/02958343.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/02992529.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/03001627.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/03046257.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/03085013.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/03207941.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/03211117.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/03261776.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/03325088.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/03337140.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/03467517.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/03513137.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/03593526.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/03624134.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/03636649.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/03642806.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/03691459.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/03710193.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/03759954.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/03761084.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/03790512.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/03797390.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/03928116.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/03938244.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/03948459.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/03991062.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/04004475.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/04074963.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/04090263.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/04099429.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/04225987.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/04256520.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/04330267.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/04379243.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/04401088.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/04460130.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/04468005.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/04530566.zip
# wget -P $DOWNLOAD_DIR --header="Authorization: Bearer $HF_TOKEN" $URL/04554684.zip

cd $DOWNLOAD_DIR
unzip '*.zip'
rm *.zip
cd -