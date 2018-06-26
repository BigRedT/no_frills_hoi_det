echo "Downloading glove.6B.zip file ..."
URL="http://nlp.stanford.edu/data/glove.6B.zip"
OUT_DIR="${PWD}/data_symlinks/hico_processed"
#wget --directory-prefix=$OUT_DIR $URL

echo "Unzipping glove.6B.zip to glove.6B ..."
GLOVE_ZIP="${OUT_DIR}/glove.6B.zip"
GLOVE_DIR="${OUT_DIR}/glove.6B"
mkdir $GLOVE_DIR
unzip $GLOVE_ZIP -d $GLOVE_DIR
