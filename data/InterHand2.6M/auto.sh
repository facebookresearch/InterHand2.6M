#!/bin/bash

# Check if the dataset folder exists or not
# Note: if the folder was not extracted well, you will need to delete and run the script again
if [ -d "InterHand2.6M_5fps_batch0" ]; then
    echo -e "Extraction output exists, Dataset Found!\n"
    exit
else
    echo -e "Output folder was not found, Auto Downloading InterHand2.6M Dataset\n"
fi

# Checksum to make sure files was downloaded proparley
FILETXT=InterHand2.6M.images.md5sum.5.fps.v0.0.txt
if [ -f "$FILETXT" ]; then
    echo -e "$FILETXT was found\n"
else
    wget https://github.com/facebookresearch/InterHand2.6M/releases/download/v0.0/InterHand2.6M.images.md5sum.5.fps.v0.0.txt
fi

# Loop over parts for downloading
for part in a b c d e f g h i j k l m n o p q r s t u
do
    FILE=InterHand2.6M.images.5.fps.v0.0.tar.parta${part}
    # if the file was found and not corrupted, it will pass
    # otherwise it will be redownloaded
    if [ -f "$FILE" ]; then
        echo "$FILE was found"
        if [[ $(cat "$FILETXT" | grep "$FILE") = $(md5sum "$FILE") ]]; then
            echo -e "No Corruption was detected\n"
        else
            # delete the corrupted part and redownload
            echo "And was found corrupted ... will need to redownload"
            rm "$File"
            wget https://github.com/facebookresearch/InterHand2.6M/releases/download/v0.0/${FILE}
        fi
    else
        wget https://github.com/facebookresearch/InterHand2.6M/releases/download/v0.0/${FILE}
    fi

done;

# Extract the dataset folder
cat InterHand2.6M.images.5.fps.v0.0.tar.parta* | tar -xvf - -i

# Download & Extract annotations
FILEAnotate=InterHand2.6M.annotations.5.fps.zip
if [ -f "$FILEAnotate" ]; then
    echo "$FILEAnotate was found."
else
    wget https://github.com/facebookresearch/InterHand2.6M/releases/download/v0.0/$FILEAnotate
fi
unzip -q "InterHand2.6M_5fps_batch0/$FILEAnotate"