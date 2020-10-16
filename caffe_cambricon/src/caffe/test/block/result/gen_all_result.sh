#!/bin/sh
export CNML_ADDITIONAL_DEBUG=PrintJsonFile
FILES_NAME=`ls ..`
CUR_DIR=`pwd`
if [ ! -d ./json ]
then
    echo "mkdir json"
    mkdir json
fi

clean_product_file=0

if [ $# = 1 ]
then
    echo "$1"
    if [ "$1" = "clean" ]
    then
        echo "clean will be done"
        clean_product_file=1
    fi
fi

for CURRENT_FILE in $FILES_NAME
do
    cd $CUR_DIR
    if [ $CURRENT_FILE != "result" ] && [ -d ../$CURRENT_FILE ]
    then
        echo "********************"
        echo "   $CURRENT_FILE"
        echo "********************"
        cd ../$CURRENT_FILE
        if [ $clean_product_file -eq 1 ]
        then
        echo "cleaning json files..."
                rm *.camb*
                rm -rf jsonfiles 2>> /dev/null
        fi
        CUR_PT=`ls *.pt 2> /dev/null`
        echo "$CUR_PT is CUR_PT"
        if [ "$CUR_PT" != "" ]
        then
	      HAS_CHANGE=`find -newer $CUR_PT | grep fusion_1.json | awk -F'/' '{print $NF}'`
	      if [ "$HAS_CHANGE" = "" ]
	      then
		      echo "$CUR_PT Has \033[31m Changed \033[0m"
		      echo "$HAS_CHANGE is HAS_CHANGE"
                      echo "Now gen json file, Waiting..."
                      cp ../result/just_gen_single_json.sh ./
                      ./just_gen_single_json.sh
                      rm just_gen_single_json.sh
	      else
		      echo "$CUR_PT is \033[32m Old \033[0m"
	      fi

	      #if [ -f jsonfiles/fusion_1.json ]abstract_fusion_1.json
	      if [ -f jsonfiles/abstract_fusion_1.json ]
	      then

		  cp jsonfiles/abstract_fusion_1.json $CUR_DIR/json/$CURRENT_FILE.json
	      fi
	else
              echo "$CURRENT_FILE has no .pt file"
	fi
    else
        echo "\033[32m Skip\033[0m $CURRENT_FILE"
    fi
    echo "\n"
done
