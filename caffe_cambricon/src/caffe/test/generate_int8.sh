name=${1%\.prototxt}
mode=common

echo "--------generate quantized pt ----------"
../build/tools/generate_quantized_pt --ini_file ../convert_quantized.ini \
   -model $1 -weights $2 -outputmodel ${name}_quantized.prototxt -mode $mode &> generate
