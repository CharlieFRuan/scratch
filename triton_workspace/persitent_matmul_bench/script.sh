# Run 8K fp16
python 09-persistent-matmul.py --prec fp16 --K_range 512 8192 --K_step 512 --MN-size 8192
mv matmul.hatchet matmul_M\=N\=8K.hatchet
proton-viewer -m "tflop16/s" matmul_M\=N\=8K.hatchet > matmul_M\=N\=8K_output.txt

# Run 4K fp16
python 09-persistent-matmul.py --prec fp16 --K_range 256 4096 --K_step 256 --MN-size 4096
mv matmul.hatchet matmul_M\=N\=4K.hatchet
proton-viewer -m "tflop16/s" matmul_M\=N\=4K.hatchet > matmul_M\=N\=4K_output.txt
