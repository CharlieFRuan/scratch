# Run 8K fp16
python 09-persistent-matmul.py --prec fp16 --K_range 512 8192 --K_step 512 --MN-size 8192
mv matmul.hatchet matmul_M\=N\=8K_fp16.hatchet
proton-viewer -m "tflop16/s" matmul_M\=N\=8K_fp16.hatchet > matmul_M\=N\=8K_output_fp16.txt

# Run 4K fp16
python 09-persistent-matmul.py --prec fp16 --K_range 256 4096 --K_step 256 --MN-size 4096
mv matmul.hatchet matmul_M\=N\=4K_fp16.hatchet
proton-viewer -m "tflop16/s" matmul_M\=N\=4K_fp16.hatchet > matmul_M\=N\=4K_output_fp16.txt

# Run 8K fp8
python 09-persistent-matmul.py --prec fp8 --K_range 512 8192 --K_step 512 --MN-size 8192
mv matmul.hatchet matmul_M\=N\=8K_fp8.hatchet
proton-viewer -m "tflop8/s" matmul_M\=N\=8K_fp8.hatchet > matmul_M\=N\=8K_output_fp8.txt

# Run 4K fp8
python 09-persistent-matmul.py --prec fp8 --K_range 256 4096 --K_step 256 --MN-size 4096
mv matmul.hatchet matmul_M\=N\=4K_fp8.hatchet
proton-viewer -m "tflop8/s" matmul_M\=N\=4K_fp8.hatchet > matmul_M\=N\=4K_output_fp8.txt
