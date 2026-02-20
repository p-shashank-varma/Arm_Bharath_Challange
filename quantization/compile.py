mkdir -p compiled_model

vai_c_onnx \
    -m best_int8.onnx \
    -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/PYNQ-Z2/arch.json \
    -o ./compiled_model \
    -n smoke_fire_human \
    --options '{"input_shape": "1,3,320,320"}'
```

After it runs share the output — if successful you'll see:
```
compiler version...
[UNILOG][INFO] Compile succeed.
