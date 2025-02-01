import torch
import json
import struct
import ctypes
import os
import zlib

# Define paths
pth_folder = "D:/llama3/Meta-Llama-3-8B"
pth_file = os.path.join(pth_folder, "consolidated.00.pth")

def save_weights_to_binary(model_path, output_file, compress=False):
    state_dict = torch.load(model_path)

    # Metadata includes the compression flag
    metadata = {
        "note": "Model weights",
        "compression_enabled": compress
    }

    tensor_info = []  # List to hold information about each tensor

    with open(output_file, 'wb') as f:
        # Write metadata with compression flag
        metadata_json = json.dumps(metadata)
        f.write(metadata_json.encode('utf-8'))
        f.write(b'\x00')

        for key, tensor in state_dict.items():
            tensor = tensor.cpu().contiguous()
            tensor_details = {
                "key": key,
                "data_type": str(tensor.dtype),
                "shape": list(tensor.shape),
                "number_of_elements": tensor.numel(),
                "element_size_bytes": tensor.element_size(),
                "total_size_bytes": tensor.numel() * tensor.element_size()
            }

            # Print tensor info for verification
            print(f"Key: {key}")
            print(f"  DataType: {tensor.dtype}")
            print(f"  Shape: {tensor.shape}")
            print(f"  Number of Elements: {tensor.numel()}")
            print(f"  Element Size (bytes): {tensor.element_size()}")
            print(f"  Total Size (bytes): {tensor.numel() * tensor.element_size()}")

            # Add tensor details to the list
            tensor_info.append(tensor_details)

            # Write the tensor key and dtype
            key_encoded = key.encode('utf-8')
            f.write(key_encoded)
            f.write(b'\x00')
            dtype_str = str(tensor.dtype).encode('utf-8')
            f.write(dtype_str)
            f.write(b'\x00')

            # Write the number of dimensions and tensor shape
            num_dims = len(tensor.shape)
            f.write(struct.pack('Q', num_dims))
            shape_encoded = struct.pack(f'{num_dims}Q', *tensor.shape)
            f.write(shape_encoded)

            # Conditionally compress tensor data based on the flag
            num_bytes = tensor.numel() * tensor.element_size()
            buffer = (ctypes.c_char * num_bytes).from_address(tensor.data_ptr())
            if compress:
                compressed_data = zlib.compress(buffer)
                data_to_write = compressed_data
                tensor_details["compressed_size_bytes"] = len(compressed_data)
                print(f"Compressed from {num_bytes} to {len(compressed_data)} bytes.")
            else:
                data_to_write = buffer

            # Write the actual tensor data
            f.write(data_to_write)

    # Set output path for the JSON file
    json_output_file = os.path.splitext(output_file)[0] + '_details.json'

    # Write tensor info to a JSON file
    with open(json_output_file, 'w') as f_json:
        json.dump(tensor_info, f_json, indent=4)

if __name__ == "__main__":
    binary_output_file = os.path.join(pth_folder, 'model_weights.bin')
    save_weights_to_binary(pth_file, binary_output_file)
    print("Done writing binary and JSON info.")
