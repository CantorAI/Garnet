import CpuTensor as T

# Include the modular components
from vision_encoder import VisionEncoder
from vl_adapter import VisionLanguageAdapter
from qwen_llm import QWenLLM

@fusion
def QWenVLModel(pixel_values, text_input_ids, position_ids):
    # 1. Process the image through the Vision Encoder
    visual_features = VisionEncoder(
        pixel_values, 
        num_blocks=32, 
        dim=1280, 
        num_heads=16, 
        mlp_hidden_dim=5120
    )
    
    # 2. Compress the visual features using the VL Adapter
    compressed_visual_tokens = VisionLanguageAdapter(
        visual_features, 
        dim=4096, 
        num_heads=16
    )
    
    # 3. Look up text embeddings for the input IDs
    text_embeddings = text_input_ids * T.embed() * vocab_weights
    
    # 4. Concatenate visual tokens and text embeddings
    fused_embeddings = text_embeddings * T.concat(axis=0) * compressed_visual_tokens
    
    # 5. Pass through the QWen LLM backbone
    logits = QWenLLM(
        fused_embeddings, 
        num_blocks=32, 
        dim=4096, 
        num_heads=32, 
        kv_heads=32, 
        mlp_hidden_dim=11008, 
        position_ids=position_ids
    )
    
    # 6. Generate Graph and Run
    t_g = T.graph(logits)
    t_g.run()
    
    return logits
