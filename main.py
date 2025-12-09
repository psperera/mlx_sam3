import mlx.nn as nn
import mlx.core as mx
from mlx.utils import tree_map_with_path
from sam3.model_builder import build_sam3_image_model, _create_transformer_encoder, _create_transformer_decoder, _create_dot_product_scoring

def main():
    checkpoint_path = "/Users/deekshith/Documents/Projects/vision-models/mlx_sam3/sam3-mod-weights/model.safetensors"
    build_sam3_image_model(
        checkpoint_path=checkpoint_path,
    )

    # prompt_mask = mx.random.normal((1, 33)) > 0.5
    # model = _create_dot_product_scoring()
    # input = {
    #     "hs": mx.random.normal((6 ,1, 200, 256)),
    #     "prompt": mx.random.normal((33, 1, 256)),
    #     "prompt_mask": prompt_mask,
    # }

    # output = model(**input)




    
    
    
if __name__ == "__main__":
    main()
