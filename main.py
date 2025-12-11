import mlx.nn as nn
import mlx.core as mx
from mlx.utils import tree_map_with_path
from sam3.model.geometry_encoders import Prompt
from sam3.model.maskformer_segmentation import PixelDecoder
from sam3.model_builder import build_sam3_image_model, _create_geometry_encoder, _create_transformer_decoder, _create_dot_product_scoring, _create_vision_backbone

def main():
    checkpoint_path = "/Users/deekshith/Documents/Projects/vision-models/mlx_sam3/sam3-mod-weights/model.safetensors"
    build_sam3_image_model(
        checkpoint_path=checkpoint_path,
    )

    # geometric_prompt = Prompt(
    #     box_embeddings=mx.zeros((0, 1, 4)),
    #     box_mask=mx.zeros((1,0), dtype=mx.bool_),
    # )
    # enc = _create_geometry_encoder()
    # inputs = {
    #     "geo_prompt": geometric_prompt,
    #     "img_feats": [mx.random.normal((5184, 1, 256))],
    #     "img_sizes": [mx.array([72, 72])],
    #     "img_pos_embeds": [mx.random.normal((5184, 1, 256))]
    # }
    # out = enc(**inputs)
    # breakpoint()

    

if __name__ == "__main__":
    main()
