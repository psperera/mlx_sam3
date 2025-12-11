import mlx.nn as nn

from sam3.model.vl_combiner import SAM3VLBackbone
from sam3.model.model_misc import MLP, DotProductScoring

class Sam3Image(nn.Module):
    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1
    TEXT_ID_FOR_GEOMETRIC = 2
    
    def __init__(
        self,
        backbone: SAM3VLBackbone,
        transformer,
        input_geometry_encoder,
        segmentation_head=None,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=None,
        use_instance_query: bool = True,
        multimask_otuput: bool = True,
        use_act_checkpoint_seg_head: bool = True,
        interactivity_in_encoder: bool = True,
        matcher=None,
        use_dot_prod_scoring=True,
        supervise_joint_box_scores: bool = False,
        detach_presence_in_joint_score: bool = False,
        separate_scorer_for_instance: bool = False,
        num_interactive_steps_val: int = 0,
        inst_interactive_predictor = None,
        **kwargs,
    ):
        super().__init__()
        self.backbone = backbone
        self.geometry_encoder = input_geometry_encoder
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.segmentation_head = segmentation_head

        self.o2m_mask_predict = o2m_mask_predict

        self.dot_prod_scoring = self.dot_prod_scoring
        self.use_act_checkpiont_seg_head = use_act_checkpoint_seg_head
        self.matcher = matcher

        self.num_interactive_steps_val = num_interactive_steps_val
        self.use_dot_prod_scoring = use_dot_prod_scoring

        if self.use_dot_prod_scoring:
            assert dot_prod_scoring is not None
            self.dot_prod_scoring = dot_prod_scoring
            self.instance_dot_prod_scoring = None
            if separate_scorer_for_instance:
                d_prompt_mlp = self.dot_prod_scoring.prompt_mlp
                prompt_mlp = MLP(
                    input_dim=d_prompt_mlp.input_dim,
                    hidden_dim=d_prompt_mlp.hidden_dim,
                    output_dim=d_prompt_mlp.output_dim,
                    num_layers=d_prompt_mlp.num_layers,
                    dropout=d_prompt_mlp.dropout,
                    residual=d_prompt_mlp.residual,
                    out_norm=nn.LayerNorm(256)
                )
                self.instance_dot_prod_scoring = DotProductScoring(
                    d_model=self.dot_prod_scoring.d_model,
                    d_proj=self.dot_prod_scoring.d_proj,
                    prompt_mlp=prompt_mlp
                )
        else:
            self.class_embed = nn.Linear(self.hidden_dim, 1)
            self.instance_class_embed = None
            if separate_scorer_for_instance:
                self.instance_class_embed = nn.Linear(self.hidden_dim, 1)
        
        self.supervise_joint_box_scores = supervise_joint_box_scores
        self.detach_presence_in_joint_score = detach_presence_in_joint_score

        # verify the number of queries for O2O and O2M
        num_o2o_static = self.transformer.decoder.num_queries
        num_o2m_static = self.transformer.decoder.num_o2m_queries
        assert num_o2m_static == (num_o2o_static if self.transformer.decoder.dac else 0)
        self.dac = self.transformer.decoder.dac

        self.use_instant_query = use_instance_query
        self.multimask_output = multimask_otuput

        self.inst_interactive_predictor = inst_interactive_predictor

    def _get_image_feats(self, backbone_out, img_ids):
        """ Retrieve correct image features from backbone output."""
        if "backbone_fpn" in backbone_out:
            if "id_mapping" in backbone_out and backbone_out["id_mapping"] is not None:
                img_ids = backbone_out["id_mapping"][img_ids]
                # If this assert fails, it likely means we're requesting different img_ids (perhaps a different frame?)
                # We currently don't expect this to happen. We could technically trigger a recompute here,
                # but likely at the cost of a cpu<->gpu sync point, which would deteriorate perf
                assert (img_ids >= 0).all()
            
            vis_feats = backbone_out

        

        
    
    def __call__(self):
        pass