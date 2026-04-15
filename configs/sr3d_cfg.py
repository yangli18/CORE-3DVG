use_color=True
data_root='DATA_ROOT/'
dataset=['sr3d']
test_dataset='sr3d'
detect_intermediate=True
log_dir='logs/gdetr/dec_match/sr3d/sr3d_pseudo_labass3_rel4_tdecl3sup14l2rel7v1l3_0_top24_sbert3_ptcorr0_dataaug0_4x3'
text_encoder_name_or_path="huggingface/roberta-base/"


extra_dataset_cfg=dict(
    text_endswith_not_mentioned=False,
    text_noun_span_file='sr3d_%s_text_span_230322.info',
    get_rand_utterance=True,

    sample_noun_texts=True,
    extract_noun_w_rand_space=False,

    anno_filter_rule = 'referit3d',
)

# Config 3D grounding model
ground_model=dict(
    type='CORE3DVG',
    init_query_build_method='sample_pt',
    extra_cfg=dict(
        ctx_decoder_input='decomp_text_feats',
        noun_span_type='default',
        bert_encode_decomp_text=None,

        decoder_s2_text_feat='proj_text_hs',
        bert_config=dict(
            num_hidden_layers=3,
        ),

        decoder_stage2_other_scores=['rel_pred_logits'],

        # Candidate selection
        candidate_select=dict(
            use_gpu_nms=True,
            topk_cand=24,
            topk_cand_by_nms=True,
            topk_cand_nms_iou=0.3,
            topk_cand_keep_best_match=True,
            topk_cand_keep_anchor=True,
            cand_matcher=dict(
                cost_class=0.5, cost_bbox=0, cost_dist=0.0,
                cost_giou=2.0,
            ),
        ),

        # To compute correlation scores of visual features and text features 
        # (for select candidate queries)
        v2t_corr_cfg=dict(
            type='Vis2TextCorrelationLayer',
            d_model=288, n_heads=8, dropout=0.1,
            vis_proj_args=dict(type='MLP', input_dim=288, hidden_dim=256, output_dim=256, num_layers=2),
            txt_proj_args=dict(type='MLP', input_dim=288, hidden_dim=256, output_dim=256, num_layers=2),
            norm_before_proj=False, inner_prod_w_bias=True,
            vis_query_with_pos=False,
        )
    )
)
pred_heads_in_decoder=True 
freeze_text_encoder=False

sample_pointcloud = dict(
    method='topk_corr_score',
    num=256,
    pred_head=dict(
        type='MLP', 
        input_dim=288, hidden_dim=288, output_dim=6, 
        num_layers=3
    )
)
decoder_extra_operation=dict(
    decoder_stage1_extra_ops=[
        dict(method='decode_noun_text'),
    ],
    decoder_stage2_extra_ops=[
        dict(method='decode_rand_text')
    ]
)


encoder=dict(
    type='TransformerEncoderLayer',
    d_model=288, dropout=0.1, n_heads=8, dim_feedforward=256
)

decoder_stage1=dict(
    type='DecoderWithPredHead',
    num_query=256, query_dim=288, 
    update_query_pos='xyz + offset, size',
    init_query_pos='sample_pt_boxes',
    init_query_src='sample_pt_feat',
    sample_pt_query_proj=dict(type='MLP', input_dim=288, hidden_dim=288, output_dim=288, num_layers=1),

    detach_query_pos_boxes=False,
    separate_query_pos_mlp=False,
    query_pos_mlp=dict(type='MLP',input_dim=6, hidden_dim=288, output_dim=288, num_layers=2),

    separate_pred_head=False,
    pred_head=dict(type='MLP', input_dim=288, hidden_dim=288, output_dim=6, num_layers=3),
    score_head=dict(type='MLP', input_dim=288, hidden_dim=288, output_dim=1, num_layers=3),

    num_layers=3,
    layer = dict(
        type='TransformerDecoderLayer2CrossAttn',
        d_model=288, n_heads=8, dim_feedforward=256, dropout=0.1,
        use_self_attn=True
    ), 
)


decoder_stage2=dict(
    type='DecoderBasedMatcherWithRelModel',
    d_model=288, 
    spatio_rel_feats_cfg=dict(
        args_to_build_spatio_rel_feats=dict(
            encode_distance='orig_dxdydz_distxy_dist', # 5-d
            encode_orientation='sinθ, cosθ, sinφ, cosφ', # 4-d
            encode_iou='3d_giou', # 1-d
            encode_size_compare='diff_vol_whd / avg_vol_whd', # 4-d
            encode_view_dependent=dict(
                compute_scene_center='(topk_max + topk_min) / 2',
                feat='ctx2center_as_yaxis',
                k=50,
            ), # 4-d
            encode_self=['tgt_dist2center_horizon'], # 1-d
        ),
        mlp_cfg=dict(type='MLP', input_dim=19, hidden_dim=144, output_dim=144,  num_layers=1),

        # dist_norm_expo=3.0, dist_norm_expo_learnable=True,
        # dxdydz_norm_expo=3.0, dxdydz_norm_expo_learnable=True,
    ),
    
    dec_layer_cfg=dict(
        type='TransformerDecoderLayerWithRelFeat',
        d_model=288, nhead=8, dim_feedforward=256,
        dropout=0.1, activation='relu', normalize_before=False,
        self_attn_cfg=dict(
            type='RelFeatsMHAttention',
            d_model=288, h=8, dropout=0.1, 
            q1_dim=288, q2_dim=288, k_dim=288, v_dim=288, rel_dim=288,
            attn_use_key=True, separate_q=True, v_use_rel_feats=True,
        )
    ),
    decoder_cfg=dict(
        type='TransformerDecoderWithRelFeat',
        num_layers=3,
        return_intermediate=True,
    ),

    score_pred_cfg=dict(type='MLP', input_dim=288, hidden_dim=288, output_dim=1, num_layers=2),
    query_pos_mlp=dict(type='MLP', input_dim=6, hidden_dim=288, output_dim=288,  num_layers=2),

    # -- for modeling rel_feats --
    ctx_feat_proj=dict(type='MLP', input_dim=288, hidden_dim=288, output_dim=144, num_layers=1),
    rel_feat_fusion='concat_ctx_feats',
    tgt_pos_mlp=dict(type='MLP', input_dim=6, hidden_dim=288, output_dim=288,  num_layers=2),
    ctx_pos_mlp=dict(type='MLP', input_dim=6, hidden_dim=288, output_dim=288,  num_layers=2),
    rel_feat_matcher=dict(
        type='TransformerDecoder',
        num_layers=2,
        return_intermediate=True,
        layer=dict(
            type='TransformerDecoderLayerForRelFeats',
            d_model=288, nhead=8, dim_feedforward=256,
            dropout=0.1, activation='relu', normalize_before=False,
            self_attn_cfg=dict(type='default', self_attn_method='full_attn')
        )
    ),
    rel_score_pred=dict(type='MLP', input_dim=288, hidden_dim=288, output_dim=1, num_layers=2),
    use_updated_rel_feat=True,
)

decoder=dict(decoder_stage1=decoder_stage1, decoder_stage2=decoder_stage2)
del decoder_stage1; del decoder_stage2


point_cloud_pos_embed=None
point_cloud_pos_mlp=dict(
    type='MLP',
    input_dim=3, hidden_dim=288, output_dim=288, 
    num_layers=2
)
