from src.model import BeatGANsEncoderConfig, BeatGANsEncoderModel
from src.unet import EncoderUNetModel

encoder_cfg = BeatGANsEncoderConfig(
    image_size=256,
    in_channels=3,
    model_channels=128,
    out_hid_channels=512,
    out_channels=512,
    num_res_blocks=2,
    attention_resolutions=(16,),
    dropout=0.1,
    channel_mult=(1, 1, 2, 2, 4, 4, 4),
    use_time_condition=False,
    conv_resample=True,
    dims=2,
    use_checkpoint=False,
    num_heads=1,
    num_head_channels=-1,
    resblock_updown=True,
    use_new_attention_order=False,
    pool="adaptivenonzero",
)

diffae_encoder = BeatGANsEncoderModel(encoder_cfg)

openai_encoder = EncoderUNetModel(
    image_size=256,
    in_channels=3,
    model_channels=128,
    out_channels=512,
    num_res_blocks=2,
    attention_resolutions=(16,),
    dropout=0.1,
    channel_mult=(1, 1, 2, 2, 4, 4, 4),
    conv_resample=True,
    dims=2,
    use_checkpoint=False,
    num_heads=1,
    num_head_channels=-1,
    resblock_updown=True,
    use_new_attention_order=False,
    # Closest match to BeatGANs "adaptivenonzero" pool in EncoderUNetModel.
    pool="adaptive",
)
