import math

from enf import EquivariantCrossAttentionENF
from enf.steerable_attention.invariant import get_sa_invariant, get_ca_invariant
from enf.latents.autodecoder import PositionOrientationFeatureAutodecoder
from enf.latents.autodecoder import PositionOrientationFeatureAutodecoderMeta


def get_model(cfg):
    """ Get autodecoders and snef based on the configuration. """

    # Determine whether we are doing meta-learning
    if "meta" not in cfg:
        # Init invariant
        self_attn_invariant = get_sa_invariant(cfg.nef)
        cross_attn_invariant = get_ca_invariant(cfg.nef)

        # Calculate initial gaussian window size
        assert math.sqrt(cfg.nef.num_latents)

        # Init autodecoder
        train_autodecoder = PositionOrientationFeatureAutodecoder(
            num_signals=cfg.dataset.num_signals_train,
            num_latents=cfg.nef.num_latents,
            latent_dim=cfg.nef.latent_dim,
            num_pos_dims=cross_attn_invariant.num_z_pos_dims,
            num_ori_dims=cross_attn_invariant.num_z_ori_dims,
            gaussian_window_size=cfg.nef.gaussian_window,
        )

        val_autodecoder = PositionOrientationFeatureAutodecoder(
            num_signals=cfg.dataset.num_signals_test,
            num_latents=cfg.nef.num_latents,
            latent_dim=cfg.nef.latent_dim,
            num_pos_dims=cross_attn_invariant.num_z_pos_dims,
            num_ori_dims=cross_attn_invariant.num_z_ori_dims,
            gaussian_window_size=cfg.nef.gaussian_window,
        )

        # Init model
        enf = EquivariantCrossAttentionENF(
            num_hidden=cfg.nef.num_hidden,
            num_heads=cfg.nef.num_heads,
            num_self_att_layers=cfg.nef.num_self_att_layers,
            num_out=cfg.nef.num_out,
            latent_dim=cfg.nef.latent_dim,
            self_attn_invariant=self_attn_invariant,
            cross_attn_invariant=cross_attn_invariant,
            embedding_type=cfg.nef.embedding_type,
            embedding_freq_multiplier=[cfg.nef.embedding_freq_multiplier_invariant,
                                       cfg.nef.embedding_freq_multiplier_value],
            condition_value_transform=cfg.nef.condition_value_transform,
            top_k_latent_sampling=cfg.nef.top_k,
        )
        return enf, train_autodecoder, val_autodecoder
    else:

        # Init invariant
        self_attn_invariant = get_sa_invariant(cfg.nef) # 返回的对象就是对输入坐标/姿态做相应的“等变或不变”特征变换的函数
        cross_attn_invariant = get_ca_invariant(cfg.nef) # 返回的具体是： BaseInvariant: The invariant module.

        # Calculate initial gaussian window size
        assert math.sqrt(cfg.nef.num_latents)

        # Init autodecoder
        # TODO:这里的两个decoder到底是都在内循环中用于分别获取隐变量表达的局部和全局形式？还是outerdecoder就已经用在外循环中可能被更新的一组全局潜变量？
        inner_autodecoder = PositionOrientationFeatureAutodecoderMeta( # 通常就是与当前任务/信号关联的隐变量𝑧
            num_signals=cfg.dataset.batch_size,
            # Since we're doing meta-learning, the inner and val autodecoders have batch_size as num signals
            num_latents=cfg.nef.num_latents,
            latent_dim=cfg.nef.latent_dim,
            num_pos_dims=cross_attn_invariant.num_z_pos_dims,
            num_ori_dims=cross_attn_invariant.num_z_ori_dims,
            gaussian_window_size=cfg.nef.gaussian_window,
        )
        # num_signals=1 表示它只维护“一套潜变量”——往往就是全局共享、跨所有任务都要用到或学习的那部分。
        # 在外循环中，它和 ENF 的参数 𝜃一起更新，扮演了“跨任务共享先验”或“外层全局潜变量” 的角色。
        # 它并不会在内循环里生成新的 z，也不参与对每条数据/每个任务的局部适配；而是和 𝜃一样，随着外循环的梯度一起被更新。？？？？
        outer_autodecoder = PositionOrientationFeatureAutodecoderMeta(
            num_signals=1,  # Since we're doing meta-learning, we only optimize one set of latents
            num_latents=cfg.nef.num_latents,
            latent_dim=cfg.nef.latent_dim,
            num_pos_dims=cross_attn_invariant.num_z_pos_dims,
            num_ori_dims=cross_attn_invariant.num_z_ori_dims,
            gaussian_window_size=cfg.nef.gaussian_window,
        )

        # Init model
        enf = EquivariantCrossAttentionENF(
            num_hidden=cfg.nef.num_hidden,
            num_heads=cfg.nef.num_heads,
            num_self_att_layers=cfg.nef.num_self_att_layers,
            num_out=cfg.nef.num_out,
            latent_dim=cfg.nef.latent_dim,
            self_attn_invariant=self_attn_invariant, # 使网络在执行注意力操作时，自动对坐标、姿态、条件进行正确的等变/不变处理；换句话说，每次网络前向中
            cross_attn_invariant=cross_attn_invariant, # 传入的(x,p,c)会先通过invariant函数得到不变表示，然后再进入q,k,v，从而保证了网络群作用的等变性
            embedding_type=cfg.nef.embedding_type,
            embedding_freq_multiplier=[cfg.nef.embedding_freq_multiplier_invariant,
                                       cfg.nef.embedding_freq_multiplier_value],
            condition_value_transform=cfg.nef.condition_value_transform,
            top_k_latent_sampling=cfg.nef.top_k,
        )
        return enf, inner_autodecoder, outer_autodecoder
