import jax.numpy as jnp

from enf.steerable_attention.invariant._base_invariant import BaseInvariant


class PonitaPos2D(BaseInvariant):

    def __init__(self):
        super().__init__()

        # This invariant is calculated based on two sets of positional coordinates and orientations.
        self.num_x_pos_dims = 2  # This is always 2, the positional coordinates are 2D.
        self.num_x_ori_dims = 0
        self.num_z_pos_dims = 2  # This is always 2, the positional coordinates are 2D.
        self.num_z_ori_dims = 1

        # This invariant is 2D.
        self.dim = 2

    def __call__(self, x, p):
        """ Calculate the Ponita invariants between two sets of coordinates.
        Args:
            x (torch.Tensor): The pose of the input coordinates. Shape (batch_size, num_coords, 2).
            p (torch.Tensor): The pose of the latent points. Shape (batch_size, num_latents, num_ori, 2 (pos) + 2 (ori)).
        Returns:
            invariants (torch.Tensor): The Ponita invariants between the input and latent coordinates.
                Shape (batch_size, num_coords, num_latents, num_x_ori, num_z_ori, 2).
        """

        # Broadcast x over num_latents.
        x_pos = x[:, :, None, :]

        # Broadcast p over num_coords.
        p_pos, p_ori = p[:, None, :, :2], p[:, None, :, 2:]

        # Calculate relative positions between x and p.
        rel_pos = x_pos - p_pos

        # Calculate ponita invariants, shapes are [batch_size, num_coords, num_latents, num_x_ori, num_z_ori, 1].
        invariant1 = (rel_pos[..., 0] * p_ori[..., 0] + rel_pos[..., 1] * p_ori[..., 1])
        invariant2 = (-rel_pos[..., 0] * p_ori[..., 1] + rel_pos[..., 1] * p_ori[..., 0])
        invariants = jnp.stack([invariant1, invariant2], axis=-1)

        return invariants


class Ponita2D(BaseInvariant):

    def __init__(self):
        super().__init__()

        # This invariant is calculated based on two sets of positional coordinates and orientations.
        self.num_x_pos_dims = 2  # This is always 2, the positional coordinates are 2D.
        self.num_x_ori_dims = 1  # Full ponita invariant is calculated if the input points have orientation.
        self.num_z_pos_dims = 2  # This is always 2, the positional coordinates are 2D.
        self.num_z_ori_dims = 1

        # This invariant is 3D
        self.dim = 3

    # a(x,p)=ϕ(p−1x) 表示：先对 (x)做p的逆作用——也就是把x转到以p为参考的相对坐标，再通过某个函数 ϕ(⋅)做进一步嵌入
    # 在SE(2) 的情形下，一个元素p可以记作 (ppos, pori)，表示平移 t和旋转 Rθ。那么 p−1x  =  R−θp(xpos−ppos)以及相应的方向差 (θx−θp).
    def __call__(self, x, p):
        """ Calculate the Ponita invariants between two sets of coordinates.
        Args:
            x (torch.Tensor): The pose of the input coordinates. Shape (batch_size, num_coords, num_ori, 2 (pos) + 2 (ori)).
            p (torch.Tensor): The pose of the latent points. Shape (batch_size, num_latents, num_ori, 2 (pos) + 2 (ori)).
        Returns:
            invariants (torch.Tensor): The Ponita invariants between the input and latent coordinates.
                Shape (batch_size, num_coords, num_latents, num_x_ori, num_z_ori, 2).
        """

        # Broadcast x over num_latents and num_z_ori.
        x_pos, x_ori = x[:, :, None, :2], x[:, :, None, 2:]

        # Broadcast p over num_coords and num_x_ori.
        p_pos, p_ori = p[:, None, :, :2], p[:, None, :, 2:]

        # Calculate relative positions between x and p.
        rel_pos = x_pos - p_pos

        # Calculate ponita invariants, shapes are [batch_size, num_coords, num_latents, num_x_ori, num_z_ori, 1].
        invariant1 = (rel_pos[..., 0] * p_ori[..., 0] + rel_pos[..., 1] * p_ori[..., 1])
        invariant2 = (-rel_pos[..., 0] * p_ori[..., 1] + rel_pos[..., 1] * p_ori[..., 0]) # inv1 和 inv2 一起等价于在 pori​ 坐标系下描述相对位置 r
        invariant3 = (x_ori * p_ori).sum(axis=-1) # inv3=cos(θx​−θp​) 描述二者朝向的差异
        invariants = jnp.stack([invariant1, invariant2, invariant3], axis=-1)

        return invariants # a(x,p)​=(r⋅pori​,r x pori,xori​⋅pori​)
