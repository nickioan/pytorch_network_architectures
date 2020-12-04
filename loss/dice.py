

import torch as th

import torch.nn as nn


class DiceLoss(nn.Module):

    r"""Criterion that computes Sørensen-Dice Coefficient loss.


    According to [1], we compute the Sørensen-Dice Coefficient as follows:​

    .. math::


        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}​

    where:

       - :math:`X` expects to be the scores of each class.

       - :math:`Y` expects to be the one-hot tensor with the class labels.


    the loss, is finally computed as:​

    .. math::


        \text{loss}(x, class) = 1 - \text{Dice}(x, class)​

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient


    Args:

        reduction (string, optional): Specifies the reduction to apply to the

            output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no

            reduction will be applied, ``'mean'``: the sum of the output will

            be divided by the number of elements in the output, ``'sum'``: the

            output will be summed. Default: ``'mean'``

        p (float, optional): denominator ||X||_p^p + ||Y||_p^p. Default: 2.0.

        smooth (float, optional): smoothing term to avoid overfitting.

            Default: 1.0.

        eps (float, optional): term added to the denominator to improve

            numerical stability. Default: 1e-8.


    Shape:

        - Input: :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 2`.

        - Target: :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 2`,

        - Output: scalar.

          If :attr:`reduction` is ``'none'``, then the same size as the target:

          :math:`(N)`.


    Examples:

        >>> N = 5  # num_classes

        >>> loss = DiceLoss()

        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)

        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)

        >>> output = loss(input, target)

        >>> output.backward()

    """


    def __init__(

            self,

            reduction: str = 'mean',

            p: float = 2.0,

            smooth: float = 1.0,

            eps: float = 1e-8) -> None:

        super(DiceLoss, self).__init__()

        self.reduction = reduction

        self.p = p

        self.smooth = smooth

        self.eps = eps


    def forward(self, input: th.Tensor, target: th.Tensor) -> th.Tensor:

        return dice_loss(input, target, reduction=self.reduction, p=self.p,

                         smooth=self.smooth, eps=self.eps)


def dice_loss(

        input: th.Tensor,

        target: th.Tensor,

        reduction: str = 'mean',

        p: float = 2.0,

        smooth: float = 1.0,

        eps: float = 1e-8) -> th.Tensor:

    r"""dice_loss(input, target, p=2.0, smooth=1.0, eps=1e-8) -> torch.Tensor


    Function that computes the Sørensen-Dice Coefficient loss.​

    See :class:`DiceLoss` for details.

    """


    if not th.is_tensor(input):

        raise TypeError("Expected torch.Tensor. Got {}".format(type(input)))

    dim = input.dim()

    if dim < 4:

        raise ValueError("invalid input size {} "

                         "(dim must be > 3)".format(input.size()))


    if input.size() != target.size():

        raise ValueError("input and target shapes must be the same. "

                         "Got: {} and {}" .format(input.size(), target.size()))


    reduction = reduction.lower()

    if reduction not in ['mean', 'sum', 'none']:

        raise ValueError("reduction must be one of 'mean', 'sum', 'none'. "

                         "Got {}".format(reduction))


    B = input.size()[0]

    input = input.contiguous().view(B, -1)

    target = target.contiguous().view(B, -1)


    # compute dice score

    intersection = th.sum(input * target, 1)

    union = input.pow(p).sum(1) + target.pow(p).sum(1)


    dice_score = (2. * intersection + smooth) / (union + smooth + eps)

    loss = 1. - dice_score

    if reduction == 'mean':

        return loss.mean()
    elif reduction == 'sum':

        return loss.sum()
    else:

        return loss

