import random


def _augment(x, hflip=True, vflip=True, rot90=True):
    if hflip:
        x = x.flip(-1)
    if vflip:
        x = x.flip(-2)
    if rot90:
        x = x.rot90(1, (1, 2))
    return x


def augment(*args, hflip=True, vflip=True, rot=True):
    h = hflip and random.random() > 0.5
    v = vflip and random.random() > 0.5
    r = rot and random.random() > 0.5
    return [_augment(x, hflip=h, vflip=v, rot90=r) for x in args]


def get_patch(x, *args, patch=(96,96,96)):

    patch_z,patch_y,patch_x = patch
    

    if patch is None:
        return x, *args

    def _sz(y):
        return tuple(y.shape[-3:])

    szx = _sz(x)

    if szx[0] <= patch_z:
        patch_z= szx[0]
    if szx[1] <= patch_y:
        patch_y = szx[1]
    if szx[2] <= patch_x:
        patch_x = szx[2]

    xi = random.randint(0, szx[2] - patch_x)
    xj = random.randint(0, szx[1] - patch_y)
    xk = random.randint(0, szx[0] - patch_z)

    def fn(y):
        szy = _sz(y)

        sk = szy[0] // szx[0]
        sj = szy[1] // szx[1]
        si = szy[2] // szx[2]

        pi = si * patch_x
        pj = sj * patch_y
        pk = sk * patch_z
        assert pi <= szy[2] and pj <= szy[1] and pk <= szy[0]

        yi = si * xi
        yj = sj * xj
        yk = sk * xk

        y = y[..., yk:yk+pk, yj:yj+pj, yi:yi+pi]

        return y

    return fn(x), *[fn(y) for y in args]
