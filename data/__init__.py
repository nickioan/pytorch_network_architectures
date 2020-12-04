from data.data import create_dataloader, create_dataset


def create_patch_scheduler(args):
    from .patch_scheduler import CyclePatchOnPlateau
    return CyclePatchOnPlateau(
        patches = args.patch,
        patience = args.patch_patience,
        patience_factor = args.patch_patience_factor,
        max_patience = args.patch_max_patience,
    )
