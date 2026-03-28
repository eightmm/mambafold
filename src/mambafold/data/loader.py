"""DataLoader utilities."""


def inf_loader(loader, sampler=None):
    """DataLoader를 무한 반복하는 제너레이터.

    DistributedSampler 사용 시 epoch마다 set_epoch()을 호출해 셔플링을 보장함.
    """
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1
