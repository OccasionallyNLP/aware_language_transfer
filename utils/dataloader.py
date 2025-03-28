
def create_train_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    datasets = []
    data_config = train_data_config if split == "train" else val_data_config
    # check the validness
    for idx in range(len(data_config) - 1, -1, -1):
        prefix = data_config[idx][0]
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}-*")))        
        if len(filenames) < total_devices:
            fabric.print("skip dataset {}".format(prefix))
            del data_config[idx]
            continue

    for idx in range(len(data_config)):
        prefix = data_config[idx][0]
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}-*")))
        random.seed(seed)
        random.shuffle(filenames)
        fabric.print("create dataset {}".format(prefix))

        dataset = PackedDataset(
            filenames,
            # n_chunks control the buffer size. 
            # Note that the buffer size also impacts the random shuffle
            # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
            n_chunks=1,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed+fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)



def create_val_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    
    val_data_loaders = []
    
    # check the validness
    for idx in range(len(val_data_config) - 1, -1, -1):
        data_config = val_data_config[idx]
        delete_val_flag = False
        for prefix, _ in data_config:
            filenames = sorted(glob.glob(str(data_dir / f"{prefix}-*")))
            if len(filenames) < total_devices:
                fabric.print("skip val dataset {}".format(prefix))
                delete_val_flag = True
                break
        if delete_val_flag:
            del val_data_config[idx]


    for data_config in val_data_config:
        datasets = []
        for prefix, _ in data_config:
            filenames = sorted(glob.glob(str(data_dir / f"{prefix}-*")))
            random.seed(seed)
            random.shuffle(filenames)

            dataset = PackedDataset(
                filenames,
                # n_chunks control the buffer size. 
                # Note that the buffer size also impacts the random shuffle
                # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
                n_chunks=1,
                block_size=block_size,
                shuffle=shuffle,
                seed=seed+fabric.global_rank,
                num_processes=fabric.world_size,
                process_rank=fabric.global_rank,
            )
            datasets.append(dataset)

        if not datasets:
            raise RuntimeError(
                f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
            )

        weights = [weight for _, weight in data_config]
        sum_weights = sum(weights)
        weights = [el / sum_weights for el in weights]

        check_flag = True
        for dataset in datasets:
            if len(dataset._filenames) == 0:
                check_flag = False
                break
        
        if check_flag:
            combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)
            val_data_loaders.append(DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True))
            fabric.print("create val dataset {}".format(data_config))
        else:
            fabric.print("there are something wrong with the val dataset {}".format(data_config))
    return val_data_loaders


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_train_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train"
    )
    val_dataloader = (
        create_val_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation"
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader
