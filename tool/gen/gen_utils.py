

def gen_helper(
    kind: str,
    process_func: callable,
    index_path,
    out_dir: Union[str, Path],
    inplace: bool = True,
    split: bool = True,
    split_files: bool = True,
    progress: bool = False,
    n_parallel: int = 1,
):
    if not split:
        raise NotImplementedError("--no-split")
    if not split_files:
        raise NotImplementedError("--no-split-files")
    if isinstance(out_dir, str):
        out_dir = Path(out_dir)
    assert isinstance(out_dir, Path)
    assert out_dir.is_dir()
    logger.info("Loading input %s", index_path)
    with open(index_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    candidates_data = yaml_data["candidates"]
    with ProcessPoolExecutor(n_parallel) as pool:
        futures = []
        for i, candidate_data in enumerate(candidates_data):
            future = pool.submit(process_func, i, candidate_data)
            futures.append(future)
        for future in tqdm(futures, disable=not progress):
            # TODO: except failing?
            out_file = future.result()
            yaml_data["candidates"][i]["artifacts"][kind]= str(out_file)
    if inplace:
        out_index_path = index_path
    else:
        out_index_path = out_dir / "index.yml"
    with open(out_index_path, "w") as f:  # TODO: reuse code from index.py
        yaml.dump(yaml_data, f, default_flow_style=False)

